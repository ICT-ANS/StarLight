# from libs.compression.utils.counter import count_flops_params
import argparse
import copy
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import *
from inplace_dict.pspnet import pspnet_inplace_dict as net_inplace_dict


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion > max_iter:
        return init_lr

    lr = init_lr * (1 - iteraion / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_pruned_model(pruner, model, mask_pt):
    prune_dic = {}

    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                module.weight_mask = mask_pt[name]['weight']
                module.bias_mask = mask_pt[name]['bias']
                prune_dic[module.name] = torch.mean(module.weight_mask, dim=(1, 2, 3)).int().cpu().numpy().tolist()
                # if '_depthwise_conv' in name:
                #     record_weight_mask = module.weight_mask
            elif '_se_expand.conv' in name:
                depthwise_countpart = mask_pt[name.replace('_se_expand', '_depthwise_conv')]['weight']
                prune_dic[name] = torch.mean(depthwise_countpart, dim=(1, 2, 3)).int().cpu().numpy().tolist()
            else:
                pass
        # mask_pt = torch.load(masks_file, map_location='cpu')
        # for name, module in model.named_modules():
        #     if hasattr(module, 'weight_mask'):
        #         module.weight_mask = mask_pt[name]['weight']
        #         module.bias_mask = mask_pt[name]['bias']
        #
        # for name, module in model.named_modules():
        #     if hasattr(module, 'weight_mask'):
        #         prune_dic[module.name] = torch.mean(module.weight_mask, dim=(1, 2, 3)).int().cpu().numpy().tolist()
        #         if '_depthwise_conv' in name:
        #             record_weight_mask = module.weight_mask
        #         # if name == 'efficientnet_features._blocks.1._depthwise_conv.conv':
        #         #     break
        #     elif '_expand_conv.conv' in name:
        #         out_channels = module.out_channels
        #         prune_dic[name] = [1 for _ in range(out_channels)]
        #     elif '_se_expand.conv' in name:
        #         prune_dic[name] = torch.mean(record_weight_mask, dim=(1, 2, 3)).int().cpu().numpy().tolist()
        #     else:
        #         pass

        pruner._unwrap_model()
        model = copy.deepcopy(pruner.bound_model)
        pruner._wrap_model()
        for name, module in model.named_modules():
            if name in net_inplace_dict:
                # if name == 'conv1':
                #     print(name)
                device = module.weight.device
                super_module, leaf_module = get_module_by_name(model, name)
                if type(module) == nn.BatchNorm2d:
                    mask = prune_dic[net_inplace_dict[name][0]]
                    mask = torch.Tensor(mask).long().to(device)
                    compressed_module = replace_batchnorm2d(leaf_module, mask)
                if type(module) == nn.Conv2d:
                    if net_inplace_dict[name][0] == None:
                        input_mask = None
                    else:
                        input_mask = []
                        for x in net_inplace_dict[name]:
                            if type(x) is int:
                                input_mask += [1] * x
                            else:
                                input_mask += prune_dic[x]
                    output_mask = None if name not in prune_dic else prune_dic[name]
                    if input_mask is not None:
                        input_mask = torch.Tensor(input_mask).long().to(device)
                    if output_mask is not None:
                        output_mask = torch.Tensor(output_mask).long().to(device)
                    # if name == 'layer1.0.downsample.0':
                    #     print(name)
                    compressed_module = replace_conv2d(module, input_mask, output_mask)
                setattr(super_module, name.split('.')[-1], compressed_module)
        return model


def get_module_by_name(model, module_name):
    """
    Get a module specified by its module name

    Parameters
    ----------
    model : pytorch model
        the pytorch model from which to get its module
    module_name : str
        the name of the required module

    Returns
    -------
    module, module
        the parent module of the required module, the required module
    """
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    leaf_module = getattr(model, name_list[-1])
    return model, leaf_module


def get_index(mask):
    index = []
    for i in range(len(mask)):
        if mask[i] == 1:
            index.append(i)
    return torch.Tensor(index).long().to(mask.device)


def replace_batchnorm2d(norm, mask):
    """
    Parameters
    ----------
    norm : torch.nn.BatchNorm2d
        The batchnorm module to be replace
    mask : ModuleMasks
        The masks of this module

    Returns
    -------
    torch.nn.BatchNorm2d
        The new batchnorm module
    """
    index = get_index(mask)
    num_features = len(index)
    new_norm = torch.nn.BatchNorm2d(num_features=num_features, eps=norm.eps, momentum=norm.momentum, affine=norm.affine, track_running_stats=norm.track_running_stats)
    # assign weights
    new_norm.weight.data = torch.index_select(norm.weight.data, 0, index)
    new_norm.bias.data = torch.index_select(norm.bias.data, 0, index)
    if norm.track_running_stats:
        new_norm.running_mean.data = torch.index_select(norm.running_mean.data, 0, index)
        new_norm.running_var.data = torch.index_select(norm.running_var.data, 0, index)
    return new_norm


def replace_conv2d(conv, input_mask, output_mask):
    """
    Parameters
    ----------
    conv : torch.nn.Conv2d
        The conv2d module to be replaced
    mask : ModuleMasks
        The masks of this module

    Returns
    -------
    torch.nn.Conv2d
        The new conv2d module
    """
    if input_mask is None:
        in_channels = conv.in_channels
    else:
        in_channels_index = get_index(input_mask)
        in_channels = len(in_channels_index)
    if output_mask is None:
        out_channels = conv.out_channels
    else:
        out_channels_index = get_index(output_mask)
        out_channels = len(out_channels_index)

    if conv.groups != 1:
        new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride,
                                   padding=conv.padding,
                                   dilation=conv.dilation,
                                   groups=out_channels,
                                   bias=conv.bias is not None,
                                   padding_mode=conv.padding_mode)
    else:
        new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride,
                                   padding=conv.padding,
                                   dilation=conv.dilation,
                                   groups=conv.groups,
                                   bias=conv.bias is not None,
                                   padding_mode=conv.padding_mode)

    new_conv.to(conv.weight.device)
    tmp_weight_data = tmp_bias_data = None

    if output_mask is not None:
        tmp_weight_data = torch.index_select(conv.weight.data, 0, out_channels_index)
        if conv.bias is not None:
            tmp_bias_data = torch.index_select(conv.bias.data, 0, out_channels_index)
    else:
        tmp_weight_data = conv.weight.data
    # For the convolutional layers that have more than one group
    # we need to copy the weight group by group, because the input
    # channal is also divided into serveral groups and each group
    # filter may have different input channel indexes.
    input_step = int(conv.in_channels / conv.groups)
    in_channels_group = int(in_channels / conv.groups)
    filter_step = int(out_channels / conv.groups)
    if input_mask is not None:
        if new_conv.groups == out_channels:
            new_conv.weight.data.copy_(tmp_weight_data)
        else:
            for groupid in range(conv.groups):
                start = groupid * input_step
                end = (groupid + 1) * input_step
                current_input_index = list(filter(lambda x: start <= x and x < end, in_channels_index.tolist()))
                # shift the global index into the group index
                current_input_index = [x - start for x in current_input_index]
                # if the groups is larger than 1, the input channels of each
                # group should be pruned evenly.
                assert len(current_input_index) == in_channels_group, \
                    'Input channels of each group are not pruned evenly'
                current_input_index = torch.tensor(current_input_index).to(tmp_weight_data.device)  # pylint: disable=not-callable
                f_start = groupid * filter_step
                f_end = (groupid + 1) * filter_step
                new_conv.weight.data[f_start:f_end] = torch.index_select(tmp_weight_data[f_start:f_end], 1, current_input_index)
    else:
        new_conv.weight.data.copy_(tmp_weight_data)

    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data if tmp_bias_data is None else tmp_bias_data)

    return new_conv
