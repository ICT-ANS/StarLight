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

from dataReader.dataset import Reader
from lib.algorithms.pytorch.pruning import (FPGMPruner)
from lib.utils import *
from utils.utils import AverageMeter, get_evaluation_score
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


def train(epoch, model, train_loader, criterion, optimizer, args):
    model.train()
    losses = AverageMeter()
    gts_all, predictions_all = [], []
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.cuda()
        images = images.cuda()
        images = Variable(images)
        labels = Variable(labels)
        # Decaying Learning Rate
        lr = poly_lr_scheduler(optimizer, args.init_lr, args.iteration, args.max_iter, args.power)
        # Forward + Backward + Optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        #  record loss
        losses.update(loss.data)
        if random.random() < args.tra_sample_rate or i == 0:
            predictions = outputs.data.max(1)[1].cpu().numpy()
            gts_all.append(labels.data.cpu().numpy())
            predictions_all.append(predictions)
        args.lr = lr
        logging.info('[epoch %d],[iter %04d/%04d]:lr = %.9f,train_losses.avg = %.9f'
                     % (epoch, args.iteration % len(train_loader) + 1, len(train_loader), args.lr, losses.avg))
        args.iteration = args.iteration + 1
        # if args.iteration == 10:
        #     break
    tra_acc, tra_acc_cls, tra_miou, tra_fwavacc = get_evaluation_score(predictions_all, gts_all, args.num_classes)
    return losses.avg, tra_acc, tra_acc_cls, tra_miou, tra_fwavacc


def get_data_loader(args):
    print("\nloading dataset ...")
    train_data = Reader(args, mode='train')
    print("Train set samples: ", len(train_data))
    val_data = Reader(args, mode='test')
    print("Validation set samples: ", len(val_data))
    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              drop_last=True, num_workers=args.workers if torch.cuda.is_available() else 0)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            drop_last=False, num_workers=args.workers if torch.cuda.is_available() else 0)

    return train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
    parser.add_argument('--data', default='./dataset/mars_seg', type=str, help='dataset path')
    parser.add_argument('--model', default='seg_deeplab_efficiennetb3', type=str, help='model name')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--pruner', default='agp', type=str, help='pruner: agp|taylor|fpgm')
    parser.add_argument('--sparsity', default=0.5, type=float, metavar='LR', help='prune sparsity')
    parser.add_argument('--finetune_epochs', default=10, type=int, metavar='N',
                        help='number of epochs for exported model')
    parser.add_argument('--finetune_lr', default=0.001, type=float, metavar='N', help='number of lr for exported model')
    parser.add_argument('--finetune_momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--finetune_weight_decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 5e-4)')

    parser.add_argument('--batch_size', default=18, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.0, type=float, help='learning rate')
    parser.add_argument('--init_lr', default=2e-5, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--prune_lr', default=2e-5, type=float, metavar='LR', help='initial prune learning rate')

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--power', default=0.9, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--tra_sample_rate', default=0.001, type=float, )
    parser.add_argument('--iteration', default=0, type=int, help='weight decay (default: 5e-4)')
    parser.add_argument('--max_iter', default=10e10, type=int, help='weight decay (default: 5e-4)')
    parser.add_argument('--epoch_num', default=100, type=int, help='weight decay (default: 5e-4)')
    parser.add_argument('--num_classes', default=6, type=int, help='weight decay (default: 5e-4)')

    parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='./checkpoint/resnet50.pth', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_dir', default='./exp_log/', help='The directory used to save the trained models',
                        type=str)
    parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int,
                        default=1)
    parser.add_argument('--seed', type=int, default=2, help='random seed')

    parser.add_argument('--baseline', action='store_true', default=False, help='evaluate model on validation set')
    parser.add_argument('--prune_eval_path', default='', type=str, metavar='PATH', help='path to eval pruned model')

    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    best_prec1 = 0
    args.inputs_shape = (1, 3, 512, 512)
    args.save_dir = os.path.join(args.save_dir, args.model + '_' + args.pruner + '_' + str(args.sparsity))
    # create_exp_dir(args.save_dir, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logging.info(args)

    from prune_seg.models.PSPNet.pspnet import PSPNet

    model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=args.num_classes, zoom_factor=8, use_ppm=True,
                   pretrained=False)
    # save_point = torch.load(
    #     '/Users/shunlu/Documents/code/model_compress/model_compress_v1/prune_seg/exp_log/seg_deeplab_efficiennetb3_agp_0.5_3/model_masked_3.pth',
    #     map_location=args.device
    # )
    # model.load_state_dict(save_point)
    print('************** Before Pruning **************')
    print("sum(params): params = {}M".format(sum(_param.numel() for _param in model.parameters()) / 1e6))


    train_loader, val_loader = get_data_loader(args)
    # # load pretrained weights
    # save_point = torch.load('./pretrained/model.pth', map_location=args.device)
    # model_param = save_point['state_dict']
    # model.load_state_dict(model_param)
    # model.eval()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr,
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(args.device)
    def trainer(model, optimizer, criterion, epoch):
        result = train(epoch, model, train_loader, criterion, optimizer, args)
        return result

    config_list = [{'sparsity': args.sparsity,
                        'op_types': ['Conv2d'],
                        'op_names':
                        ['layer0.0', 'layer0.3', 'layer0.6', 'layer1.0', 'layer1.0.conv1', 'layer1.0.conv2',
                         'layer1.0.conv3', 'layer1.0.downsample', 'layer1.0.downsample.0', 'layer1.1',
                         'layer1.1.conv1', 'layer1.1.conv2', 'layer1.1.conv3', 'layer1.2', 'layer1.2.conv1',
                         'layer1.2.conv2', 'layer1.2.conv3', 'layer2.0', 'layer2.0.conv1', 'layer2.0.conv2',
                         'layer2.0.conv3', 'layer2.0.downsample', 'layer2.0.downsample.0', 'layer2.1',
                         'layer2.1.conv1', 'layer2.1.conv2', 'layer2.1.conv3', 'layer2.2', 'layer2.2.conv1',
                         'layer2.2.conv2', 'layer2.2.conv3', 'layer2.3', 'layer2.3.conv1', 'layer2.3.conv2',
                         'layer2.3.conv3', 'layer3.0', 'layer3.0.conv1', 'layer3.0.conv2', 'layer3.0.conv3',
                         'layer3.0.downsample', 'layer3.0.downsample.0', 'layer3.1', 'layer3.1.conv1',
                         'layer3.1.conv2', 'layer3.1.conv3', 'layer3.2', 'layer3.2.conv1',
                         'layer3.2.conv2', 'layer3.2.conv3', 'layer3.3', 'layer3.3.conv1',
                         'layer3.3.conv2', 'layer3.3.conv3', 'layer3.4', 'layer3.4.conv1',
                         'layer3.4.conv2', 'layer3.4.conv3', 'layer3.5', 'layer3.5.conv1',
                         'layer3.5.conv2', 'layer3.5.conv3', 'layer4.0', 'layer4.0.conv1',
                         'layer4.0.conv2', 'layer4.0.conv3', 'layer4.0.downsample',
                         'layer4.0.downsample.0', 'layer4.1', 'layer4.1.conv1', 'layer4.1.conv2',
                         'layer4.1.conv3', 'layer4.2', 'layer4.2.conv1', 'layer4.2.conv2', 'layer4.2.conv3',
                         'ppm.features', 'ppm.features.0', 'ppm.features.0.1', 'ppm.features.1', 'ppm.features.1.1',
                         'ppm.features.2', 'ppm.features.2.1', 'ppm.features.3', 'ppm.features.3.1',
                         'cls.0', 'cls.4']
                        }]
    prune_optimizer = optim.Adam(model.parameters(), lr=args.prune_lr,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    pruner = FPGMPruner(
        model,
        config_list,
        prune_optimizer,
        dummy_input=torch.rand(size=args.inputs_shape).to(args.device),
    )
    mask_pt = torch.load(
        './exp_log/pspnet_fpgm_0.5_p2e-05_f2e-05/pspnet_mask.pth',
        map_location=args.device
    )

    # fix downsample mask be the same with the lateral conv
    key_list = [key for key in mask_pt.keys()]
    for key in key_list:
        if 'downsample' in key:
            conv_key = key.split('downsample')[0] + 'conv3'
            mask_pt[key] = mask_pt[conv_key]

    model = get_pruned_model(pruner, model, mask_pt)
    model.eval()

    print('************** After Pruning **************')
    print("sum(params): params = {}M".format(sum(_param.numel() for _param in model.parameters()) / 1e6))

    image = torch.randn(size=args.inputs_shape)
    image = image.to(args.device)
    model.to(args.device)
    with torch.no_grad():
        output = model.forward(image)
        print('Output size:', output.size())
