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

from lib.algorithms.pytorch.pruning import (FPGMPruner)
from lib.utils import *

import network
import utils
from torch.utils import data
from inplace_dict.deeplabv3plus import deeplabv3plus_inplace_dict as net_inplace_dict


def get_pruned_model(args, pruner, model, model_mask):
    prune_dict = {}
    mask_pt = torch.load(model_mask, map_location=args.device)

    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                module.weight_mask = mask_pt[name]['weight']
                module.bias_mask = mask_pt[name]['bias']
                prune_dict[module.name] = torch.mean(module.weight_mask, dim=(1, 2, 3)).int().cpu().numpy().tolist()
            else:
                pass

        pruner._unwrap_model()
        model = copy.deepcopy(pruner.bound_model)
        pruner._wrap_model()
        for name, module in model.named_modules():
            if name in net_inplace_dict:
                # print(name)
                # if name == 'classifier.aspp.convs.0.0':
                #     print(name)
                device = module.weight.device
                super_module, leaf_module = get_module_by_name(model, name)
                if type(module) == nn.BatchNorm2d:
                    mask = prune_dict[net_inplace_dict[name][0]]
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
                                input_mask += prune_dict[x]
                    output_mask = None if name not in prune_dict else prune_dict[name]
                    if input_mask is not None:
                        input_mask = torch.Tensor(input_mask).long().to(device)
                    if output_mask is not None:
                        output_mask = torch.Tensor(output_mask).long().to(device)
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


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtRandomCrop(size=(args.crop_size, args.crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    train_dst = Cityscapes(root=args.data_root,
                           split='train', transform=train_transform)
    val_dst = Cityscapes(root=args.data_root,
                         split='val', transform=val_transform)
    return train_dst, val_dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepLabV3Plus for Cityscapes')
    parser.add_argument('--model', default='deeplabv3plus_resnet101', type=str, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    parser.add_argument('--data_root', default='/home/lushun/dataset/cityscapes', type=str, help='dataset path')
    parser.add_argument('--index_start', default=0, type=int)
    parser.add_argument('--index_step', default=0, type=int)
    parser.add_argument('--index_split', default=5, type=int)
    parser.add_argument('--zoom_factor', default=8, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Prune options
    parser.add_argument('--pruner', default='fpgm', type=str, help='pruner: agp|taylor|fpgm')
    parser.add_argument('--sparsity', default=0.5, type=float, metavar='LR', help='prune sparsity')
    parser.add_argument('--finetune_epochs', default=100, type=int, metavar='N',
                        help='number of epochs for exported model')
    parser.add_argument('--finetune_lr', default=0.01, type=float, metavar='N', help='initial finetune learning rate')
    # Train options
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--val_batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--base_size', default=2048, type=int, )
    parser.add_argument("--crop_size", type=int, default=513)
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument('--prune_lr', default=0.01, type=float, metavar='LR', help='initial prune learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--power', default=0.9, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--tra_sample_rate', default=0.001, type=float, )
    parser.add_argument('--iteration', default=0, type=int, help='weight decay (default: 5e-4)')
    parser.add_argument('--max_iter', default=10e10, type=int, help='weight decay (default: 5e-4)')
    parser.add_argument('--epoch_num', default=100, type=int, help='weight decay (default: 5e-4)')
    parser.add_argument('--num_classes', default=19, type=int)
    parser.add_argument('--ignore_label', default=255, type=int)

    parser.add_argument('--print_freq', default=50, type=int, metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--ckpt', default='./checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar',
                        type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_dir', default='./exp_log/', help='The directory used to save the trained models',
                        type=str)
    parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int,
                        default=5)
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--baseline', action='store_true', default=False, help='evaluate model on validation set')
    parser.add_argument('--multiprocessing_distributed', action='store_true', default=False)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--ngpus_per_node', type=int, default=1)
    parser.add_argument('--prune_eval_path', default='', type=str, metavar='PATH', help='path to eval pruned model')

    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    best_prec1 = 0
    args.inputs_shape = (1, 3, 513, 513)
    args.save_dir = os.path.join(args.save_dir, args.model + '_' + args.pruner + '_' + str(args.sparsity))
    # create_exp_dir(args.save_dir, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    logging.info(args)
    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[args.model](num_classes=args.num_classes, output_stride=args.output_stride)
    if args.separable_conv and 'plus' in args.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # save_point = torch.load(
    #     '/Users/shunlu/Documents/code/model_compress/model_compress_v1/prune_seg/exp_log/seg_deeplab_efficiennetb3_agp_0.5_3/model_masked_3.pth',
    #     map_location=args.device
    # )
    # model.load_state_dict(save_point)
    print('************** Before Pruning **************')
    print("sum(params): params = {}M".format(sum(_param.numel() for _param in model.parameters()) / 1e6))

    # train_dst, val_dst = get_dataset(args)
    # train_loader = data.DataLoader(
    #     train_dst, batch_size=args.batch_size, shuffle=True, num_workers=2,
    #     drop_last=True)  # drop_last=True to ignore single-image batches.
    # val_loader = data.DataLoader(
    #     val_dst, batch_size=args.val_batch_size, shuffle=True, num_workers=2)
    # logging.info("Dataset: %s, Train set: %d, Val set: %d" %
    #              ('Cityscapes', len(train_dst), len(val_dst)))

    # # load pretrained weights
    # save_point = torch.load('./pretrained/model.pth', map_location=args.device)
    # model_param = save_point['state_dict']
    # model.load_state_dict(model_param)
    # model.eval()

    config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d'],
                    'op_names': ['backbone.conv1', 'backbone.conv1', 'backbone.layer1.0.conv1',
                                 'backbone.layer1.0.conv2',
                                 'backbone.layer1.0.conv3', 'backbone.layer1.0.downsample.0',
                                 'backbone.layer1.1.conv1', 'backbone.layer1.1.conv2',
                                 'backbone.layer1.1.conv3', 'backbone.layer1.2.conv1',
                                 'backbone.layer1.2.conv2', 'backbone.layer1.2.conv3',
                                 'backbone.layer2.0.conv1', 'backbone.layer2.0.conv2',
                                 'backbone.layer2.0.conv3', 'backbone.layer2.0.downsample.0',
                                 'backbone.layer2.1.conv1', 'backbone.layer2.1.conv2',
                                 'backbone.layer2.1.conv3', 'backbone.layer2.2.conv1',
                                 'backbone.layer2.2.conv2', 'backbone.layer2.2.conv3',
                                 'backbone.layer2.3.conv1', 'backbone.layer2.3.conv2',
                                 'backbone.layer2.3.conv3', 'backbone.layer3.0.conv1',
                                 'backbone.layer3.0.conv2', 'backbone.layer3.0.conv3',
                                 'backbone.layer3.0.downsample.0', 'backbone.layer3.1.conv1',
                                 'backbone.layer3.1.conv2', 'backbone.layer3.1.conv3',
                                 'backbone.layer3.2.conv1', 'backbone.layer3.2.conv2',
                                 'backbone.layer3.2.conv3', 'backbone.layer3.3.conv1',
                                 'backbone.layer3.3.conv2', 'backbone.layer3.3.conv3',
                                 'backbone.layer3.4.conv1', 'backbone.layer3.4.conv2',
                                 'backbone.layer3.4.conv3', 'backbone.layer3.5.conv1',
                                 'backbone.layer3.5.conv2', 'backbone.layer3.5.conv3',
                                 'backbone.layer3.6.conv1', 'backbone.layer3.6.conv2',
                                 'backbone.layer3.6.conv3', 'backbone.layer3.7.conv1',
                                 'backbone.layer3.7.conv2', 'backbone.layer3.7.conv3',
                                 'backbone.layer3.8.conv1', 'backbone.layer3.8.conv2',
                                 'backbone.layer3.8.conv3', 'backbone.layer3.9.conv1',
                                 'backbone.layer3.9.conv2', 'backbone.layer3.9.conv3',
                                 'backbone.layer3.10.conv1', 'backbone.layer3.10.conv2',
                                 'backbone.layer3.10.conv3', 'backbone.layer3.11.conv1',
                                 'backbone.layer3.11.conv2', 'backbone.layer3.11.conv3',
                                 'backbone.layer3.12.conv1', 'backbone.layer3.12.conv2',
                                 'backbone.layer3.12.conv3', 'backbone.layer3.13.conv1',
                                 'backbone.layer3.13.conv2', 'backbone.layer3.13.conv3',
                                 'backbone.layer3.14.conv1', 'backbone.layer3.14.conv2',
                                 'backbone.layer3.14.conv3', 'backbone.layer3.15.conv1',
                                 'backbone.layer3.15.conv2', 'backbone.layer3.15.conv3',
                                 'backbone.layer3.16.conv1', 'backbone.layer3.16.conv2',
                                 'backbone.layer3.16.conv3', 'backbone.layer3.17.conv1',
                                 'backbone.layer3.17.conv2', 'backbone.layer3.17.conv3',
                                 'backbone.layer3.18.conv1', 'backbone.layer3.18.conv2',
                                 'backbone.layer3.18.conv3', 'backbone.layer3.19.conv1',
                                 'backbone.layer3.19.conv2', 'backbone.layer3.19.conv3',
                                 'backbone.layer3.20.conv1', 'backbone.layer3.20.conv2',
                                 'backbone.layer3.20.conv3', 'backbone.layer3.21.conv1',
                                 'backbone.layer3.21.conv2', 'backbone.layer3.21.conv3',
                                 'backbone.layer3.22.conv1', 'backbone.layer3.22.conv2',
                                 'backbone.layer3.22.conv3', 'backbone.layer4.0.conv1',
                                 'backbone.layer4.0.conv2', 'backbone.layer4.0.conv3',
                                 'backbone.layer4.0.downsample.0', 'backbone.layer4.1.conv1',
                                 'backbone.layer4.1.conv2', 'backbone.layer4.1.conv3',
                                 'backbone.layer4.2.conv1', 'backbone.layer4.2.conv2',
                                 'backbone.layer4.2.conv3', ]
                    }]
    prune_optimizer = torch.optim.SGD(
        model.parameters(), lr=args.prune_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    pruner = FPGMPruner(
        model,
        config_list,
        prune_optimizer,
        dummy_input=torch.rand(size=args.inputs_shape).to(args.device),
    )

    model_mask = './exp_log/prune_deeplabv3plus_resnet101_fpgm_s0.3/deeplabv3plus_resnet101_mask.pth'
    model = get_pruned_model(args, pruner, model, model_mask)
    model.eval()

    print('************** After Pruning **************')
    print("sum(params): params = {}M".format(sum(_param.numel() for _param in model.parameters()) / 1e6))

    image = torch.randn(size=args.inputs_shape)
    image = image.to(args.device)
    model.to(args.device)
    with torch.no_grad():
        output = model.forward(image)
        print('Output size:', output.size())
