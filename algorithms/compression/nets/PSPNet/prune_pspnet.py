import sys

sys.path.append('.')
sys.path.append('..')
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import argparse
import sys
import logging
from torch.utils.data import DataLoader

import glob
import copy
from utils import *

from lib.algorithms.pytorch.pruning import (FPGMPruner, AGPPruner)
from semseg.util import transform, dataset
from semseg.tool.test import evaluate
from semseg.tool.train import train_pruned_model
from export_utils_pspnet import get_pruned_model

parser = argparse.ArgumentParser(description='PSPNet50 for Cityscapes')
parser.add_argument('--model', default='pspnet', type=str, help='model name')
parser.add_argument('--data_root', default='./dataset/cityscapes', type=str, help='dataset path')
parser.add_argument('--train_list', default='./dataset/cityscapes/cityscapes_train_list.txt', type=str)
parser.add_argument('--test_list', default='./dataset/cityscapes/cityscapes_val_list.txt', type=str)
parser.add_argument('--index_start', default=0, type=int)
parser.add_argument('--index_step', default=0, type=int)
parser.add_argument('--index_split', default=5, type=int)
parser.add_argument('--zoom_factor', default=8, type=int)
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

parser.add_argument('--pruner', default='fpgm', type=str, help='pruner: agp|taylor|fpgm')
parser.add_argument('--sparsity', default=0.5, type=float, metavar='LR', help='prune sparsity')
parser.add_argument('--finetune_epochs', default=100, type=int, metavar='N', help='number of epochs for exported model')
parser.add_argument('--finetune_lr', default=0.01, type=float, metavar='N', help='initial finetune learning rate')

parser.add_argument('--batch_size', default=6, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--base_size', default=2048, type=int, )
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
parser.add_argument('--resume', default='./checkpoint/resnet50.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', default='./exp_log/', help='The directory used to save the trained models', type=str)
parser.add_argument('--colors_path', default='./semseg/data/cityscapes/cityscapes_colors.txt', type=str)
parser.add_argument('--names_path', default='./semseg/data/cityscapes/cityscapes_names.txt', type=str)
parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=5)
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--baseline', action='store_true', default=False, help='evaluate model on validation set')
parser.add_argument('--multiprocessing_distributed', action='store_true', default=False)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--ngpus_per_node', type=int, default=1)
parser.add_argument('--prune_eval_path', default='', type=str, metavar='PATH', help='path to eval pruned model')

args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

best_prec1 = 0
args.inputs_shape = (1, 3, 713, 713)
args.save_dir = os.path.join(
    args.save_dir, '%s_%s_s%s_p%s_f%s' % (args.model, args.pruner, args.sparsity, args.prune_lr, args.finetune_lr)
)
# if os.path.exists(args.save_dir):
#     raise ValueError('Repeated save dir !!!')

create_exp_dir(args.save_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')


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


def compute_Params_FLOPs(model, device):
    from thop import profile
    model.eval()
    inputs = torch.rand(size=(2, 3, 713, 713)).to(device)
    flops, params = profile(model, inputs=(inputs,), verbose=False)
    print("Thop: params = {}M, flops = {}M".format(params / 1e6, flops / 1e6))
    print("sum(params): params = {}M".format(sum(_param.numel() for _param in model.parameters()) / 1e6))


def main():
    train_loader, test_loader, train_data, test_data, mean, std = get_data_loader(args)
    colors = np.loadtxt(args.colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.names_path)]

    args.max_iter = len(train_loader) * args.epoch_num
    logging.info(args)

    model = get_model(args.model).to(args.device)
    logging.info("[Before Pruning] sum Param: %f" % (float(sum(_param.numel() for _param in model.parameters())) / 1e6))
    compute_Params_FLOPs(copy.deepcopy(model), device=args.device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255).to(args.device)
    prune_optimizer = torch.optim.SGD(
        model.parameters(), lr=args.prune_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # evaluate before pruning
    gray_folder = os.path.join(args.save_dir, 'before_prune_gray')
    color_folder = os.path.join(args.save_dir, 'before_prune_color')
    evaluate(test_loader, test_data.data_list, model, args.num_classes, mean, std, args.base_size, 713, 713,
             [1.0], gray_folder, color_folder, colors, names)

    def trainer(model, optimizer, criterion, epoch):
        result = train_pruned_model(args, train_loader, model, optimizer, epoch, criterion)
        return result

    # get pruner, agp|taylor|fpgm
    if args.pruner == 'agp':
        config_list = [{'sparsity': args.sparsity,
                        'op_types': ['Conv2d'],
                        }]
        pruner = AGPPruner(
            model,
            config_list,
            prune_optimizer,
            trainer,
            criterion,
            num_iterations=1,
            epochs_per_iteration=1,
            pruning_algorithm='taylorfo',
        )
    elif args.pruner == 'fpgm':
        config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d'],
                        'op_names': ['layer0.0', 'layer0.3', 'layer0.6', 'layer1.0', 'layer1.0.conv1', 'layer1.0.conv2',
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
                                     'ppm.features', 'ppm.features.0', 'ppm.features.0.1', 'ppm.features.1',
                                     'ppm.features.1.1',
                                     'ppm.features.2', 'ppm.features.2.1', 'ppm.features.3', 'ppm.features.3.1',
                                     'cls.0']}]
        pruner = FPGMPruner(
            model,
            config_list,
            prune_optimizer,
            dummy_input=torch.rand(size=args.inputs_shape).to(args.device),
        )
    else:
        raise NotImplementedError

    logging.info('Start pruning ...')
    pruner.compress()

    # save masked pruned model (model + mask)
    model_weights_with_mask = os.path.join(args.save_dir, '%s_with_mask.pth' % args.model)
    model_mask = os.path.join(args.save_dir, '%s_mask.pth' % args.model)
    pruner.export_model(model_weights_with_mask, model_mask)

    # Export pruned model
    logging.info('Speed up masked model ...')
    mask_pt = torch.load(model_mask, map_location=args.device)
    # fix downsample mask be the same with the lateral conv
    key_list = [key for key in mask_pt.keys()]
    for key in key_list:
        if 'downsample' in key:
            conv_key = key.split('downsample')[0] + 'conv3'
            mask_pt[key] = mask_pt[conv_key]
    model = get_pruned_model(pruner, model, mask_pt)
    logging.info("[After Pruning] sum Param: %f" % (float(sum(_param.numel() for _param in model.parameters())) / 1e6))
    compute_Params_FLOPs(copy.deepcopy(model), device=args.device)

    # evaluate after pruning
    gray_folder = os.path.join(args.save_dir, 'after_prune_gray')
    color_folder = os.path.join(args.save_dir, 'after_prune_color')
    evaluate(test_loader, test_data.data_list, model, args.num_classes, mean, std, args.base_size, 713, 713,
             [1.0], gray_folder, color_folder, colors, names)

    # finetune pruned model
    logging.info('Finetuning export pruned model...')
    finetune_optimizer = torch.optim.SGD(
        model.parameters(), lr=args.finetune_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    best_miou = 0.0
    args.iteration = 0
    for epoch in range(args.finetune_epochs):
        loss_train, mIoU_train, mAcc_train, allAcc_train = \
            train_pruned_model(args, train_loader, model, finetune_optimizer, epoch, criterion)
        logging.info(
            "Epoch[%d]\t train_loss:%.4f\t mIoU_train:%.4f\t mAcc_train:%.4f\t allAcc_train:%.4f"
            % (epoch, loss_train, mIoU_train, mAcc_train, allAcc_train)
        )

        test_miou, mAcc, allAcc = evaluate(test_loader, test_data.data_list, model, args.num_classes, mean, std,
                                           args.base_size, 713, 713, [1.0], gray_folder, color_folder, colors, names)
        if best_miou < test_miou:
            best_miou = test_miou
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'miou': best_miou,
            }, os.path.join(args.save_dir, '%s_s%s_pruned_best_miou.pth' % (args.model, args.sparsity)))


def get_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 200)
    elif model_name == 'resnet101':
        model = models.resnet101()
        model.fc = nn.Linear(model.fc.in_features, 200)
    elif model_name == 'pspnet':
        from models.PSPNet.pspnet import PSPNet
        model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=args.num_classes, zoom_factor=8, use_ppm=True,
                       pretrained=False)
        pretrained_model_path = './semseg-master/exp/cityscapes/pspnet50/model/train_epoch_200.pth'
        checkpoint = torch.load(pretrained_model_path, map_location=args.device)

        # change keys format
        model_weights = {}
        ckpt_keys = checkpoint['state_dict'].keys()
        for _key in ckpt_keys:
            if 'aux' in _key:
                continue
            elif 'module' in _key:
                model_weights[_key[7:]] = checkpoint['state_dict'][_key]

        model.load_state_dict(model_weights, strict=True)
        logging.info("=> Load checkpoint '{}' done!".format(pretrained_model_path))
    else:
        raise NotImplementedError('Not Support {}'.format(model_name))
    return model


def get_data_loader(args):
    # load Cityscapes train_loader
    num_workers = 0
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    train_transform = transform.Compose([
        transform.RandScale([0.5, 2.0]),
        transform.RandRotate([-10, 10], padding=mean, ignore_label=255),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([713, 713], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])
    train_data = dataset.SemData(
        split='train', data_root=args.data_root, data_list=args.train_list, transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=4, shuffle=True,
        num_workers=num_workers, pin_memory=True, sampler=None, drop_last=True
    )

    # load Cityscapes test_loader
    test_transform = transform.Compose([transform.ToTensor()])
    test_data = dataset.SemData(
        split='val', data_root=args.data_root, data_list=args.test_list, transform=test_transform
    )
    index_start = args.index_start
    if args.index_step == 0:
        index_end = len(test_data.data_list)
    else:
        index_end = min(index_start + args.index_step, len(test_data.data_list))
    test_data.data_list = test_data.data_list[index_start:index_end]
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    logging.info('Successfully load cityscapes dataset!!! trainset=%s, train_batches=%s, testset=%s, test_batches=%s'
                 % (len(train_data), len(train_loader), len(test_data), len(test_loader)))

    return train_loader, test_loader, train_data, test_data, mean, std


if __name__ == "__main__":
    main()
