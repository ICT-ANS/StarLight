import sys

sys.path.append('..')

import torch.nn as nn
import torch.utils.data as data

import torchvision.models as models
import argparse
import sys
import logging

from utils import *
from lib.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT
from export_utils_pspnet import get_pruned_model
from semseg.tool.test_quan import evaluate_quan


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--model', default='pspnet', type=str, help='model name')
parser.add_argument('--sparsity', default=0.0, type=float, help='only for models with sparsity')
parser.add_argument('--data_root', default='./dataset/cityscapes', type=str, help='dataset path')
parser.add_argument('--train_list', default='./dataset/cityscapes/cityscapes_train_list.txt', type=str)
parser.add_argument('--test_list', default='./dataset/cityscapes/cityscapes_val_list.txt', type=str)
parser.add_argument('--pretrained', default='./pretrained/train_epoch_200.pth', type=str)
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--index_start', default=0, type=int)
parser.add_argument('--index_step', default=0, type=int)
parser.add_argument('--index_split', default=5, type=int)
parser.add_argument('--zoom_factor', default=8, type=int)
parser.add_argument('--colors_path', default='./semseg/data/cityscapes/cityscapes_colors.txt', type=str)
parser.add_argument('--names_path', default='./semseg/data/cityscapes/cityscapes_names.txt', type=str)
parser.add_argument('--batch_size', default=2, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--base_size', default=2048, type=int, )

parser.add_argument('--print_freq', default=1000, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--save_dir', default='./exp_log/', help='The directory used to save the trained models', type=str)
parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=1)
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--num_classes', default=19, type=int, help='weight decay (default: 5e-4)')

parser.add_argument('--prune_eval_path', default=None, type=str, metavar='PATH', help='path to eval pruned model')
parser.add_argument('--pruner', default='agp', type=str, help='pruner: agp|taylor|fpgm')

parser.add_argument('--quan_mode', default='fp32', help='fp16 int8 best', type=str)
parser.add_argument('--calib_num', type=int, default=100, help='calibrating number')

args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

best_prec1 = 0

args.save_dir += 'quan_%s' % args.model
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)


def main():
    global args, best_prec1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    cudnn.deterministic = True

    # inputs_shape = (args.batch_size, 3, 512, 512)
    args.inputs_shape = (args.batch_size, 3, 713, 713)

    model = get_model(args.model).cuda()

    # x = torch.rand(size=inputs_shape).to(device)
    model.eval()
    # with torch.no_grad():
    #     elapse = time.time()
    #     torch_out = model(x)
    #     elapse = time.time() - elapse
    # print('inputs_shape: %s, output_shape: %s, elapse: %s' % (x.shape, torch_out.shape, elapse))

    train_loader, test_loader, calib_loader, train_data, test_data, mean, std = get_data_loader(args)
    colors = np.loadtxt(args.colors_path).astype('uint8')
    names = [line.rstrip('\n') for line in open(args.names_path)]

    criterion = nn.CrossEntropyLoss(ignore_index=255).to(args.device)
    gray_folder = os.path.join(args.save_dir, '{}_s{}_{}_before_prune_gray'.format(args.model, args.sparsity, args.quan_mode))
    color_folder = os.path.join(args.save_dir, '{}_s{}_{}_before_prune_color'.format(args.model, args.sparsity, args.quan_mode))
    logging.info('Before Quantization !')
    evaluate_quan(test_loader, test_data.data_list, model, args.num_classes, mean, std, args.base_size, 713, 713,
                  [1.0], gray_folder, color_folder, colors, names, is_trt=False)
    print('\n')

    if 'sparse' in args.model:
        onnx_path = os.path.join(args.save_dir, '{}_s{}_{}.onnx'.format(args.model, args.sparsity, args.quan_mode))
        trt_path = os.path.join(args.save_dir, '{}_s{}_{}.trt'.format(args.model, args.sparsity, args.quan_mode))
        cache_path = os.path.join(args.save_dir, '{}_s{}_{}.cache'.format(args.model, args.sparsity, args.quan_mode))
    else:
        onnx_path = os.path.join(args.save_dir, '{}_{}.onnx'.format(args.model, args.quan_mode))
        trt_path = os.path.join(args.save_dir, '{}_{}.trt'.format(args.model, args.quan_mode))
        cache_path = os.path.join(args.save_dir, '{}_{}.cache'.format(args.model, args.quan_mode))

    if args.quan_mode == "int8":
        extra_layer_bit = 8
    elif args.quan_mode == "fp16":
        extra_layer_bit = 16
    elif args.quan_mode == "best":
        extra_layer_bit = -1
    else:
        extra_layer_bit = 32

    engine = ModelSpeedupTensorRT(
        model,
        args.inputs_shape,
        config=None,
        calib_data_loader=calib_loader,
        batchsize=args.inputs_shape[0],  # error
        onnx_path=onnx_path,
        calibration_cache=cache_path,
        extra_layer_bit=extra_layer_bit,
    )
    if not os.path.exists(trt_path):
        engine.compress()
        engine.export_quantized_model(trt_path)
    else:
        engine.load_quantized_model(trt_path)
        # engine = common.load_engine(trt_path)
        logging.info('Directly load quantized model from %s' % trt_path)

    logging.info('After Quantization !')
    gray_folder = os.path.join(args.save_dir, '{}_s{}_{}_after_prune_gray'.format(args.model, args.sparsity, args.quan_mode))
    color_folder = os.path.join(args.save_dir, '{}_s{}_{}_after_prune_color'.format(args.model, args.sparsity, args.quan_mode))
    evaluate_quan(test_loader, test_data.data_list, engine, args.num_classes, mean, std, args.base_size, 713, 713,
                  [1.0], gray_folder, color_folder, colors, names, is_trt=True)


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
        # pretrained_model_path = './semseg-master/exp/cityscapes/pspnet50/model/train_epoch_200.pth'
        pretrained_model_path = args.pretrained
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

    elif 'pspnet_sparse' in model_name:
        # Define Model
        from models.PSPNet.pspnet import PSPNet
        model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=args.num_classes,
                       zoom_factor=8, use_ppm=True, pretrained=False).to(args.device)

        # Acquire Mask
        sparsity_dir = './exp_log/pspnet_fpgm_s%s_p0.01_f0.01/' % args.sparsity
        model_mask = os.path.join(sparsity_dir, 'pspnet_mask.pth')
        mask_pt = torch.load(model_mask, map_location=args.device)
        # Fix downsample mask be the same with the lateral conv
        key_list = [key for key in mask_pt.keys()]
        for key in key_list:
            if 'downsample' in key:
                conv_key = key.split('downsample')[0] + 'conv3'
                mask_pt[key] = mask_pt[conv_key]

        # Define Pruner
        from lib.algorithms.pytorch.pruning import FPGMPruner
        prune_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
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

        # Prune Model
        model = get_pruned_model(pruner, model, mask_pt)
        pruned_weights = torch.load(os.path.join(sparsity_dir, 'pspnet_s%s_pruned_best_miou.pth' % args.sparsity))
        model.load_state_dict(pruned_weights['state_dict'])
        logging.info('Load pruned models from %s done!' % sparsity_dir)

    else:
        raise NotImplementedError('Not Support {}'.format(model_name))

    return model


def get_data_loader(args):
    from semseg.util import transform, dataset
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

    n_train = len(train_data)
    indices = list(range(n_train))
    random.shuffle(indices)
    calib_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:args.calib_num])
    calib_loader = data.DataLoader(train_data, batch_size=args.batch_size, sampler=calib_sampler)

    return train_loader, test_loader, calib_loader, train_data, test_data, mean, std


if __name__ == "__main__":
    main()
