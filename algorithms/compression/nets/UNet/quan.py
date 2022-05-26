import argparse
import logging
import os
import sys
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval import eval_net
from unet import UNet
from torch.utils.tensorboard import SummaryWriter

from utils.mars_dataset import MarsDataset
from utils.mars_img_dataset import MarsImgDataset
from torch.utils.data import DataLoader, random_split, Subset

from infer import img_predict, count_flops_params
from lib.compression.pytorch import ModelSpeedup
from lib.algorithms.pytorch.pruning import (TaylorFOWeightFilterPruner, FPGMPruner, AGPPruner)

from lib.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dir_checkpoint = 'virtual_checkpoints/'
root_dir = 'Pytorch-UNet-master/data/data/train/'
mark_name = 'data'


class WrapDataloader(MarsImgDataset):
    def __getitem__(self, i):
        data = super().__getitem__(i)
        return data['image'], data['label']


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-5, help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default="checkpoints/origin.pth", help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--sparsity', default=0.5, type=float, metavar='LR', help='prune sparsity')

    parser.add_argument('--save_dir', type=str, default='logs/quan/test', help='model name')

    # for quan
    parser.add_argument('--baseline', action='store_true', default=False, help='evaluate model on validation set')
    parser.add_argument('--pruner', default='fpgm', type=str, help='pruner: agp|taylor|fpgm')
    parser.add_argument('--prune_eval_path', default=None, type=str, metavar='PATH', help='path to eval pruned model')
    parser.add_argument('--quan_mode', default='int8', help='fp16 int8 best', type=str)
    parser.add_argument('--calib_num', type=int, default=1280, help='random seed')

    parser.add_argument('--data_path', type=str, default="", help='dataset path')

    parser.add_argument('--write_yaml', action='store_true', default=False, help='write yaml filt')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    name = torch.cuda.get_device_name(0)

    seed_torch()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    net.set_pad(150)
    logging.info(f'Network:\n' f'\t{net.n_channels} input channels\n' f'\t{net.n_classes} output channels (classes)\n' f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device), strict=False)
        logging.info(f'Model loaded from {args.load}')

    # origin
    logging.info(f'Origin Net:')
    img_predict(net, device, root_path=os.path.join(args.data_path, 'predict/'))
    count_flops_params(net, device)

    # torch.save(net, \
    #         os.path.join(
    #             args.save_dir, \
    #             '../../..', \
    #             'model_vis/KeTi3Dataset-UNet', \
    #             'model.pth'))

    if args.write_yaml:
        flops, params = count_flops_params(net, device)
        mpa, infer_time = img_predict(net, device, root_path=os.path.join(args.data_path, 'predict/'))
        storage = os.path.getsize(args.load)
        with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
            yaml_data = {
                'MPA': {'baseline': round(mpa*100, 2), 'method': None},
                'FLOPs': {'baseline': round(flops/1e6, 2), 'method': None},
                'Parameters': {'baseline': round(params/1e3, 2), 'method': None},
                'Infer_times': {'baseline': round(infer_time*1e3, 2), 'method': None},
                'Storage': {'baseline': round(storage/1e6, 2), 'method': None},
            }
            yaml.dump(yaml_data, f)

    if args.prune_eval_path:
        if os.path.isdir(args.prune_eval_path):
            prune_net = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
            prune_net.set_pad(150)
            print("=> loading pruned model '{}'".format(args.prune_eval_path))
            prune_net.load_state_dict(torch.load(os.path.join(args.prune_eval_path, 'model_masked.pth')))
            masks_file = os.path.join(args.prune_eval_path, 'mask.pth')
            m_speedup = ModelSpeedup(prune_net, torch.randn(1, 3, 150, 150).to(device), masks_file, device)
            m_speedup.speedup_model()
            prune_net.load_state_dict(torch.load(os.path.join(args.prune_eval_path, 'export_pruned_model.pth'))['state_dict'])
            print("=> loaded pruned model")
            net = prune_net.to(device)
            count_flops_params(net, device)

    # dataloader
    if args.quan_mode != "fp16":
        # dataset = MarsImgDataset(root_dir, args.scale)
        dataset = WrapDataloader(root_dir, args.scale)
        n_val = int(len(dataset) * args.val / 100)
        n_train = len(dataset) - n_val
        random = False
        if random:
            train, val = random_split(dataset, [n_train, n_val])
        else:
            train = Subset(dataset, list(range(n_train)))
            val = Subset(dataset, list(range(n_train, len(dataset))))
        calib_loader = DataLoader(train, batch_size=args.batchsize, shuffle=False, num_workers=4, pin_memory=True)
    else:
        calib_loader = None

    input_shape = (1, 3, 150, 150)

    onnx_path = os.path.join(args.save_dir, '{}.onnx'.format(args.quan_mode))
    trt_path = os.path.join(args.save_dir, '{}.trt'.format(args.quan_mode))
    cache_path = os.path.join(args.save_dir, '{}.cache'.format(args.quan_mode))

    if args.quan_mode == "int8":
        extra_layer_bit = 8
    elif args.quan_mode == "fp16":
        extra_layer_bit = 16
    elif args.quan_mode == "best":
        extra_layer_bit = -1
    else:
        extra_layer_bit = 32

    engine = ModelSpeedupTensorRT(
        net,
        input_shape,
        config=None,
        calib_data_loader=calib_loader,
        batchsize=1,
        onnx_path=onnx_path,
        calibration_cache=cache_path,
        extra_layer_bit=extra_layer_bit,
    )
    if not os.path.exists(trt_path):
        engine.compress()
        engine.export_quantized_model(trt_path)
    else:
        engine.load_quantized_model(trt_path)

    # origin
    engine.n_classes = 1
    logging.info(f'After Quan:')
    img_predict(engine, device, root_path=os.path.join(args.data_path, 'predict/'))
    # count_flops_params(net, device)

    if args.write_yaml:
        storage = os.path.getsize(trt_path)
        mpa, infer_time = img_predict(engine, device, root_path=os.path.join(args.data_path, 'predict/'))
        with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
            yaml_data = {
                'MPA': {'baseline': yaml_data['MPA']['baseline'], 'method': round(mpa*100, 2)},
                'FLOPs': {'baseline': yaml_data['FLOPs']['baseline'], 'method': round(flops/1e6, 2)},
                'Parameters': {'baseline': yaml_data['Parameters']['baseline'], 'method': round(params/1e3, 2)},
                'Infer_times': {'baseline': yaml_data['Infer_times']['baseline'], 'method': round(infer_time*1e3, 2)},
                'Storage': {'baseline': yaml_data['Storage']['baseline'], 'method': round(storage/1e6, 2)},
                'Output_file': os.path.join(args.save_dir, 'export_pruned_model.pth'),
            }
            yaml.dump(yaml_data, f)