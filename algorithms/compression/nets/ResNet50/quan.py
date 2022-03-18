import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torchvision.models as models
import argparse
import os
import sys
import logging
import random
import numpy as np
import time
import glob
from utils import *
from lib.compression.pytorch import ModelSpeedup
from lib.compression.pytorch.utils.counter import count_flops_params
from lib.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

import yaml

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--data', default='./data/tiny-imagenet-200', type=str, help='dataset path')
parser.add_argument('--model', default='resnet50', type=str, help='model name')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--print_freq', default=1000, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='checkpoint/resnet50.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--save_dir', default='./log/quan/test', help='The directory used to save the trained models', type=str)
parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=1)
parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--baseline', action='store_true', help='evaluate model on validation set')
parser.add_argument('--prune_eval_path', default=None, type=str, metavar='PATH', help='path to eval pruned model')
parser.add_argument('--pruner', default='agp', type=str, help='pruner: agp|taylor|fpgm')

parser.add_argument('--quan_mode', default='int8', help='fp16 int8 best', type=str)
parser.add_argument('--calib_num', type=int, default=1280, help='random seed')

parser.add_argument('--write_yaml', action='store_true', default=False, help='write yaml filt')

args = parser.parse_args()
best_prec1 = 0

create_exp_dir(args.save_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
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

    device = torch.device('cuda')
    model = get_model(args.model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    train_loader, val_loader, calib_loader = get_data_loader(args)
    criterion = nn.CrossEntropyLoss().cuda()

    if args.write_yaml:
        flops, params, _ = count_flops_params(model, (1, 3, 64, 64), verbose=False)
        _, top1, _, infer_time, _ = validate(model, val_loader, criterion, is_trt=False)
        storage = os.path.getsize(args.resume)
        with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
            yaml_data = {
                'Accuracy': {'baseline': round(top1, 2), 'method': None},
                'FLOPs': {'baseline': round(flops/1e6, 2), 'method': None},
                'Parameters': {'baseline': round(params/1e6, 2), 'method': None},
                'Infer_times': {'baseline': round(infer_time*1e3, 2), 'method': None},
                'Storage': {'baseline': round(storage/1e6, 2), 'method': None},
            }
            yaml.dump(yaml_data, f)

    # load pruned model
    if args.prune_eval_path:
        if os.path.isdir(args.prune_eval_path):
            logging.info("=> loading pruned model '{}'".format(args.prune_eval_path))
            model.load_state_dict(torch.load(os.path.join(args.prune_eval_path, 'model_masked.pth')))
            masks_file = os.path.join(args.prune_eval_path, 'mask.pth')
            m_speedup = ModelSpeedup(model, torch.rand(1, 3, 64, 64).to(device), masks_file, device)
            m_speedup.speedup_model()
            model.load_state_dict(torch.load(os.path.join(args.prune_eval_path, 'model_speed_up_finetuned.pth'))['state_dict'])
            logging.info("=> loaded pruned model")
        else:
            logging.info("=> no pruned model found at '{}'".format(args.prune_eval_path))

    # define loss function (criterion) and optimizer

    flops, params, _ = count_flops_params(model, (1, 3, 64, 64), verbose=False)
    if args.baseline:
        logging.info("Baseline: ")
        loss, top1, pre_time, infer_time, post_time = validate(model, val_loader, criterion, is_trt=False)
        logging.info("Baseline: [Top1: {:.2f}%][FLops: {:.5f}M][Params: {:.5f}M][Infer: {:.2f}ms]\n".format(top1, flops / 1e6, params / 1e6, infer_time * 1000))
    
    input_shape = (args.batch_size, 3, 64, 64)

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
        input_shape,
        config=None,
        calib_data_loader=calib_loader,
        batchsize=args.batch_size,
        onnx_path=onnx_path,
        calibration_cache=cache_path,
        extra_layer_bit=extra_layer_bit,
    )
    if not os.path.exists(trt_path):
        engine.compress()
        engine.export_quantized_model(trt_path)
    else:
        engine.load_quantized_model(trt_path)

    loss, top1, pre_time, infer_time, post_time = validate(engine, val_loader, criterion)
    logging.info("Quan model: [Top1: {:.2f}%][FLops: {:.5f}M][Params: {:.5f}M][Infer: {:.2f}ms]\n".format(top1, flops / 1e6, params / 1e6, infer_time * 1000))

    if args.write_yaml:
        storage = os.path.getsize(trt_path)
        with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
            yaml_data = {
                'Accuracy': {'baseline': yaml_data['Accuracy']['baseline'], 'method': round(top1, 2)},
                'FLOPs': {'baseline': yaml_data['FLOPs']['baseline'], 'method': round(flops/1e6, 2)},
                'Parameters': {'baseline': yaml_data['Parameters']['baseline'], 'method': round(params/1e6, 2)},
                'Infer_times': {'baseline': yaml_data['Infer_times']['baseline'], 'method': round(infer_time*1e3, 2)},
                'Storage': {'baseline': yaml_data['Storage']['baseline'], 'method': round(storage/1e6, 2)},
                'Output_file': os.path.join(args.save_dir, '{}_{}.trt'.format(args.model, args.quan_mode)),
            }
            yaml.dump(yaml_data, f)


def validate(model, val_loader, criterion, is_trt=True):
    """
    Run evaluation
    """
    pre_time = AverageMeter()
    infer_time = AverageMeter()
    post_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if not is_trt:
        model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (image, label) in enumerate(val_loader):
            if not is_trt:
                image = image.cuda()
                label = label.cuda()
            pre_time.update(time.time() - end)
            end = time.time()

            if is_trt:
                output, trt_infer_time = model.inference(image)
                infer_time.update(trt_infer_time)
            else:
                output = model(image)
                infer_time.update(time.time() - end)
            end = time.time()

            loss = criterion(output, label)

            prec1 = accuracy(output.data, label.data)[0]
            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            post_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info('Test: [{0}/{1}]\t'
                             'Infer Time {infer_time.val:.3f} ({infer_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i, len(val_loader), infer_time=infer_time, loss=losses, top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg, pre_time.avg, infer_time.avg, post_time.avg


def get_model(model_name):
    if model_name.lower() == 'resnet50':
        model = models.resnet50()
    elif model_name.lower() == 'resnet101':
        model = models.resnet101()
    else:
        raise NotImplementedError('Not Support {}'.format(model_name))
    model.fc = nn.Linear(model.fc.in_features, 200)
    return model


def get_data_loader(args):
    train_dir = os.path.join(args.data, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dir = os.path.join(args.data, 'val')
    val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    n_train = len(train_dataset)
    indices = list(range(n_train))
    random.shuffle(indices)
    calib_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:args.calib_num])
    calib_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=calib_sampler)

    return train_loader, val_loader, calib_loader


if __name__ == "__main__":
    main()