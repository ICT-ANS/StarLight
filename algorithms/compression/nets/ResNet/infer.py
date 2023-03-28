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

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--data', default='./data/tiny-imagenet-200', type=str, help='dataset path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--print_freq', default=1000, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--save_dir', default='./log/infer/test', help='The directory used to save the trained models', type=str)
parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--model', default='resnet50', type=str, help='model name')
parser.add_argument('--infer_mode', default='origin', type=str, help='origin|prune|quan|prune_quan')
# infer origin model
parser.add_argument('--origin_model_path', default=None, type=str, help='prigin model path')
# infer pruned model
parser.add_argument('--prune_model_path', default=None, type=str, help='prune model path')
parser.add_argument('--prune_model_masked', default=None, type=str, help='prune model masked pth')
parser.add_argument('--prune_masked', default=None, type=str, help='prune mask')
# infer quan or prune_quan model
parser.add_argument('--quan_model_path', default=None, type=str, help='quan model path')
parser.add_argument('--quan_cache_path', default=None, type=str, help='quan cache path')

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

    if args.infer_mode == 'origin':
        checkpoint = torch.load(args.origin_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        flops, params, _ = count_flops_params(model, (1, 3, 64, 64), verbose=False)
        is_trt = False
    elif args.infer_mode == 'prune':
        model.load_state_dict(torch.load(args.prune_model_masked))
        m_speedup = ModelSpeedup(model, torch.rand(1, 3, 64, 64).to(device), args.prune_masked, device)
        m_speedup.speedup_model()
        model.load_state_dict(torch.load(args.prune_model_path))
        flops, params, _ = count_flops_params(model, (1, 3, 64, 64), verbose=False)
        is_trt = False
    elif args.infer_mode == 'quan' or args.infer_mode == 'prune_quan':
        flops, params, _ = count_flops_params(model, (1, 3, 64, 64), verbose=False)
        model = ModelSpeedupTensorRT(model, input_shape=(args.batch_size, 3, 64, 64), config=None, calib_data_loader=None, batchsize=args.batch_size, onnx_path=None, calibration_cache=args.quan_cache_path, extra_layer_bit=-1)
        model.load_quantized_model(args.quan_model_path)
        is_trt = True
    else:
        raise NotImplementedError

    train_loader, val_loader = get_data_loader(args)
    criterion = nn.CrossEntropyLoss().cuda()

    loss, top1, pre_time, infer_time, post_time = validate(model, val_loader, criterion, is_trt)
    logging.info("Model: [Top1: {:.2f}%][FLops: {:.5f}M][Params: {:.5f}M][Infer: {:.2f}ms]\n".format(top1, flops / 1e6, params / 1e6, infer_time * 1000))


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
    if model_name == 'resnet50':
        model = models.resnet50()
    elif model_name == 'resnet101':
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

    return train_loader, val_loader


if __name__ == "__main__":
    main()