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
from lib.compression.pytorch.utils.counter import count_flops_params

from lib.compression.pytorch import ModelSpeedup
from lib.algorithms.pytorch.pruning import (TaylorFOWeightFilterPruner, FPGMPruner, AGPPruner)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--data', default='./data/tiny-imagenet-200', type=str, help='dataset path')
parser.add_argument('--model', default='resnet50', type=str, help='model name')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

parser.add_argument('--pruner', default='fpgm', type=str, help='pruner: agp|taylor|fpgm')
parser.add_argument('--sparsity', default=0.2, type=float, metavar='LR', help='prune sparsity')
parser.add_argument('--finetune_epochs', default=10, type=int, metavar='N', help='number of epochs for exported model')
parser.add_argument('--finetune_lr', default=0.001, type=float, metavar='N', help='number of lr for exported model')
parser.add_argument('--finetune_momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--finetune_weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')

parser.add_argument('--batch_size', default=128, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', default=0.005, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')

parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='./checkpoint/resnet50.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', default='./log/prune/test', help='The directory used to save the trained models', type=str)
parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=1)
parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--baseline', action='store_true', default=False, help='evaluate model on validation set')
parser.add_argument('--prune_eval_path', default='', type=str, metavar='PATH', help='path to eval pruned model')

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
    model = get_model(args.model).to(device)

    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded checkpoint")
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    train_loader, val_loader = get_data_loader(args)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr)

    if args.baseline:
        logging.info("Baseline: ")
        flops, params, _ = count_flops_params(model, (1, 3, 64, 64), verbose=False)
        loss, top1, pre_time, infer_time, post_time = validate(model, val_loader, criterion)
        logging.info("Baseline: [Top1: {:.2f}%][FLops: {:.5f}M][Params: {:.5f}M][Infer: {:.2f}ms]\n".format(top1, flops / 1e6, params / 1e6, infer_time * 1000))

    # only eval pruned model
    if args.prune_eval_path:
        model = get_model(args.model).to(device)
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_masked.pth')))
        masks_file = os.path.join(args.save_dir, 'mask.pth')
        m_speedup = ModelSpeedup(model, torch.rand(1, 3, 64, 64).to(device), masks_file, device)
        m_speedup.speedup_model()
        model.load_state_dict(args.prune_eval_path)

        flops, params, _ = count_flops_params(model, (1, 3, 64, 64), verbose=False)
        loss, top1, pre_time, infer_time, post_time = validate(model, val_loader, criterion)
        logging.info("Evaluation result : [Top1: {:.2f}%][FLops: {:.5f}M][Params: {:.5f}M][Infer: {:.2f}ms]\n".format(top1, flops / 1e6, params / 1e6, infer_time * 1000))

        exit(0)

    def trainer(model, optimizer, criterion, epoch):
        result = train(epoch, model, train_loader, criterion, optimizer, args)
        return result

    # get pruner, agp|taylor|fpgm
    if args.pruner == 'agp':
        config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
        pruner = AGPPruner(
            model,
            config_list,
            optimizer,
            trainer,
            criterion,
            num_iterations=1,
            epochs_per_iteration=1,
            pruning_algorithm='taylorfo',
        )
    elif args.pruner == 'taylor':
        config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
        pruner = TaylorFOWeightFilterPruner(
            model,
            config_list,
            optimizer,
            trainer,
            criterion,
            sparsifying_training_batches=1,
        )
    elif args.pruner == 'fpgm':
        config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
        pruner = FPGMPruner(
            model,
            config_list,
            optimizer,
            dummy_input=torch.rand(1, 3, 64, 64).to(device),
        )
    else:
        raise NotImplementedError
    pruner.compress()

    # save masked pruned model (model + mask)
    pruner.export_model(os.path.join(args.save_dir, 'model_masked.pth'), os.path.join(args.save_dir, 'mask.pth'))

    # export pruned model
    logging.info('Speed up masked model...')
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_masked.pth')))
    masks_file = os.path.join(args.save_dir, 'mask.pth')

    m_speedup = ModelSpeedup(model, torch.rand(1, 3, 64, 64).to(device), masks_file, device)
    m_speedup.speedup_model()

    loss, top1, pre_time, infer_time, post_time = validate(model, val_loader, criterion)
    flops, params, _ = count_flops_params(model, (1, 3, 64, 64), verbose=False)
    logging.info('Evaluation result (speed up model): [Top1: {:.2f}%][FLops: {:.5f}M][Params: {:.5f}M][Infer: {:.2f}ms]\n'.format(top1, flops / 1e6, params / 1e6, infer_time * 1000))

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_speed_up.pth'))

    # fintune pruned model
    logging.info('Finetuning export pruned model...')
    export_pruned_model = model
    optimizer = torch.optim.SGD(export_pruned_model.parameters(), args.finetune_lr, momentum=args.finetune_momentum, weight_decay=args.finetune_weight_decay)
    for epoch in range(args.finetune_epochs):
        train(epoch, export_pruned_model, train_loader, criterion, optimizer, args)

        loss, top1, pre_time, infer_time, post_time = validate(export_pruned_model, val_loader, criterion)
        logging.info("Finetune Epoch {}: [Top1: {:.2f}%][FLops: {:.5f}M][Params: {:.5f}M][Infer: {:.2f}ms]\n".format(epoch, top1, flops / 1e6, params / 1e6, infer_time * 1000))

        if epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': export_pruned_model.state_dict(),
                'prec1': top1,
            }, filename=os.path.join(args.save_dir, 'model_speed_up_finetuned.pth'))


def train(epoch, model, train_loader, criterion, optimizer, args):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (image, label) in enumerate(train_loader):
        # if i > 10: break
        data_time.update(time.time() - end)

        image = image.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        prec1 = accuracy(output.data, label.data)[0]
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))


def validate(model, val_loader, criterion):
    """
    Run evaluation
    """
    pre_time = AverageMeter()
    infer_time = AverageMeter()
    post_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (image, label) in enumerate(val_loader):
            # if i > 10: break
            image = image.cuda()
            label = label.cuda()

            pre_time.update(time.time() - end)
            end = time.time()

            torch.cuda.synchronize()
            output = model(image)
            torch.cuda.synchronize()

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
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_dir = os.path.join(args.data, 'val')
    val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    return train_loader, val_loader


if __name__ == "__main__":
    main()