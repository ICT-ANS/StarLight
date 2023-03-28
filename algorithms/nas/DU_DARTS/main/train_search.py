import glob
import logging
import os
import sys
sys.path.append('../')
sys.path.append('../../../')
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as dset

from algorithms.nas.DU_DARTS.utils import utils
from algorithms.nas.DU_DARTS.models.architect import Architect
from algorithms.nas.DU_DARTS.models.model_search import Network
from algorithms.nas.DU_DARTS.config.search_config import args

# if args.debug:
#     args.save = 'log/debug-search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# else:
#     args.save = 'log/search-{}-{}-{}'.format(args.dataset, args.save, time.strftime("%Y%m%d-%H%M%S"))

if not os.path.exists('log'):
    os.mkdir('log')
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == 'cifar10':
    CIFAR_CLASSES = 10
elif args.dataset == 'cifar100':
    CIFAR_CLASSES = 100
else:
    raise ValueError('No Defined Dataset!!!')


def main():
    utils.set_seed(seed=0)
    logging.info('gpu device = %d' % args.gpu_id)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args, args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    architect = Architect(model, args)

    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils.data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True,
                                  transform=train_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils.data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True,
                                   transform=train_transform)
    else:
        raise ValueError('No Defined Dataset!!!')

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))
    if args.debug:
        split = args.batch_size
        num_train = 2 * args.batch_size

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        # training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
        logging.info('train_acc %f', train_acc)

        # validation
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)

        # print arch params
        logging.info('normal alpha = %s', F.softmax(model.alphas_normal, dim=-1))
        logging.info('normal beta = %s', F.sigmoid(model.betas_normal))
        logging.info('reduce alpha = %s', F.softmax(model.alphas_reduce, dim=-1))
        logging.info('reduce beta = %s', F.sigmoid(model.betas_reduce))

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    args.epoch = epoch
    if args.warmup and epoch < args.warmup_epoch:
        logging.info('epoch %d warming up!!!', epoch)
    else:
        logging.info('epoch %d train arch!!!', epoch)

    for step, (input, target) in enumerate(train_queue):
        args.step = step

        model.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        if architect is not None:
            if args.warmup and epoch < args.warmup_epoch:
                pass
            else:
                # get a random minibatch from the search queue with replacement
                input_search, target_search = next(iter(valid_queue))
                input_search = input_search.cuda()
                target_search = target_search.cuda(non_blocking=True)

                architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    utils.run_func(args, main)
