import torch
import os
import sys
import time
import glob
import numpy as np
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from ..models.model_search import Network
from ..models.architect import Architect
from ..models.genotypes import init_space

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--no_norm', action='store_true', default=False, help='not norm')
parser.add_argument('--begin', type=int, default=0, help='epoch of begining search')
parser.add_argument('--rdss_prob', type=float, default=0., help='random dynamic search space probability')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100')
parser.add_argument('--space_config', type=str, default='space_config_static', help='which architecture to save')
parser.add_argument('--arch', type=str, help='which architecture to save')
parser.add_argument('--dss_freq', type=int, default=1, help='frequence of changing dynamic search space')
parser.add_argument('--dss_max_ops', type=int, default=28, help='max ops num in each dynamic search space')
parser.add_argument('--confidence', type=float, default=1.44, help='confidence of ucb')
parser.add_argument('--saliency_type', type=str, default='all', help='simple, all')
parser.add_argument('--data', type=str, default='data/cifar10', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--resume', action='store_true', default=False, help='resume')
parser.add_argument('--resume_path', type=str, default='', help='resume_path')
# gumbel
parser.add_argument('--gumbel', action='store_true', default=False, help='use gumbel softmax')
parser.add_argument('--tau_min', type=float, help='The minimum tau for Gumbel')
parser.add_argument('--tau_max', type=float, help='The maximum tau for Gumbel')
args = parser.parse_args()

if args.resume:
    args.save = args.resume_path
else:
    args.save = 'logs/search-{}-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"), int(round(time.time() * 1000)))
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

space_config = init_space(args.space_config)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == 'cifar10':
    CIFAR_CLASSES = 10
elif args.dataset == 'cifar100':
    CIFAR_CLASSES = 100


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    if args.resume:
        state = utils.load_checkpoint(os.path.join(args.save, 'checkpoint.pt'))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion, args.confidence, args.dss_max_ops, args.saliency_type, not args.no_norm, primitives=(None if space_config is None else space_config['PRIMITIVES']))
    if args.resume:
        with torch.no_grad():
            for i, alpha in enumerate(model.arch_parameters()):
                alpha.copy_(state['alpha'][i])
            for i, beta in enumerate(model.space_parameters()):
                beta.copy_(state['beta'][i])
        model.load_state_dict(state['model'])
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.dataset == 'cifar10':
        dataset_class = dset.CIFAR10
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
    elif args.dataset == 'cifar100':
        dataset_class = dset.CIFAR100
        train_transform, valid_transform = utils._data_transforms(args)

    train_data = dataset_class(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), pin_memory=True, num_workers=0)
    valid_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]), pin_memory=True, num_workers=0)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    architect = Architect(model, args)

    start_epoch = 0
    stage_id = 0
    if args.resume:
        start_epoch = state['epoch'] + 1
        stage_id = state['stage_id']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if architect is not None:
            architect.optimizer.load_state_dict(state['arch_optimizer'])
        # validation check
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)
    for epoch in range(start_epoch, args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        if space_config is not None:
            if space_config['type'] == 'shrink':
                if stage_id < len(space_config['stage']) and epoch == space_config['stage'][stage_id][0]:
                    model.shrink_space(space_config['stage'][stage_id][1])
                    stage_id += 1
            elif space_config['type'] == 'expand':
                if stage_id < len(space_config['stage']) and epoch == space_config['stage'][stage_id][0]:
                    if stage_id == 0:
                        assert space_config['stage'][0][0] == 0
                        model.set_space(range(0, len(space_config['PRIMITIVES'])), False)
                    start = 0 if stage_id == 0 else space_config['stage'][stage_id - 1][1]
                    end = space_config['stage'][stage_id][1]
                    model.set_space(range(start, end), True)
                    stage_id += 1

        # training
        train_acc, train_obj = train(epoch, train_queue, valid_queue, model, architect, criterion, optimizer, lr, args.rdss_prob, args)
        logging.info('train_acc %f', train_acc)

        # validation
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue, model, criterion)
            logging.info('valid_acc %f', valid_acc)

        logging.info(model.op_saliency()[0])
        logging.info(model.op_saliency()[1])
        logging.info(model._opt_steps_normal)
        logging.info(model._opt_steps_reduce)

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        utils.save(model, os.path.join(args.save, 'weights.pt'))

        utils.save_checkpoint(
            {
                'alpha': tuple(model.arch_parameters()),
                'beta': tuple(model.space_parameters()),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'stage_id': stage_id,
                'arch_optimizer': None if architect is None else architect.optimizer.state_dict()
            }, os.path.join(args.save, 'checkpoint.pt'))

    if args.arch is not None:
        utils.write_to_file('{} = {}'.format(args.arch, model.genotype()), 'genotypes.py')


def train(epoch, train_queue, valid_queue, model, architect, criterion, optimizer, lr, rdss_prob, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    if args.gumbel:
        tau = args.tau_max - (args.tau_max - args.tau_min) * epoch / (args.epochs - 1)
        logging.info('Tau: {}'.format(tau))
    else:
        tau = None
    model.set_tau(tau)

    end = time.time()
    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)
        data_time.update(time.time() - end)

        if epoch >= args.begin:
            if step % args.dss_freq == 0:
                model.update_dss()
                # model.save_info(os.path.join(args.save, 'info'))
                if torch.rand([]) < args.rdss_prob:
                    model.generate_random_dss()
                else:
                    model.generate_dss()
        else:
            model.generate_random_dss()

        if architect is not None:
            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.cuda()
            target_search = target_search.cuda()

            architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.report_freq == 0:
            logging.info('train %03d [data/s: %.5f][batch/s: %.5f][loss: %e][top1: %f][top5: %f]', step, data_time.avg, batch_time.avg, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    end = time.time()
    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input).cuda()
        target = Variable(target).cuda(async=True)
        data_time.update(time.time() - end)

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        batch_time.update(time.time() - end)
        end = time.time()

        if step % args.report_freq == 0:
            logging.info('valid %03d [data/s: %.5f][batch/s: %.5f][loss: %e][top1: %f][top5: %f]', step, data_time.avg, batch_time.avg, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
