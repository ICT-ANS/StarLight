import sys
import torch
from algorithms.nas.DU_DARTS.utils import utils
import logging
from algorithms.nas.DU_DARTS.models import genotypes

import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
from algorithms.nas.DU_DARTS.models.model import NetworkCIFAR as Network


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--arch', type=str, default='du_darts_c10_s0', help='which architecture to use')
parser.add_argument('--auxiliary', action='store_false', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--data', type=str, default='./dataset/', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--gpu_id', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--pretrained_ckpt', type=str, default=None, help='path to pretrained checkpoint')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

if args.dataset == 'cifar10':
    CIFAR_CLASSES = 10
elif args.dataset == 'cifar100':
    CIFAR_CLASSES = 100
else:
    raise ValueError('No Defined Dataset!!!')


def main():
    utils.set_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu_id)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda()

    params = utils.count_parameters_in_MB(model)
    logging.info("param size = %f M", params)

    # load pretrained weights
    model.load_state_dict(torch.load(args.pretrained_ckpt))
    logging.info("load pretrained weights done")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if args.dataset == 'cifar10':
        train_transform, valid_transform = utils.data_transforms_cifar10(args)
        valid_data = dset.CIFAR10(root=args.data+args.dataset, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = utils.data_transforms_cifar100(args)
        valid_data = dset.CIFAR100(root=args.data+args.dataset, train=False, download=True, transform=valid_transform)
    else:
        raise ValueError('No Defined Dataset!!!')
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    with torch.no_grad():
        model.drop_path_prob = args.drop_path_prob
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('valid_acc %f\t', valid_acc)


def infer(valid_queue, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        logits, _ = model(input)
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
    main()
