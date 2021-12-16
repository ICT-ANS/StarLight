import argparse
import logging
import os
import sys
from algorithms.nas.DU_DARTS.models import genotypes

import torch
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

from algorithms.nas.DU_DARTS.utils import utils
from algorithms.nas.DU_DARTS.models.model import NetworkCIFAR as Network

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--arch', type=str, default='du_darts_c10_s0', help='which architecture to use')
parser.add_argument('--auxiliary', action='store_false', default=True, help='use auxiliary tower')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--data', type=str, default='./dataset/imagenet/', help='location of the data corpus')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--pretrained_ckpt', type=str, default='./pretrained/c10_imagenet_transfer_best_model.pth')
parser.add_argument('--seed', type=int, default=0, help='random seed')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CLASSES = 1000


def main():
    utils.set_seed(args.seed)
    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model.drop_path_prob = args.drop_path_prob
    model = nn.DataParallel(model, device_ids=[args.gpu])
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    model.load_state_dict(torch.load(args.pretrained_ckpt)['state_dict'], strict=False)
    logging.info("load pretrained weights done")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    validdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8
    )

    valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc_top1 %f valid_acc_top5 %f', valid_acc_top1, valid_acc_top5)


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

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
    main()
