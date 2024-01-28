import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import json
import os
import sys
import copy
import pickle

from pathlib import Path
from contextlib import suppress
from collections import OrderedDict

from torch.nn.parallel import DistributedDataParallel as NativeDDP

from nas_vis.nas_burgerformer.timm.models import create_model
from nas_vis.nas_burgerformer.timm.utils import *
from nas_vis.nas_burgerformer.timm.data import create_dataset, create_loader, resolve_data_config
from nas_vis.nas_burgerformer.timm.models import create_model, resume_checkpoint

import logging

# for evolutionary search
from nas_vis.nas_burgerformer.evolver import PopulationEvolver

from nas_vis.nas_burgerformer.arch_generate import arch2_weight, weight2_arch
import nas_vis.nas_burgerformer.supernet
from nas_vis.nas_burgerformer.cal_flops_params import cal_flops_parameters

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format=log_format, datefmt='%m/%d %I:%M:%S %p')
_log = logging.getLogger('search')


def get_args_parser():
    parser = argparse.ArgumentParser('evolutionary search', add_help=False)
    parser.add_argument('-b', '--batch-size', type=int, default=100, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--model-path', default=None, type=str, help='Path to super-network checkpoint')
    parser.add_argument('--output_dir', default='output/search_evo_test', help='path where to save, empty for no saving')

    # model
    parser.add_argument('--model', default='unifiedarch_s28', type=str, help='Model type to evaluate')
    parser.add_argument('--pre_defined_arch', default='', type=str)
    parser.add_argument('--only_micro', action='store_true', default=False)
    parser.add_argument('--only_macro', action='store_true', default=False)
    parser.add_argument('--net_config', default='a', type=str, help='arch_name')
    parser.add_argument('--num-classes', type=int, default=100, metavar='N', help='number of label classes (Model default if None)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='Resume full model and optimizer state from checkpoint (default: none)')
    parser.add_argument('--dataset-download', action='store_true', default=False, help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME', help='path to class to idx mapping file (default: "")')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--pin-mem', action='store_true', default=False, help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--norm-track', action='store_true', default=False, help='track in bn')

    parser.add_argument('--seed', default=0, type=int)

    # Dataset parameters
    parser.add_argument('--data-dir', default='/home/workspace/yanglongxing/dataset/imagenet_100/', type=str, help='dataset path')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='', help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--val-split', metavar='NAME', default='validation', help='dataset validation split (default: validation)')
    parser.add_argument('--no-prefetcher', action='store_true', default=False, help='disable fast prefetcher')
    parser.add_argument('--workers', default=8, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--sync-bn', action='store_true', help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--local_rank", default=0, type=int)

    # Misc
    parser.add_argument('--print-freq', default=100, type=int, help='Interval of iterations to print training/eval info.')

    # for searching
    parser.add_argument('--no-use-holdout', action='store_false', dest='use_holdout', default=True, help='Use sub-train and sub-eval set for evolutionary search.')

    # evolutionary search hyper-parameters
    parser.add_argument('--init-popu-size', default=500, type=int, help='Initial population size, which determines how many sub-networks are randomly sampled at the first iteration.')
    parser.add_argument('--search-iter', default=20, type=int, help='Search iterations, with the first one being random sampling.')
    parser.add_argument('--parent-size', default=75, type=int, help='Number of top-performing sub-networks used to generate new sub-networks')
    parser.add_argument('--mutate-size', default=75, type=int, help='Number of sub-networks generated from mutation/crossover.')
    parser.add_argument('--mutate-prob', default=0.3, type=float, help='Mutation probability.')
    # 500 20 75 75 0.3

    # resourse
    parser.add_argument('--target', default='flops', type=str)  # flops, params, flops_params, none
    parser.add_argument('--target_flops', default=1.8e9, type=float)
    parser.add_argument('--target_params', default=12e6, type=float)
    parser.add_argument('--thr', default=0.9, type=float)

    return parser


def pickle_save(obj, path):
    with open(path, 'wb') as file_id:
        pickle.dump(obj, file_id)


def write_results(obj, path, item_name_list=None):
    '''
        `obj`: is a list of Individual class
    '''

    if item_name_list is None:
        item_name_list = ['Idx', 'Acc', 'Network_def']
    else:
        assert len(item_name_list) == 3
    with open(path, 'w') as file_id:
        file_id.write('{}, {}, {}\n'.format(item_name_list[0], item_name_list[1], item_name_list[2]))
        for i in range(len(obj)):
            file_id.write('{}, {}, {}\n'.format(i, obj[i].score, obj[i].network_str))


def speedup_loader(loader, device):
    new_loader = []
    for input, target in loader:
        new_loader.append((input.to(device), target.to(device)))
    return new_loader


def validate(model, loader, skip_weightsss, norm_mixer_act_weightssss, depths, widths, ratios, args, amp_autocast=suppress, log_suffix=''):
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    start_time = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()

            with amp_autocast():
                output = model(input, skip_weightsss, norm_mixer_act_weightssss, depths, widths, ratios)
            if isinstance(output, (tuple, list)):
                output = output[0]

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)

            torch.cuda.synchronize()

            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            if last_batch:
                arch = weight2_arch(skip_weightsss, norm_mixer_act_weightssss, depths, widths, ratios)
                # arch_str = str(arch)
                flops, params = cal_flops_parameters(arch)
                log_name = 'Test' + log_suffix
                if args.local_rank == 0:
                    _log.info('{}: ' 'Time: {:.3f}s  ' 'Acc@1: {:>7.4f}  ' 'Acc@5: {:>7.4f}  ' 'flops/G: {:>7.4f}  ' 'params/M: {:>7.4f}  \n'.format(log_name, time.time() - start_time, top1_m.avg, top5_m.avg, flops / 1e9, params / 1e6))
                # print('{}: ' 'Time: {:.3f}  ' 'Acc@1: {:>7.4f}  ' 'Acc@5: {:>7.4f}  \n' 'arch : {}  '.format(log_name, time.time() - start_time, top1_m.avg, top5_m.avg, arch_str))

    metrics = OrderedDict([('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


def init_multigpu(args):
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        if args.local_rank == 0:
            _log.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.' % (args.rank, args.world_size))
    else:
        _log.info('Training with a single process on 1 GPUs.')


def multigpu_model(model, args):
    assert args.rank >= 0
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=True, find_unused_parameters=True)
    return model


def main(args):

    if not hasattr(args, 'gpu'):
        args.gpu = 0

    if args.local_rank == 0:
        _log.info(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed

    random_seed(seed, 0)

    cudnn.benchmark = False
    cudnn.deterministic = True

    init_multigpu(args)

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        norm_track=args.norm_track,
    ).cuda()

    if args.resume:
        resume_checkpoint(model, args.resume, optimizer=None, loss_scaler=None, log_info=args.local_rank == 0)

    model = multigpu_model(model, args)

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    dataset_eval = create_dataset(args.dataset, root=args.data_dir, split=args.val_split, is_training=False, class_map=args.class_map, download=args.dataset_download, batch_size=args.batch_size)
    data_loader_val = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        use_prefetcher=not args.no_prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )
    data_loader_val = speedup_loader(data_loader_val, device)

    # evolutionary search population
    popu_evolve = PopulationEvolver(device)

    _best_result_history = []
    print(f"{args.search_iter = }")
    for search_iter in range(args.search_iter):
        # generate sub-networks to evaluate
        if search_iter == 0:
            popu_evolve.random_sample(
                args.init_popu_size,
                only_micro=args.only_micro,
                only_macro=args.only_macro,
                pre_defined_arch=args.pre_defined_arch,
                target=args.target,
                target_flops=args.target_flops,
                target_params=args.target_params,
                thr=args.thr,
            )
        else:
            popu_evolve.evolve_sample(
                parent_size=args.parent_size,
                mutate_prob=args.mutate_prob,
                mutate_size=args.mutate_size,
                only_micro=args.only_micro,
                only_macro=args.only_macro,
                pre_defined_arch=args.pre_defined_arch,
                target=args.target,
                target_flops=args.target_flops,
                target_params=args.target_params,
                thr=args.thr,
            )
        # evaluate sub-networks
        for subnet_idx in range(len(popu_evolve.popu)):
            sub_network_def = popu_evolve.popu[subnet_idx].network_def
            sub_network_str = popu_evolve.popu[subnet_idx].network_str
            if args.local_rank == 0:
                _log.info('Iter: [{}][{}/{}]: {}'.format(search_iter, subnet_idx, len(popu_evolve.popu), sub_network_str))

            # test_stats = evaluate(data_loader_val, model, device, args.print_freq, logger=_dummy_log)
            test_stats = validate(model, data_loader_val, sub_network_def[0], sub_network_def[1], sub_network_def[2], sub_network_def[3], sub_network_def[4], args)

            popu_evolve.popu[subnet_idx].score = test_stats['top1']

        if args.output_dir:
            # create directory to save iter data
            Path(os.path.join(args.output_dir, 'iter@{}'.format(search_iter))).mkdir(parents=True, exist_ok=True)

            # save population
            pickle_save(popu_evolve.popu, path=os.path.join(args.output_dir, 'iter@{}'.format(search_iter), 'popu.pickle'))
            write_results(popu_evolve.popu, path=os.path.join(args.output_dir, 'iter@{}'.format(search_iter), 'popu.txt'))

        popu_evolve.update_history()
        popu_evolve.sort_history()
        if args.local_rank == 0:
            _log.info('{}\n'.format(popu_evolve.history_popu[0]))
        _best_result_history.append(popu_evolve.history_popu[0])
        
        flops, params = cal_flops_parameters(weight2_arch(*_best_result_history[-1].network_def))
        _log.info('Iter [{}]: Acc = {:.2f}, flops = {:.2f}G, params = {:.2f}M Network_def = {}\n'.format(len(_best_result_history), _best_result_history[-1].score, flops / 1e9, params / 1e6, _best_result_history[-1].network_str))

        # print(popu_evolve.history_popu[0])
        if args.output_dir:
            # save history popu
            pickle_save(popu_evolve.history_popu, path=os.path.join(args.output_dir, 'iter@{}'.format(search_iter), 'history_popu.pickle'))
            # save top-`ags.parent_size` to a text file
            write_results(popu_evolve.history_popu[0:args.parent_size], path=os.path.join(args.output_dir, 'iter@{}'.format(search_iter), 'history_popu_top.txt'))
            # save best netwrok_def for each iteration
            write_results(_best_result_history, path=os.path.join(args.output_dir, 'summary.txt'), item_name_list=['Iter', 'Acc', 'Network_def'])

    for i in range(len(_best_result_history)):
        flops, params = cal_flops_parameters(weight2_arch(*_best_result_history[i].network_def))
        if args.local_rank == 0:
            _log.info('Iter_summary [{}]: Acc = {:.2f}, flops = {:.2f}G, params = {:.2f}M Network_def = {}'.format(i, _best_result_history[i].score, flops / 1e9, params / 1e6, _best_result_history[i].network_str))

    with open('search_retrain/arch.py', 'a') as f:
        f.write('\n{} = {}\n'.format(args.net_config, _best_result_history[-1].network_str))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser('evolutionary search', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
