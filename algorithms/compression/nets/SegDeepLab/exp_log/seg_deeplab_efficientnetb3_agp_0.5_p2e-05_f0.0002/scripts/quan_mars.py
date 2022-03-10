import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from dataReader.dataset import Reader
from torch.utils.data import DataLoader

import torchvision.models as models
import argparse
import os
import sys
import logging
import random
import numpy as np
import time
import torch.nn.functional as F

import glob
from utils.utils import get_evaluation_score
from lib.utils import *
from lib.compression.pytorch import ModelSpeedup
from lib.compression.pytorch.utils.counter import count_flops_params
from lib.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT
from prune_seg.export_utils_seg_deeplab import get_pruned_model
from prune_seg.utils.utils import run_func

from eval import evaluate_quan
from models.EfficientNet.efficientnet import EfficientNet


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--data_root', default='./dataset/Mars_Seg_1119/Data', type=str, help='dataset path')
parser.add_argument('--sparsity', default=0.0, type=float, help='only for models with sparsity')
parser.add_argument('--model', default='seg_deeplab_efficientnetb3', type=str, help='model name')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--print_freq', default=1000, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--save_dir', default='./exp_log/', help='The directory used to save the trained models', type=str)
parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=1)
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--num_classes', default=8, type=int, help='weight decay (default: 5e-4)')
parser.add_argument('--gpu_id', default=0, type=int)

parser.add_argument('--prune_eval_path', default=None, type=str, metavar='PATH', help='path to eval pruned model')
parser.add_argument('--pruner', default='agp', type=str, help='pruner: agp|taylor|fpgm')

parser.add_argument('--quan_mode', default='int8', help='fp16 int8 best', type=str)
parser.add_argument('--calib_num', type=int, default=1280, help='random seed')

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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(args.model).cuda()
    # model = Debug_Model().cuda()
    # from models.EfficientNet.deeplab_efficiennetb1 import DeepLabv3_plus_debug
    # model = DeepLabv3_plus_debug(n_classes=6).to(device)

    # inputs_shape = (args.batch_size, 3, 512, 512)
    inputs_shape = (args.batch_size, 3, 2048, 2048)

    # x = torch.rand(size=inputs_shape).to(device)
    model.eval()
    # with torch.no_grad():
    #     elapse = time.time()
    #     torch_out = model(x)
    #     elapse = time.time() - elapse
    # print('inputs_shape: %s, output_shape: %s, elapse: %s' % (x.shape, torch_out.shape, elapse))

    train_loader, val_loader, calib_loader = get_data_loader(args)
    criterion = nn.CrossEntropyLoss().to(args.device)

    # flops, params, _ = count_flops_params(model, inputs_shape, verbose=False)
    # val_loss, val_acc, val_acc_cls, val_miou, val_fwavacc, infer_time, fps_avg \
    #     = validate(model, val_loader, criterion, is_trt=False)
    # logging.info(
    #     "Before Quan: val_loss:%.4f\t val_acc:%.4f \tval_acc_cls:%.4f "
    #     "\tval_miou:%.4f \tval_fwavacc:%.4f \tinfer: %.2f ms \t fps: %.2f"
    #     % (val_loss, val_acc, val_acc_cls, val_miou, val_fwavacc, 1000*infer_time, fps_avg)
    # )
    # logging.info('Before quan: flops: %.2f M, params: %.2f M' % (flops/1e6, params/1e6))
    # # evaluate
    # logging.info('[Before Quantization]')
    # miou, avg_time, fps = evaluate_quan(args, model, is_trt=False)
    # # # out_before_quan = evaluate_quan(args, model, is_trt=False)
    # logging.info("Before quantization: [Test ACC = %.4f] [Infer Time = %.4f s] [FPS = %.2f ]" %
    #              (miou, avg_time, fps))
    # print('\n')

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
        inputs_shape,
        config=None,
        calib_data_loader=calib_loader,
        batchsize=inputs_shape[0],  # error
        onnx_path=onnx_path,
        calibration_cache=cache_path,
        extra_layer_bit=extra_layer_bit,
    )
    if not os.path.exists(trt_path):
        engine.compress()
        engine.export_quantized_model(trt_path)
    else:
        engine.load_quantized_model(trt_path)
        logging.info('Directly load quantized model from %s' % trt_path)

    # val_loss, val_acc, val_acc_cls, val_miou, val_fwavacc, infer_time, fps_avg \
    #     = validate(engine, val_loader, criterion, is_trt=True)
    # logging.info(
    #     "After Quan\t val_loss:%.4f\t val_acc:%.4f \tval_acc_cls:%.4f "
    #     "\tval_miou:%.4f \tval_fwavacc:%.4f \tinfer: %.2f ms \t fps: %.2f"
    #     % (val_loss, val_acc, val_acc_cls, val_miou, val_fwavacc, 1000*infer_time, fps_avg)
    # )
    # evaluate
    logging.info('[After Quantization]')
    miou, avg_time, fps = evaluate_quan(args, engine, is_trt=True)
    # out_after_quan = evaluate_quan(args, engine, is_trt=True)
    logging.info("After quantization: [Test ACC = %.4f] [Infer Time = %.4f s] [FPS = %.2f ]" %
                 (miou, avg_time, fps))
    # logging.info('Difference between quan: %f' % (np.sum(out_after_quan - out_before_quan)))


def validate(model, val_loader, criterion, is_trt=True):
    if not is_trt:
        model.eval()
    losses = AverageMeter()
    infer_time = AverageMeter()
    fps_record = AverageMeter()
    gts_all, predictions_all = [], []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            # print(inputs.shape)
            # print(inputs.dtype)
            if is_trt:
                output, trt_infer_time = model.inference(inputs)
                tmp_infer_time = trt_infer_time
                infer_time.update(tmp_infer_time)
                # output = output.reshape(-1, 6, 512, 512)
                print('output_shape: ', output.shape)
            else:
                inputs = inputs.cuda()
                targets = targets.cuda()
                start_time = time.time()
                output = model(inputs)
                tmp_infer_time = time.time() - start_time
                infer_time.update(tmp_infer_time)
            frames = inputs.shape[0]
            tmp_fps = frames / tmp_infer_time
            fps_record.update(tmp_fps)
            loss = criterion(output, targets)
            losses.update(loss.data)
            predictions = output.data.max(1)[1].cpu().numpy()
            gts_all.append(targets.data.cpu().numpy())
            predictions_all.append(predictions)
    val_acc, val_acc_cls, val_miou, val_fwavacc = get_evaluation_score(predictions_all, gts_all, args.num_classes)
    return losses.avg, val_acc, val_acc_cls, val_miou, val_fwavacc, infer_time.avg, fps_record.avg


def get_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 200)
    elif model_name == 'resnet101':
        model = models.resnet101()
        model.fc = nn.Linear(model.fc.in_features, 200)
    elif model_name == 'seg_deeplab_efficientnetb3':
        # model define
        from models.EfficientNet.deeplab_efficiennetb1 import DeepLabv3_plus as deeplab_efficiennetb3
        # from models.EfficientNet.deeplab_efficiennetb1 import DeepLabv3_plus_debug as deeplab_efficiennetb3
        model = deeplab_efficiennetb3(nInputChannels=3, n_classes=args.num_classes, os=16,
                                      pretrained=False, _print=False)
        # load pretrained weights
        # save_point = torch.load('./pretrained/model.pth', map_location=args.device)
        save_point = torch.load('./pretrained/model_1119.pth', map_location=args.device)
        model_param = model.state_dict()
        pretrained_param = save_point['state_dict']
        model_keys = list(model_param.keys())
        pretrained_keys = list(pretrained_param.keys())
        for i, key in enumerate(model_keys):
            # if key == 'efficientnet_features._blocks.0._depthwise_conv.conv.weight':
            #     print(key)
            # if key in state_dict_param.keys():
            #     model_param[key] = state_dict_param[key]
            # else:
            #     new_key = key.split('.')
            #     del new_key[-2]
            #     new_key.insert(0, 'module')
            #     new_key = ".".join(new_key)
            #     assert new_key in state_dict_param.keys()
            #     model_param[key] = state_dict_param[new_key]
            assert model_param[model_keys[i]].shape == pretrained_param[pretrained_keys[i]].shape
            model_param[model_keys[i]] = pretrained_param[pretrained_keys[i]]
        model.load_state_dict(model_param)

        logging.info('Load model weights done!')

        # torch.save(model, 'seg_deeplab_efficientnetb3_full.pt')
        # import sys
        # sys.exit()

    elif model_name == 'seg_deeplab_efficientnetb3_sparse':
        # Define Model
        from models.EfficientNet.deeplab_efficiennetb1 import DeepLabv3_plus as deeplab_efficiennetb3
        model = deeplab_efficiennetb3(nInputChannels=3, n_classes=args.num_classes, os=16,
                                      pretrained=False, _print=False)

        # Acquire Mask
        sparsity_dir = './exp_log/seg_deeplab_efficientnetb3_agp_%s_p2e-05_f0.0002/' % args.sparsity
        model_mask = os.path.join(sparsity_dir, 'mask.pth')
        mask_pt = torch.load(model_mask, map_location=args.device)

        # Define Pruner
        from lib.algorithms.pytorch.pruning import AGPPruner
        criterion = nn.CrossEntropyLoss().to(args.device)
        prune_optimizer = torch.optim.Adam(model.parameters(), lr=2e-05, betas=(0.9, 0.999),
                                           eps=1e-08, weight_decay=1e-4)
        config_list = [{'sparsity': args.sparsity,
                        'op_types': ['Conv2d'],
                        'op_names':
                            ['efficientnet_features._conv_stem.conv',
                             'efficientnet_features._blocks.0._depthwise_conv.conv',
                             'efficientnet_features._blocks.0._se_reduce.conv',
                             'efficientnet_features._blocks.0._project_conv.conv',
                             'efficientnet_features._blocks.1._depthwise_conv.conv',
                             'efficientnet_features._blocks.1._se_reduce.conv',
                             'efficientnet_features._blocks.1._project_conv.conv',
                             'efficientnet_features._blocks.2._expand_conv.conv',
                             'efficientnet_features._blocks.2._depthwise_conv.conv',
                             'efficientnet_features._blocks.2._se_reduce.conv',
                             'efficientnet_features._blocks.2._project_conv.conv',
                             'efficientnet_features._blocks.3._expand_conv.conv',
                             'efficientnet_features._blocks.3._depthwise_conv.conv',
                             'efficientnet_features._blocks.3._se_reduce.conv',
                             'efficientnet_features._blocks.3._project_conv.conv',
                             'efficientnet_features._blocks.4._expand_conv.conv',
                             'efficientnet_features._blocks.4._depthwise_conv.conv',
                             'efficientnet_features._blocks.4._se_reduce.conv',
                             'efficientnet_features._blocks.4._project_conv.conv',
                             'efficientnet_features._blocks.5._expand_conv.conv',
                             'efficientnet_features._blocks.5._depthwise_conv.conv',
                             'efficientnet_features._blocks.5._se_reduce.conv',
                             'efficientnet_features._blocks.5._project_conv.conv',
                             'efficientnet_features._blocks.6._expand_conv.conv',
                             'efficientnet_features._blocks.6._depthwise_conv.conv',
                             'efficientnet_features._blocks.6._se_reduce.conv',
                             'efficientnet_features._blocks.6._project_conv.conv',
                             'efficientnet_features._blocks.7._expand_conv.conv',
                             'efficientnet_features._blocks.7._depthwise_conv.conv',
                             'efficientnet_features._blocks.7._se_reduce.conv',
                             'efficientnet_features._blocks.7._project_conv.conv',
                             'efficientnet_features._blocks.8._expand_conv.conv',
                             'efficientnet_features._blocks.8._depthwise_conv.conv',
                             'efficientnet_features._blocks.8._se_reduce.conv',
                             'efficientnet_features._blocks.8._project_conv.conv',
                             'efficientnet_features._blocks.9._expand_conv.conv',
                             'efficientnet_features._blocks.9._depthwise_conv.conv',
                             'efficientnet_features._blocks.9._se_reduce.conv',
                             'efficientnet_features._blocks.9._project_conv.conv',
                             'efficientnet_features._blocks.10._expand_conv.conv',
                             'efficientnet_features._blocks.10._depthwise_conv.conv',
                             'efficientnet_features._blocks.10._se_reduce.conv',
                             'efficientnet_features._blocks.10._project_conv.conv',
                             'efficientnet_features._blocks.11._expand_conv.conv',
                             'efficientnet_features._blocks.11._depthwise_conv.conv',
                             'efficientnet_features._blocks.11._se_reduce.conv',
                             'efficientnet_features._blocks.11._project_conv.conv',
                             'efficientnet_features._blocks.12._expand_conv.conv',
                             'efficientnet_features._blocks.12._depthwise_conv.conv',
                             'efficientnet_features._blocks.12._se_reduce.conv',
                             'efficientnet_features._blocks.12._project_conv.conv',
                             'efficientnet_features._blocks.13._expand_conv.conv',
                             'efficientnet_features._blocks.13._depthwise_conv.conv',
                             'efficientnet_features._blocks.13._se_reduce.conv',
                             'efficientnet_features._blocks.13._project_conv.conv',
                             'efficientnet_features._blocks.14._expand_conv.conv',
                             'efficientnet_features._blocks.14._depthwise_conv.conv',
                             'efficientnet_features._blocks.14._se_reduce.conv',
                             'efficientnet_features._blocks.14._project_conv.conv',
                             'efficientnet_features._blocks.15._expand_conv.conv',
                             'efficientnet_features._blocks.15._depthwise_conv.conv',
                             'efficientnet_features._blocks.15._se_reduce.conv',
                             'efficientnet_features._blocks.15._project_conv.conv',
                             'efficientnet_features._blocks.16._expand_conv.conv',
                             'efficientnet_features._blocks.16._depthwise_conv.conv',
                             'efficientnet_features._blocks.16._se_reduce.conv',
                             'efficientnet_features._blocks.16._project_conv.conv',
                             'efficientnet_features._blocks.17._expand_conv.conv',
                             'efficientnet_features._blocks.17._depthwise_conv.conv',
                             'efficientnet_features._blocks.17._se_reduce.conv',
                             'efficientnet_features._blocks.17._project_conv.conv',
                             'efficientnet_features._blocks.18._expand_conv.conv',
                             'efficientnet_features._blocks.18._depthwise_conv.conv',
                             'efficientnet_features._blocks.18._se_reduce.conv',
                             'efficientnet_features._blocks.18._project_conv.conv',
                             'efficientnet_features._blocks.19._expand_conv.conv',
                             'efficientnet_features._blocks.19._depthwise_conv.conv',
                             'efficientnet_features._blocks.19._se_reduce.conv',
                             'efficientnet_features._blocks.19._project_conv.conv',
                             'efficientnet_features._blocks.20._expand_conv.conv',
                             'efficientnet_features._blocks.20._depthwise_conv.conv',
                             'efficientnet_features._blocks.20._se_reduce.conv',
                             'efficientnet_features._blocks.20._project_conv.conv',
                             'efficientnet_features._blocks.21._expand_conv.conv',
                             'efficientnet_features._blocks.21._depthwise_conv.conv',
                             'efficientnet_features._blocks.21._se_reduce.conv',
                             'efficientnet_features._blocks.21._project_conv.conv',
                             'efficientnet_features._blocks.22._expand_conv.conv',
                             'efficientnet_features._blocks.22._depthwise_conv.conv',
                             'efficientnet_features._blocks.22._se_reduce.conv',
                             'efficientnet_features._blocks.22._project_conv.conv',
                             'efficientnet_features._blocks.23._expand_conv.conv',
                             'efficientnet_features._blocks.23._depthwise_conv.conv',
                             'efficientnet_features._blocks.23._se_reduce.conv',
                             'efficientnet_features._blocks.23._project_conv.conv',
                             'efficientnet_features._blocks.24._expand_conv.conv',
                             'efficientnet_features._blocks.24._depthwise_conv.conv',
                             'efficientnet_features._blocks.24._se_reduce.conv',
                             'efficientnet_features._blocks.24._project_conv.conv',
                             'efficientnet_features._blocks.25._expand_conv.conv',
                             'efficientnet_features._blocks.25._depthwise_conv.conv',
                             'efficientnet_features._blocks.25._se_reduce.conv',
                             'efficientnet_features._blocks.25._project_conv.conv',
                             'efficientnet_features._conv_head.conv',
                             'efficientnet_features._conv_head', 'conv1', 'conv2', 'last_conv', 'last_conv.0',
                             'last_conv.1', 'last_conv.2', 'last_conv.3', 'last_conv.4', 'last_conv.5'
                             ]
                        }]
        pruner = AGPPruner(
            model,
            config_list,
            prune_optimizer,
            None,
            criterion,
            num_iterations=1,
            epochs_per_iteration=1,
            pruning_algorithm='taylorfo',
        )

        # Prune Model
        model = get_pruned_model(pruner, model, mask_pt)
        pruned_weights = torch.load(os.path.join(sparsity_dir, 'model_pruned_best_miou.pth'))
        model.load_state_dict(pruned_weights['state_dict'])
        logging.info('Load pruned models from %s done!' % sparsity_dir)

        # torch.save(model, 'seg_deeplab_efficientnetb3_s%s_full.pt' % args.sparsity)
        # import sys
        # sys.exit()


    else:
        raise NotImplementedError('Not Support {}'.format(model_name))
    return model


def get_data_loader(args):
    print("loading dataset ...")
    train_data = Reader(args, mode='train')
    print("Train set samples: ", len(train_data))
    val_data = Reader(args, mode='test')
    print("Validation set samples: ", len(val_data))
    test_data = Reader(args, mode='eval')
    print("Test set samples: ", len(test_data))
    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              drop_last=True, num_workers=args.workers if torch.cuda.is_available() else 0)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            drop_last=False, num_workers=args.workers if torch.cuda.is_available() else 0)
    eval_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            drop_last=False, num_workers=args.workers if torch.cuda.is_available() else 0)

    # n_train = len(train_data)
    # indices = list(range(n_train))
    # random.shuffle(indices)
    # calib_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:args.calib_num])
    # calib_loader = data.DataLoader(train_data, batch_size=args.batch_size, sampler=calib_sampler)

    return train_loader, val_loader, eval_loader


if __name__ == "__main__":
    run_func(args, main)
