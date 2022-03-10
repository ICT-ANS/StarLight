# from libs.compression.utils.counter import count_flops_params
import argparse
import copy
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataReader.dataset import Reader
from lib.algorithms.pytorch.pruning import (AGPPruner)
from utils import *
from utils.utils import AverageMeter, get_evaluation_score


in_replace_dic = {
    'efficientnet_features._conv_stem.conv': (None,),
    'efficientnet_features._bn0': ('efficientnet_features._conv_stem.conv',),
    'efficientnet_features._blocks.0._depthwise_conv.conv': ('efficientnet_features._conv_stem.conv',),
    # layer  0
    'efficientnet_features._blocks.0._bn1': ('efficientnet_features._blocks.0._depthwise_conv.conv',),
    'efficientnet_features._blocks.0._se_reduce.conv': ('efficientnet_features._blocks.0._depthwise_conv.conv',),
    'efficientnet_features._blocks.0._se_expand.conv': ('efficientnet_features._blocks.0._se_reduce.conv',),
    'efficientnet_features._blocks.0._project_conv.conv': ('efficientnet_features._blocks.0._depthwise_conv.conv',),
    'efficientnet_features._blocks.0._bn2': ('efficientnet_features._blocks.0._project_conv.conv',),
    'efficientnet_features._blocks.1._depthwise_conv.conv': ('efficientnet_features._blocks.0._project_conv.conv',),
    # layer  1
    'efficientnet_features._blocks.1._bn1': ('efficientnet_features._blocks.1._depthwise_conv.conv',),
    'efficientnet_features._blocks.1._se_reduce.conv': ('efficientnet_features._blocks.1._depthwise_conv.conv',),
    'efficientnet_features._blocks.1._se_expand.conv': ('efficientnet_features._blocks.1._se_reduce.conv',),
    'efficientnet_features._blocks.1._project_conv.conv': ('efficientnet_features._blocks.1._depthwise_conv.conv',),
    'efficientnet_features._blocks.1._bn2': ('efficientnet_features._blocks.1._project_conv.conv',),
    'efficientnet_features._blocks.2._expand_conv.conv': ('efficientnet_features._blocks.1._project_conv.conv',),
    # layer  2
    'efficientnet_features._blocks.2._bn0': ('efficientnet_features._blocks.2._expand_conv.conv',),
    'efficientnet_features._blocks.2._depthwise_conv.conv': ('efficientnet_features._blocks.2._expand_conv.conv',),
    'efficientnet_features._blocks.2._bn1': ('efficientnet_features._blocks.2._depthwise_conv.conv',),
    'efficientnet_features._blocks.2._se_reduce.conv': ('efficientnet_features._blocks.2._depthwise_conv.conv',),
    'efficientnet_features._blocks.2._se_expand.conv': ('efficientnet_features._blocks.2._se_reduce.conv',),
    'efficientnet_features._blocks.2._project_conv.conv': ('efficientnet_features._blocks.2._depthwise_conv.conv',),
    'efficientnet_features._blocks.2._bn2': ('efficientnet_features._blocks.2._project_conv.conv',),
    'efficientnet_features._blocks.3._expand_conv.conv': ('efficientnet_features._blocks.2._project_conv.conv',),
    # layer  3
    'efficientnet_features._blocks.3._bn0': ('efficientnet_features._blocks.3._expand_conv.conv',),
    'efficientnet_features._blocks.3._depthwise_conv.conv': ('efficientnet_features._blocks.3._expand_conv.conv',),
    'efficientnet_features._blocks.3._bn1': ('efficientnet_features._blocks.3._depthwise_conv.conv',),
    'efficientnet_features._blocks.3._se_reduce.conv': ('efficientnet_features._blocks.3._depthwise_conv.conv',),
    'efficientnet_features._blocks.3._se_expand.conv': ('efficientnet_features._blocks.3._se_reduce.conv',),
    'efficientnet_features._blocks.3._project_conv.conv': ('efficientnet_features._blocks.3._depthwise_conv.conv',),
    'efficientnet_features._blocks.3._bn2': ('efficientnet_features._blocks.3._project_conv.conv',),
    'efficientnet_features._blocks.4._expand_conv.conv': ('efficientnet_features._blocks.3._project_conv.conv',),
    # layer  4
    'efficientnet_features._blocks.4._bn0': ('efficientnet_features._blocks.4._expand_conv.conv',),
    'efficientnet_features._blocks.4._depthwise_conv.conv': ('efficientnet_features._blocks.4._expand_conv.conv',),
    'efficientnet_features._blocks.4._bn1': ('efficientnet_features._blocks.4._depthwise_conv.conv',),
    'efficientnet_features._blocks.4._se_reduce.conv': ('efficientnet_features._blocks.4._depthwise_conv.conv',),
    'efficientnet_features._blocks.4._se_expand.conv': ('efficientnet_features._blocks.4._se_reduce.conv',),
    'efficientnet_features._blocks.4._project_conv.conv': ('efficientnet_features._blocks.4._depthwise_conv.conv',),
    'efficientnet_features._blocks.4._bn2': ('efficientnet_features._blocks.4._project_conv.conv',),
    'efficientnet_features._blocks.5._expand_conv.conv': ('efficientnet_features._blocks.4._project_conv.conv',),
    # layer  5
    'efficientnet_features._blocks.5._bn0': ('efficientnet_features._blocks.5._expand_conv.conv',),
    'efficientnet_features._blocks.5._depthwise_conv.conv': ('efficientnet_features._blocks.5._expand_conv.conv',),
    'efficientnet_features._blocks.5._bn1': ('efficientnet_features._blocks.5._depthwise_conv.conv',),
    'efficientnet_features._blocks.5._se_reduce.conv': ('efficientnet_features._blocks.5._depthwise_conv.conv',),
    'efficientnet_features._blocks.5._se_expand.conv': ('efficientnet_features._blocks.5._se_reduce.conv',),
    'efficientnet_features._blocks.5._project_conv.conv': ('efficientnet_features._blocks.5._depthwise_conv.conv',),
    'efficientnet_features._blocks.5._bn2': ('efficientnet_features._blocks.5._project_conv.conv',),
    'efficientnet_features._blocks.6._expand_conv.conv': ('efficientnet_features._blocks.5._project_conv.conv',),
    # layer  6
    'efficientnet_features._blocks.6._bn0': ('efficientnet_features._blocks.6._expand_conv.conv',),
    'efficientnet_features._blocks.6._depthwise_conv.conv': ('efficientnet_features._blocks.6._expand_conv.conv',),
    'efficientnet_features._blocks.6._bn1': ('efficientnet_features._blocks.6._depthwise_conv.conv',),
    'efficientnet_features._blocks.6._se_reduce.conv': ('efficientnet_features._blocks.6._depthwise_conv.conv',),
    'efficientnet_features._blocks.6._se_expand.conv': ('efficientnet_features._blocks.6._se_reduce.conv',),
    'efficientnet_features._blocks.6._project_conv.conv': ('efficientnet_features._blocks.6._depthwise_conv.conv',),
    'efficientnet_features._blocks.6._bn2': ('efficientnet_features._blocks.6._project_conv.conv',),
    'efficientnet_features._blocks.7._expand_conv.conv': ('efficientnet_features._blocks.6._project_conv.conv',),
    # layer  7
    'efficientnet_features._blocks.7._bn0': ('efficientnet_features._blocks.7._expand_conv.conv',),
    'efficientnet_features._blocks.7._depthwise_conv.conv': ('efficientnet_features._blocks.7._expand_conv.conv',),
    'efficientnet_features._blocks.7._bn1': ('efficientnet_features._blocks.7._depthwise_conv.conv',),
    'efficientnet_features._blocks.7._se_reduce.conv': ('efficientnet_features._blocks.7._depthwise_conv.conv',),
    'efficientnet_features._blocks.7._se_expand.conv': ('efficientnet_features._blocks.7._se_reduce.conv',),
    'efficientnet_features._blocks.7._project_conv.conv': ('efficientnet_features._blocks.7._depthwise_conv.conv',),
    'efficientnet_features._blocks.7._bn2': ('efficientnet_features._blocks.7._project_conv.conv',),
    'efficientnet_features._blocks.8._expand_conv.conv': ('efficientnet_features._blocks.7._project_conv.conv',),
    # layer  8
    'efficientnet_features._blocks.8._bn0': ('efficientnet_features._blocks.8._expand_conv.conv',),
    'efficientnet_features._blocks.8._depthwise_conv.conv': ('efficientnet_features._blocks.8._expand_conv.conv',),
    'efficientnet_features._blocks.8._bn1': ('efficientnet_features._blocks.8._depthwise_conv.conv',),
    'efficientnet_features._blocks.8._se_reduce.conv': ('efficientnet_features._blocks.8._depthwise_conv.conv',),
    'efficientnet_features._blocks.8._se_expand.conv': ('efficientnet_features._blocks.8._se_reduce.conv',),
    'efficientnet_features._blocks.8._project_conv.conv': ('efficientnet_features._blocks.8._depthwise_conv.conv',),
    'efficientnet_features._blocks.8._bn2': ('efficientnet_features._blocks.8._project_conv.conv',),
    'efficientnet_features._blocks.9._expand_conv.conv': ('efficientnet_features._blocks.8._project_conv.conv',),
    # layer  9
    'efficientnet_features._blocks.9._bn0': ('efficientnet_features._blocks.9._expand_conv.conv',),
    'efficientnet_features._blocks.9._depthwise_conv.conv': ('efficientnet_features._blocks.9._expand_conv.conv',),
    'efficientnet_features._blocks.9._bn1': ('efficientnet_features._blocks.9._depthwise_conv.conv',),
    'efficientnet_features._blocks.9._se_reduce.conv': ('efficientnet_features._blocks.9._depthwise_conv.conv',),
    'efficientnet_features._blocks.9._se_expand.conv': ('efficientnet_features._blocks.9._se_reduce.conv',),
    'efficientnet_features._blocks.9._project_conv.conv': ('efficientnet_features._blocks.9._depthwise_conv.conv',),
    'efficientnet_features._blocks.9._bn2': ('efficientnet_features._blocks.9._project_conv.conv',),
    'efficientnet_features._blocks.10._expand_conv.conv': ('efficientnet_features._blocks.9._project_conv.conv',),
    # layer  10
    'efficientnet_features._blocks.10._bn0': ('efficientnet_features._blocks.10._expand_conv.conv',),
    'efficientnet_features._blocks.10._depthwise_conv.conv': (
        'efficientnet_features._blocks.10._expand_conv.conv',),
    'efficientnet_features._blocks.10._bn1': ('efficientnet_features._blocks.10._depthwise_conv.conv',),
    'efficientnet_features._blocks.10._se_reduce.conv': ('efficientnet_features._blocks.10._depthwise_conv.conv',),
    'efficientnet_features._blocks.10._se_expand.conv': ('efficientnet_features._blocks.10._se_reduce.conv',),
    'efficientnet_features._blocks.10._project_conv.conv': (
        'efficientnet_features._blocks.10._depthwise_conv.conv',),
    'efficientnet_features._blocks.10._bn2': ('efficientnet_features._blocks.10._project_conv.conv',),
    'efficientnet_features._blocks.11._expand_conv.conv': ('efficientnet_features._blocks.10._project_conv.conv',),
    # layer  11
    'efficientnet_features._blocks.11._bn0': ('efficientnet_features._blocks.11._expand_conv.conv',),
    'efficientnet_features._blocks.11._depthwise_conv.conv': (
        'efficientnet_features._blocks.11._expand_conv.conv',),
    'efficientnet_features._blocks.11._bn1': ('efficientnet_features._blocks.11._depthwise_conv.conv',),
    'efficientnet_features._blocks.11._se_reduce.conv': ('efficientnet_features._blocks.11._depthwise_conv.conv',),
    'efficientnet_features._blocks.11._se_expand.conv': ('efficientnet_features._blocks.11._se_reduce.conv',),
    'efficientnet_features._blocks.11._project_conv.conv': (
        'efficientnet_features._blocks.11._depthwise_conv.conv',),
    'efficientnet_features._blocks.11._bn2': ('efficientnet_features._blocks.11._project_conv.conv',),
    'efficientnet_features._blocks.12._expand_conv.conv': ('efficientnet_features._blocks.11._project_conv.conv',),
    # layer  12
    'efficientnet_features._blocks.12._bn0': ('efficientnet_features._blocks.12._expand_conv.conv',),
    'efficientnet_features._blocks.12._depthwise_conv.conv': (
        'efficientnet_features._blocks.12._expand_conv.conv',),
    'efficientnet_features._blocks.12._bn1': ('efficientnet_features._blocks.12._depthwise_conv.conv',),
    'efficientnet_features._blocks.12._se_reduce.conv': ('efficientnet_features._blocks.12._depthwise_conv.conv',),
    'efficientnet_features._blocks.12._se_expand.conv': ('efficientnet_features._blocks.12._se_reduce.conv',),
    'efficientnet_features._blocks.12._project_conv.conv': (
        'efficientnet_features._blocks.12._depthwise_conv.conv',),
    'efficientnet_features._blocks.12._bn2': ('efficientnet_features._blocks.12._project_conv.conv',),
    'efficientnet_features._blocks.13._expand_conv.conv': ('efficientnet_features._blocks.12._project_conv.conv',),
    # layer  13
    'efficientnet_features._blocks.13._bn0': ('efficientnet_features._blocks.13._expand_conv.conv',),
    'efficientnet_features._blocks.13._depthwise_conv.conv': (
        'efficientnet_features._blocks.13._expand_conv.conv',),
    'efficientnet_features._blocks.13._bn1': ('efficientnet_features._blocks.13._depthwise_conv.conv',),
    'efficientnet_features._blocks.13._se_reduce.conv': ('efficientnet_features._blocks.13._depthwise_conv.conv',),
    'efficientnet_features._blocks.13._se_expand.conv': ('efficientnet_features._blocks.13._se_reduce.conv',),
    'efficientnet_features._blocks.13._project_conv.conv': (
        'efficientnet_features._blocks.13._depthwise_conv.conv',),
    'efficientnet_features._blocks.13._bn2': ('efficientnet_features._blocks.13._project_conv.conv',),
    'efficientnet_features._blocks.14._expand_conv.conv': ('efficientnet_features._blocks.13._project_conv.conv',),
    # layer  14
    'efficientnet_features._blocks.14._bn0': ('efficientnet_features._blocks.14._expand_conv.conv',),
    'efficientnet_features._blocks.14._depthwise_conv.conv': (
        'efficientnet_features._blocks.14._expand_conv.conv',),
    'efficientnet_features._blocks.14._bn1': ('efficientnet_features._blocks.14._depthwise_conv.conv',),
    'efficientnet_features._blocks.14._se_reduce.conv': ('efficientnet_features._blocks.14._depthwise_conv.conv',),
    'efficientnet_features._blocks.14._se_expand.conv': ('efficientnet_features._blocks.14._se_reduce.conv',),
    'efficientnet_features._blocks.14._project_conv.conv': (
        'efficientnet_features._blocks.14._depthwise_conv.conv',),
    'efficientnet_features._blocks.14._bn2': ('efficientnet_features._blocks.14._project_conv.conv',),
    'efficientnet_features._blocks.15._expand_conv.conv': ('efficientnet_features._blocks.14._project_conv.conv',),
    # layer  15
    'efficientnet_features._blocks.15._bn0': ('efficientnet_features._blocks.15._expand_conv.conv',),
    'efficientnet_features._blocks.15._depthwise_conv.conv': (
        'efficientnet_features._blocks.15._expand_conv.conv',),
    'efficientnet_features._blocks.15._bn1': ('efficientnet_features._blocks.15._depthwise_conv.conv',),
    'efficientnet_features._blocks.15._se_reduce.conv': ('efficientnet_features._blocks.15._depthwise_conv.conv',),
    'efficientnet_features._blocks.15._se_expand.conv': ('efficientnet_features._blocks.15._se_reduce.conv',),
    'efficientnet_features._blocks.15._project_conv.conv': (
        'efficientnet_features._blocks.15._depthwise_conv.conv',),
    'efficientnet_features._blocks.15._bn2': ('efficientnet_features._blocks.15._project_conv.conv',),
    'efficientnet_features._blocks.16._expand_conv.conv': ('efficientnet_features._blocks.15._project_conv.conv',),
    # layer  16
    'efficientnet_features._blocks.16._bn0': ('efficientnet_features._blocks.16._expand_conv.conv',),
    'efficientnet_features._blocks.16._depthwise_conv.conv': (
        'efficientnet_features._blocks.16._expand_conv.conv',),
    'efficientnet_features._blocks.16._bn1': ('efficientnet_features._blocks.16._depthwise_conv.conv',),
    'efficientnet_features._blocks.16._se_reduce.conv': ('efficientnet_features._blocks.16._depthwise_conv.conv',),
    'efficientnet_features._blocks.16._se_expand.conv': ('efficientnet_features._blocks.16._se_reduce.conv',),
    'efficientnet_features._blocks.16._project_conv.conv': (
        'efficientnet_features._blocks.16._depthwise_conv.conv',),
    'efficientnet_features._blocks.16._bn2': ('efficientnet_features._blocks.16._project_conv.conv',),
    'efficientnet_features._blocks.17._expand_conv.conv': ('efficientnet_features._blocks.16._project_conv.conv',),
    # layer  17
    'efficientnet_features._blocks.17._bn0': ('efficientnet_features._blocks.17._expand_conv.conv',),
    'efficientnet_features._blocks.17._depthwise_conv.conv': (
        'efficientnet_features._blocks.17._expand_conv.conv',),
    'efficientnet_features._blocks.17._bn1': ('efficientnet_features._blocks.17._depthwise_conv.conv',),
    'efficientnet_features._blocks.17._se_reduce.conv': ('efficientnet_features._blocks.17._depthwise_conv.conv',),
    'efficientnet_features._blocks.17._se_expand.conv': ('efficientnet_features._blocks.17._se_reduce.conv',),
    'efficientnet_features._blocks.17._project_conv.conv': (
        'efficientnet_features._blocks.17._depthwise_conv.conv',),
    'efficientnet_features._blocks.17._bn2': ('efficientnet_features._blocks.17._project_conv.conv',),
    'efficientnet_features._blocks.18._expand_conv.conv': ('efficientnet_features._blocks.17._project_conv.conv',),
    # layer  18
    'efficientnet_features._blocks.18._bn0': ('efficientnet_features._blocks.18._expand_conv.conv',),
    'efficientnet_features._blocks.18._depthwise_conv.conv': (
        'efficientnet_features._blocks.18._expand_conv.conv',),
    'efficientnet_features._blocks.18._bn1': ('efficientnet_features._blocks.18._depthwise_conv.conv',),
    'efficientnet_features._blocks.18._se_reduce.conv': ('efficientnet_features._blocks.18._depthwise_conv.conv',),
    'efficientnet_features._blocks.18._se_expand.conv': ('efficientnet_features._blocks.18._se_reduce.conv',),
    'efficientnet_features._blocks.18._project_conv.conv': (
        'efficientnet_features._blocks.18._depthwise_conv.conv',),
    'efficientnet_features._blocks.18._bn2': ('efficientnet_features._blocks.18._project_conv.conv',),
    'efficientnet_features._blocks.19._expand_conv.conv': ('efficientnet_features._blocks.18._project_conv.conv',),
    # layer  19
    'efficientnet_features._blocks.19._bn0': ('efficientnet_features._blocks.19._expand_conv.conv',),
    'efficientnet_features._blocks.19._depthwise_conv.conv': (
        'efficientnet_features._blocks.19._expand_conv.conv',),
    'efficientnet_features._blocks.19._bn1': ('efficientnet_features._blocks.19._depthwise_conv.conv',),
    'efficientnet_features._blocks.19._se_reduce.conv': ('efficientnet_features._blocks.19._depthwise_conv.conv',),
    'efficientnet_features._blocks.19._se_expand.conv': ('efficientnet_features._blocks.19._se_reduce.conv',),
    'efficientnet_features._blocks.19._project_conv.conv': (
        'efficientnet_features._blocks.19._depthwise_conv.conv',),
    'efficientnet_features._blocks.19._bn2': ('efficientnet_features._blocks.19._project_conv.conv',),
    'efficientnet_features._blocks.20._expand_conv.conv': ('efficientnet_features._blocks.19._project_conv.conv',),
    # layer  20
    'efficientnet_features._blocks.20._bn0': ('efficientnet_features._blocks.20._expand_conv.conv',),
    'efficientnet_features._blocks.20._depthwise_conv.conv': (
        'efficientnet_features._blocks.20._expand_conv.conv',),
    'efficientnet_features._blocks.20._bn1': ('efficientnet_features._blocks.20._depthwise_conv.conv',),
    'efficientnet_features._blocks.20._se_reduce.conv': ('efficientnet_features._blocks.20._depthwise_conv.conv',),
    'efficientnet_features._blocks.20._se_expand.conv': ('efficientnet_features._blocks.20._se_reduce.conv',),
    'efficientnet_features._blocks.20._project_conv.conv': (
        'efficientnet_features._blocks.20._depthwise_conv.conv',),
    'efficientnet_features._blocks.20._bn2': ('efficientnet_features._blocks.20._project_conv.conv',),
    'efficientnet_features._blocks.21._expand_conv.conv': ('efficientnet_features._blocks.20._project_conv.conv',),
    # layer  21
    'efficientnet_features._blocks.21._bn0': ('efficientnet_features._blocks.21._expand_conv.conv',),
    'efficientnet_features._blocks.21._depthwise_conv.conv': (
        'efficientnet_features._blocks.21._expand_conv.conv',),
    'efficientnet_features._blocks.21._bn1': ('efficientnet_features._blocks.21._depthwise_conv.conv',),
    'efficientnet_features._blocks.21._se_reduce.conv': ('efficientnet_features._blocks.21._depthwise_conv.conv',),
    'efficientnet_features._blocks.21._se_expand.conv': ('efficientnet_features._blocks.21._se_reduce.conv',),
    'efficientnet_features._blocks.21._project_conv.conv': (
        'efficientnet_features._blocks.21._depthwise_conv.conv',),
    'efficientnet_features._blocks.21._bn2': ('efficientnet_features._blocks.21._project_conv.conv',),
    'efficientnet_features._blocks.22._expand_conv.conv': ('efficientnet_features._blocks.21._project_conv.conv',),
    # layer  22
    'efficientnet_features._blocks.22._bn0': ('efficientnet_features._blocks.22._expand_conv.conv',),
    'efficientnet_features._blocks.22._depthwise_conv.conv': (
        'efficientnet_features._blocks.22._expand_conv.conv',),
    'efficientnet_features._blocks.22._bn1': ('efficientnet_features._blocks.22._depthwise_conv.conv',),
    'efficientnet_features._blocks.22._se_reduce.conv': ('efficientnet_features._blocks.22._depthwise_conv.conv',),
    'efficientnet_features._blocks.22._se_expand.conv': ('efficientnet_features._blocks.22._se_reduce.conv',),
    'efficientnet_features._blocks.22._project_conv.conv': (
        'efficientnet_features._blocks.22._depthwise_conv.conv',),
    'efficientnet_features._blocks.22._bn2': ('efficientnet_features._blocks.22._project_conv.conv',),
    'efficientnet_features._blocks.23._expand_conv.conv': ('efficientnet_features._blocks.22._project_conv.conv',),
    # layer  23
    'efficientnet_features._blocks.23._bn0': ('efficientnet_features._blocks.23._expand_conv.conv',),
    'efficientnet_features._blocks.23._depthwise_conv.conv': (
        'efficientnet_features._blocks.23._expand_conv.conv',),
    'efficientnet_features._blocks.23._bn1': ('efficientnet_features._blocks.23._depthwise_conv.conv',),
    'efficientnet_features._blocks.23._se_reduce.conv': ('efficientnet_features._blocks.23._depthwise_conv.conv',),
    'efficientnet_features._blocks.23._se_expand.conv': ('efficientnet_features._blocks.23._se_reduce.conv',),
    'efficientnet_features._blocks.23._project_conv.conv': (
        'efficientnet_features._blocks.23._depthwise_conv.conv',),
    'efficientnet_features._blocks.23._bn2': ('efficientnet_features._blocks.23._project_conv.conv',),
    'efficientnet_features._blocks.24._expand_conv.conv': ('efficientnet_features._blocks.23._project_conv.conv',),
    # layer  24
    'efficientnet_features._blocks.24._bn0': ('efficientnet_features._blocks.24._expand_conv.conv',),
    'efficientnet_features._blocks.24._depthwise_conv.conv': (
        'efficientnet_features._blocks.24._expand_conv.conv',),
    'efficientnet_features._blocks.24._bn1': ('efficientnet_features._blocks.24._depthwise_conv.conv',),
    'efficientnet_features._blocks.24._se_reduce.conv': ('efficientnet_features._blocks.24._depthwise_conv.conv',),
    'efficientnet_features._blocks.24._se_expand.conv': ('efficientnet_features._blocks.24._se_reduce.conv',),
    'efficientnet_features._blocks.24._project_conv.conv': (
        'efficientnet_features._blocks.24._depthwise_conv.conv',),
    'efficientnet_features._blocks.24._bn2': ('efficientnet_features._blocks.24._project_conv.conv',),
    'efficientnet_features._blocks.25._expand_conv.conv': ('efficientnet_features._blocks.24._project_conv.conv',),
    # layer  25
    'efficientnet_features._blocks.25._bn0': ('efficientnet_features._blocks.25._expand_conv.conv',),
    'efficientnet_features._blocks.25._depthwise_conv.conv': (
        'efficientnet_features._blocks.25._expand_conv.conv',),
    'efficientnet_features._blocks.25._bn1': ('efficientnet_features._blocks.25._depthwise_conv.conv',),
    'efficientnet_features._blocks.25._se_reduce.conv': ('efficientnet_features._blocks.25._depthwise_conv.conv',),
    'efficientnet_features._blocks.25._se_expand.conv': ('efficientnet_features._blocks.25._se_reduce.conv',),
    'efficientnet_features._blocks.25._project_conv.conv': (
        'efficientnet_features._blocks.25._depthwise_conv.conv',),
    'efficientnet_features._blocks.25._bn2': ('efficientnet_features._blocks.25._project_conv.conv',),
    'efficientnet_features._conv_head.conv': ('efficientnet_features._blocks.25._project_conv.conv',),
    'efficientnet_features._bn1': ('efficientnet_features._conv_head.conv',),
    'conv1': ('efficientnet_features._conv_head.conv',),
    'bn1': ('conv1',),
    'conv2': ('efficientnet_features._blocks.4._project_conv.conv',),
    'bn2': ('conv2',),
    'last_conv.0': ('conv1', 'conv2'),
    'last_conv.1': ('last_conv.0',),
    'last_conv.3': ('last_conv.0',),
    'last_conv.4': ('last_conv.3',),
    'last_conv.6': ('last_conv.3',),
}


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion > max_iter:
        return init_lr

    lr = init_lr * (1 - iteraion / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_pruned_model(pruner, model, mask_pt):
    prune_dic = {}

    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                module.weight_mask = mask_pt[name]['weight']
                module.bias_mask = mask_pt[name]['bias']
                prune_dic[module.name] = torch.mean(module.weight_mask, dim=(1, 2, 3)).int().cpu().numpy().tolist()
                # if '_depthwise_conv' in name:
                #     record_weight_mask = module.weight_mask
            elif '_se_expand.conv' in name:
                depthwise_countpart = mask_pt[name.replace('_se_expand', '_depthwise_conv')]['weight']
                prune_dic[name] = torch.mean(depthwise_countpart, dim=(1, 2, 3)).int().cpu().numpy().tolist()
            else:
                pass
        # mask_pt = torch.load(masks_file, map_location='cpu')
        # for name, module in model.named_modules():
        #     if hasattr(module, 'weight_mask'):
        #         module.weight_mask = mask_pt[name]['weight']
        #         module.bias_mask = mask_pt[name]['bias']
        #
        # for name, module in model.named_modules():
        #     if hasattr(module, 'weight_mask'):
        #         prune_dic[module.name] = torch.mean(module.weight_mask, dim=(1, 2, 3)).int().cpu().numpy().tolist()
        #         if '_depthwise_conv' in name:
        #             record_weight_mask = module.weight_mask
        #         # if name == 'efficientnet_features._blocks.1._depthwise_conv.conv':
        #         #     break
        #     elif '_expand_conv.conv' in name:
        #         out_channels = module.out_channels
        #         prune_dic[name] = [1 for _ in range(out_channels)]
        #     elif '_se_expand.conv' in name:
        #         prune_dic[name] = torch.mean(record_weight_mask, dim=(1, 2, 3)).int().cpu().numpy().tolist()
        #     else:
        #         pass

        pruner._unwrap_model()
        model = copy.deepcopy(pruner.bound_model)
        pruner._wrap_model()
        for name, module in model.named_modules():
            if name in in_replace_dic:
                # if name == 'conv1':
                #     print(name)
                device = module.weight.device
                super_module, leaf_module = get_module_by_name(model, name)
                if type(module) == nn.BatchNorm2d:
                    mask = prune_dic[in_replace_dic[name][0]]
                    mask = torch.Tensor(mask).long().to(device)
                    compressed_module = replace_batchnorm2d(leaf_module, mask)
                if type(module) == nn.Conv2d:
                    if in_replace_dic[name][0] == None:
                        input_mask = None
                    else:
                        input_mask = []
                        for x in in_replace_dic[name]:
                            if type(x) is int:
                                input_mask += [1] * x
                            else:
                                input_mask += prune_dic[x]
                    output_mask = None if name not in prune_dic else prune_dic[name]
                    if input_mask is not None:
                        input_mask = torch.Tensor(input_mask).long().to(device)
                    if output_mask is not None:
                        output_mask = torch.Tensor(output_mask).long().to(device)
                    compressed_module = replace_conv2d(module, input_mask, output_mask)
                setattr(super_module, name.split('.')[-1], compressed_module)
        return model


def get_module_by_name(model, module_name):
    """
    Get a module specified by its module name

    Parameters
    ----------
    model : pytorch model
        the pytorch model from which to get its module
    module_name : str
        the name of the required module

    Returns
    -------
    module, module
        the parent module of the required module, the required module
    """
    name_list = module_name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    leaf_module = getattr(model, name_list[-1])
    return model, leaf_module


def get_index(mask):
    index = []
    for i in range(len(mask)):
        if mask[i] == 1:
            index.append(i)
    return torch.Tensor(index).long().to(mask.device)


def replace_batchnorm2d(norm, mask):
    """
    Parameters
    ----------
    norm : torch.nn.BatchNorm2d
        The batchnorm module to be replace
    mask : ModuleMasks
        The masks of this module

    Returns
    -------
    torch.nn.BatchNorm2d
        The new batchnorm module
    """
    index = get_index(mask)
    num_features = len(index)
    new_norm = torch.nn.BatchNorm2d(num_features=num_features, eps=norm.eps, momentum=norm.momentum, affine=norm.affine, track_running_stats=norm.track_running_stats)
    # assign weights
    new_norm.weight.data = torch.index_select(norm.weight.data, 0, index)
    new_norm.bias.data = torch.index_select(norm.bias.data, 0, index)
    if norm.track_running_stats:
        new_norm.running_mean.data = torch.index_select(norm.running_mean.data, 0, index)
        new_norm.running_var.data = torch.index_select(norm.running_var.data, 0, index)
    return new_norm


def replace_conv2d(conv, input_mask, output_mask):
    """
    Parameters
    ----------
    conv : torch.nn.Conv2d
        The conv2d module to be replaced
    mask : ModuleMasks
        The masks of this module

    Returns
    -------
    torch.nn.Conv2d
        The new conv2d module
    """
    if input_mask is None:
        in_channels = conv.in_channels
    else:
        in_channels_index = get_index(input_mask)
        in_channels = len(in_channels_index)
    if output_mask is None:
        out_channels = conv.out_channels
    else:
        out_channels_index = get_index(output_mask)
        out_channels = len(out_channels_index)

    if conv.groups != 1:
        new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride,
                                   padding=conv.padding,
                                   dilation=conv.dilation,
                                   groups=out_channels,
                                   bias=conv.bias is not None,
                                   padding_mode=conv.padding_mode)
    else:
        new_conv = torch.nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride,
                                   padding=conv.padding,
                                   dilation=conv.dilation,
                                   groups=conv.groups,
                                   bias=conv.bias is not None,
                                   padding_mode=conv.padding_mode)

    new_conv.to(conv.weight.device)
    tmp_weight_data = tmp_bias_data = None

    if output_mask is not None:
        tmp_weight_data = torch.index_select(conv.weight.data, 0, out_channels_index)
        if conv.bias is not None:
            tmp_bias_data = torch.index_select(conv.bias.data, 0, out_channels_index)
    else:
        tmp_weight_data = conv.weight.data
    # For the convolutional layers that have more than one group
    # we need to copy the weight group by group, because the input
    # channal is also divided into serveral groups and each group
    # filter may have different input channel indexes.
    input_step = int(conv.in_channels / conv.groups)
    in_channels_group = int(in_channels / conv.groups)
    filter_step = int(out_channels / conv.groups)
    if input_mask is not None:
        if new_conv.groups == out_channels:
            new_conv.weight.data.copy_(tmp_weight_data)
        else:
            for groupid in range(conv.groups):
                start = groupid * input_step
                end = (groupid + 1) * input_step
                current_input_index = list(filter(lambda x: start <= x and x < end, in_channels_index.tolist()))
                # shift the global index into the group index
                current_input_index = [x - start for x in current_input_index]
                # if the groups is larger than 1, the input channels of each
                # group should be pruned evenly.
                assert len(current_input_index) == in_channels_group, \
                    'Input channels of each group are not pruned evenly'
                current_input_index = torch.tensor(current_input_index).to(tmp_weight_data.device)  # pylint: disable=not-callable
                f_start = groupid * filter_step
                f_end = (groupid + 1) * filter_step
                new_conv.weight.data[f_start:f_end] = torch.index_select(tmp_weight_data[f_start:f_end], 1, current_input_index)
    else:
        new_conv.weight.data.copy_(tmp_weight_data)

    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data if tmp_bias_data is None else tmp_bias_data)

    return new_conv


def train(epoch, model, train_loader, criterion, optimizer, args):
    model.train()
    losses = AverageMeter()
    gts_all, predictions_all = [], []
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.cuda()
        images = images.cuda()
        images = Variable(images)
        labels = Variable(labels)
        # Decaying Learning Rate
        lr = poly_lr_scheduler(optimizer, args.init_lr, args.iteration, args.max_iter, args.power)
        # Forward + Backward + Optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        #  record loss
        losses.update(loss.data)
        if random.random() < args.tra_sample_rate or i == 0:
            predictions = outputs.data.max(1)[1].cpu().numpy()
            gts_all.append(labels.data.cpu().numpy())
            predictions_all.append(predictions)
        args.lr = lr
        logging.info('[epoch %d],[iter %04d/%04d]:lr = %.9f,train_losses.avg = %.9f'
                     % (epoch, args.iteration % len(train_loader) + 1, len(train_loader), args.lr, losses.avg))
        args.iteration = args.iteration + 1
        # if args.iteration == 10:
        #     break
    tra_acc, tra_acc_cls, tra_miou, tra_fwavacc = get_evaluation_score(predictions_all, gts_all, args.num_classes)
    return losses.avg, tra_acc, tra_acc_cls, tra_miou, tra_fwavacc


def get_data_loader(args):
    print("\nloading dataset ...")
    train_data = Reader(args, mode='train')
    print("Train set samples: ", len(train_data))
    val_data = Reader(args, mode='test')
    print("Validation set samples: ", len(val_data))
    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              drop_last=True, num_workers=args.workers if torch.cuda.is_available() else 0)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            drop_last=False, num_workers=args.workers if torch.cuda.is_available() else 0)

    return train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
    parser.add_argument('--data', default='./dataset/mars_seg', type=str, help='dataset path')
    parser.add_argument('--model', default='seg_deeplab_efficiennetb3', type=str, help='model name')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--pruner', default='agp', type=str, help='pruner: agp|taylor|fpgm')
    parser.add_argument('--sparsity', default=0.5, type=float, metavar='LR', help='prune sparsity')
    parser.add_argument('--finetune_epochs', default=10, type=int, metavar='N',
                        help='number of epochs for exported model')
    parser.add_argument('--finetune_lr', default=0.001, type=float, metavar='N', help='number of lr for exported model')
    parser.add_argument('--finetune_momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--finetune_weight_decay', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 5e-4)')

    parser.add_argument('--batch_size', default=18, type=int, metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', default=0.0, type=float, help='learning rate')
    parser.add_argument('--init_lr', default=2e-5, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--power', default=0.9, type=float, help='weight decay (default: 5e-4)')
    parser.add_argument('--tra_sample_rate', default=0.001, type=float, )
    parser.add_argument('--iteration', default=0, type=int, help='weight decay (default: 5e-4)')
    parser.add_argument('--max_iter', default=10e10, type=int, help='weight decay (default: 5e-4)')
    parser.add_argument('--epoch_num', default=100, type=int, help='weight decay (default: 5e-4)')
    parser.add_argument('--num_classes', default=6, type=int, help='weight decay (default: 5e-4)')

    parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 50)')
    parser.add_argument('--resume', default='./checkpoint/resnet50.pth', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_dir', default='./exp_log/', help='The directory used to save the trained models',
                        type=str)
    parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int,
                        default=1)
    parser.add_argument('--seed', type=int, default=2, help='random seed')

    parser.add_argument('--baseline', action='store_true', default=False, help='evaluate model on validation set')
    parser.add_argument('--prune_eval_path', default='', type=str, metavar='PATH', help='path to eval pruned model')

    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    best_prec1 = 0
    args.inputs_shape = (1, 3, 512, 512)
    args.save_dir = os.path.join(args.save_dir, args.model + '_' + args.pruner + '_' + str(args.sparsity))
    # create_exp_dir(args.save_dir, scripts_to_save=glob.glob('*.py'))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(args)

    # in_replace_dic = {
    #     'backbone_net.sa1.mlp_module.layer0.conv': (None, ),
    #     'backbone_net.sa1.mlp_module.layer0.bn.bn': ('backbone_net.sa1.mlp_module.layer0.conv', ),
    #     'backbone_net.sa1.mlp_module.layer1.conv': ('backbone_net.sa1.mlp_module.layer0.conv', ),
    #     'backbone_net.sa1.mlp_module.layer1.bn.bn': ('backbone_net.sa1.mlp_module.layer1.conv', ),
    #     'backbone_net.sa1.mlp_module.layer2.conv': ('backbone_net.sa1.mlp_module.layer1.conv', ),
    #     'backbone_net.sa1.mlp_module.layer2.bn.bn': ('backbone_net.sa1.mlp_module.layer2.conv', ),
    #     'backbone_net.sa2.mlp_module.layer0.conv': (
    #         3,
    #         'backbone_net.sa1.mlp_module.layer2.conv',
    #     ),
    #     'backbone_net.sa2.mlp_module.layer0.bn.bn': ('backbone_net.sa2.mlp_module.layer0.conv', ),
    #     'backbone_net.sa2.mlp_module.layer1.conv': ('backbone_net.sa2.mlp_module.layer0.conv', ),
    #     'backbone_net.sa2.mlp_module.layer1.bn.bn': ('backbone_net.sa2.mlp_module.layer1.conv', ),
    #     'backbone_net.sa2.mlp_module.layer2.conv': ('backbone_net.sa2.mlp_module.layer1.conv', ),
    #     'backbone_net.sa2.mlp_module.layer2.bn.bn': ('backbone_net.sa2.mlp_module.layer2.conv', ),
    #     'backbone_net.sa3.mlp_module.layer0.conv': (
    #         3,
    #         'backbone_net.sa2.mlp_module.layer2.conv',
    #     ),
    #     'backbone_net.sa3.mlp_module.layer0.bn.bn': ('backbone_net.sa3.mlp_module.layer0.conv', ),
    #     'backbone_net.sa3.mlp_module.layer1.conv': ('backbone_net.sa3.mlp_module.layer0.conv', ),
    #     'backbone_net.sa3.mlp_module.layer1.bn.bn': ('backbone_net.sa3.mlp_module.layer1.conv', ),
    #     'backbone_net.sa3.mlp_module.layer2.conv': ('backbone_net.sa3.mlp_module.layer1.conv', ),
    #     'backbone_net.sa3.mlp_module.layer2.bn.bn': ('backbone_net.sa3.mlp_module.layer2.conv', ),
    #     'backbone_net.sa4.mlp_module.layer0.conv': (
    #         3,
    #         'backbone_net.sa3.mlp_module.layer2.conv',
    #     ),
    #     'backbone_net.sa4.mlp_module.layer0.bn.bn': ('backbone_net.sa4.mlp_module.layer0.conv', ),
    #     'backbone_net.sa4.mlp_module.layer1.conv': ('backbone_net.sa4.mlp_module.layer0.conv', ),
    #     'backbone_net.sa4.mlp_module.layer1.bn.bn': ('backbone_net.sa4.mlp_module.layer1.conv', ),
    #     'backbone_net.sa4.mlp_module.layer2.conv': ('backbone_net.sa4.mlp_module.layer1.conv', ),
    #     'backbone_net.sa4.mlp_module.layer2.bn.bn': ('backbone_net.sa4.mlp_module.layer2.conv', ),
    #     'backbone_net.fp1.mlp.layer0.conv': ('backbone_net.sa3.mlp_module.layer2.conv', 'backbone_net.sa4.mlp_module.layer2.conv'),
    #     'backbone_net.fp1.mlp.layer0.bn.bn': ('backbone_net.fp1.mlp.layer0.conv', ),
    #     'backbone_net.fp1.mlp.layer1.conv': ('backbone_net.fp1.mlp.layer0.conv', ),
    #     'backbone_net.fp1.mlp.layer1.bn.bn': ('backbone_net.fp1.mlp.layer1.conv', ),
    #     'backbone_net.fp2.mlp.layer0.conv': ('backbone_net.sa2.mlp_module.layer2.conv', 'backbone_net.fp1.mlp.layer1.conv'),
    #     'backbone_net.fp2.mlp.layer0.bn.bn': ('backbone_net.fp2.mlp.layer0.conv', ),
    #     'backbone_net.fp2.mlp.layer1.conv': ('backbone_net.fp2.mlp.layer0.conv', ),
    #     'pnet.vote_aggregation.mlp_module.layer0.conv': (None, ),
    #     'pnet.vote_aggregation.mlp_module.layer0.bn.bn': ('pnet.vote_aggregation.mlp_module.layer0.conv', ),
    #     'pnet.vote_aggregation.mlp_module.layer1.conv': ('pnet.vote_aggregation.mlp_module.layer0.conv', ),
    #     'pnet.vote_aggregation.mlp_module.layer1.bn.bn': ('pnet.vote_aggregation.mlp_module.layer1.conv', ),
    #     'pnet.vote_aggregation.mlp_module.layer2.conv': ('pnet.vote_aggregation.mlp_module.layer1.conv', ),
    #     'operation.mlp_module.layer0.conv': (None, ),
    #     'operation.mlp_module.layer0.bn.bn': ('operation.mlp_module.layer0.conv', ),
    #     'operation.mlp_module.layer1.conv': ('operation.mlp_module.layer0.conv', ),
    #     'operation.mlp_module.layer1.bn.bn': ('operation.mlp_module.layer1.conv', ),
    #     'operation.mlp_module.layer2.conv': ('operation.mlp_module.layer1.conv', ),
    # }

    masks_file = '/Users/lushun/Documents/code/model_compress/model_compress_v1/prune_seg/models/EfficientNet/mask_1.pth'

    # model define
    from models.EfficientNet.deeplab_efficiennetb1 import DeepLabv3_plus as deeplab_efficiennetb3
    from models.EfficientNet.deeplab_efficiennetb1 import compute_Params_FLOPs

    model = deeplab_efficiennetb3(nInputChannels=3, n_classes=6, os=16, pretrained=False,
                                  _print=False)
    save_point = torch.load(
        '/Users/shunlu/Documents/code/model_compress/model_compress_v1/prune_seg/exp_log/seg_deeplab_efficiennetb3_agp_0.5_3/model_masked_3.pth',
        map_location=args.device
    )
    model.load_state_dict(save_point)
    print('************** Before Pruning **************')
    compute_Params_FLOPs(copy.deepcopy(model), args.device)

    train_loader, val_loader = get_data_loader(args)
    # # load pretrained weights
    # save_point = torch.load('./pretrained/model.pth', map_location=args.device)
    # model_param = save_point['state_dict']
    # model.load_state_dict(model_param)
    # model.eval()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr,
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss().to(args.device)
    def trainer(model, optimizer, criterion, epoch):
        result = train(epoch, model, train_loader, criterion, optimizer, args)
        return result

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
                             'last_conv.1', 'last_conv.2', 'last_conv.3', 'last_conv.4', 'last_conv.5', 'last_conv.6'
                             ]
                        }]
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
    mask_pt = torch.load(
        '/Users/shunlu/Documents/code/model_compress/model_compress_v1/prune_seg/exp_log/seg_deeplab_efficiennetb3_agp_0.5_3/mask_3.pth',
        map_location=args.device
    )

    model = get_pruned_model(pruner, model, mask_pt)

    model.eval()
    image = torch.randn(size=args.inputs_shape)
    image = image.to(args.device)
    model.to(args.device)
    with torch.no_grad():
        output = model.forward(image)
        print('Output size:', output.size())
    print('************** After Pruning **************')
    compute_Params_FLOPs(copy.deepcopy(model), args.device)
