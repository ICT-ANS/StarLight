# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from utils_.box_utils import match, log_sum_exp, refine_match
from layers.modules import WeightSoftmaxLoss, WeightSmoothL1Loss
GPU = False
if torch.cuda.is_available():
    GPU = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


class RefineMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, cfg, num_classes):
        super(RefineMultiBoxLoss, self).__init__()
        self.cfg = cfg
        self.size = cfg.MODEL.SIZE
        if self.size == '300':
            size_cfg = cfg.SMALL
        else:
            size_cfg = cfg.BIG
        self.variance = size_cfg.VARIANCE
        self.num_classes = num_classes
        self.threshold = cfg.TRAIN.OVERLAP
        self.OHEM = cfg.TRAIN.OHEM
        self.negpos_ratio = cfg.TRAIN.NEG_RATIO
        self.object_score = cfg.MODEL.OBJECT_SCORE
        self.variance = size_cfg.VARIANCE
        if cfg.TRAIN.FOCAL_LOSS:
            if cfg.TRAIN.FOCAL_LOSS_TYPE == 'SOFTMAX':
                self.focaloss = FocalLossSoftmax(
                    self.num_classes, gamma=2, size_average=False)
            else:
                self.focaloss = FocalLossSigmoid()

    def forward(self,
                predictions,
                targets,
                use_arm=False,
                filter_object=False,
                debug=False):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # arm_loc_data, arm_conf_data, loc_data, conf_data, priors = predictions
        if use_arm:
            arm_loc_data, arm_conf_data, loc_data, conf_data, priors = predictions
        else:
            loc_data, conf_data, _, _, priors = predictions
        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        defaults = priors.data
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            if self.num_classes == 2:
                labels = labels > 0
            if use_arm:
                bbox_weight = refine_match(
                    self.threshold,
                    truths,
                    defaults,
                    self.variance,
                    labels,
                    loc_t,
                    conf_t,
                    idx,
                    arm_loc_data[idx].data,
                    use_weight=False)
            else:
                match(self.threshold, truths, defaults, self.variance, labels,
                      loc_t, conf_t, idx)

        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        if use_arm and filter_object:
            P = F.softmax(arm_conf_data, 2)
            arm_conf_data_temp = P[:, :, 1]
            object_score_index = arm_conf_data_temp <= self.object_score
            pos = conf_t > 0
            pos[object_score_index.detach()] = 0
        else:
            pos = conf_t > 0
        num_pos = pos.sum(1, keepdim=True)
        if debug:
            if use_arm:
                print("odm pos num: ", str(loc_t.size(0)), str(loc_t.size(1)))
            else:
                print("arm pos num", str(loc_t.size(0)), str(loc_t.size(1)))

        if self.OHEM:
            # Compute max conf across batch for hard negative mining
            batch_conf = conf_data.view(-1, self.num_classes)

            loss_c = log_sum_exp(batch_conf) - batch_conf.gather(
                1, conf_t.view(-1, 1))

            # Hard Negative Mining
            loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
            loss_c = loss_c.view(num, -1)
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)

            if num_pos.data.sum() > 0:
                num_neg = torch.clamp(
                self.negpos_ratio * num_pos, max=pos.size(1) - 1)
            else:
                fake_num_pos = torch.ones(32, 1).long() * 15
                num_neg = torch.clamp(
                self.negpos_ratio * fake_num_pos, max=pos.size(1) - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos.unsqueeze(2).expand_as(conf_data)
            neg_idx = neg.unsqueeze(2).expand_as(conf_data)
            conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(
                -1, self.num_classes)

            targets_weighted = conf_t[(pos + neg).gt(0)]
            loss_c = F.cross_entropy(
                conf_p, targets_weighted, size_average=False)
        else:
            loss_c = F.cross_entropy(conf_p, conf_t, size_average=False)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        if num_pos.data.sum() > 0:
            pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
            loc_p = loc_data[pos_idx].view(-1, 4)
            loc_t = loc_t[pos_idx].view(-1, 4)
            loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
            N = num_pos.data.sum()
        else:
            loss_l = torch.zeros(1)
            N = 1.0

        loss_l /= float(N)
        loss_c /= float(N)
        return loss_l, loss_c
