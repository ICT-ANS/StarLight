import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='elementwise_mean', ignore_index=255):  # reduction='elementwise_mean' / 'sum'
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, size_average=None, ignore_index=ignore_index,
                 reduce=None, reduction=reduction)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class CriterionGALD(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=255,reduce=True):
        super(CriterionGALD, self).__init__()

        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, preds, target):
        scale_pred = preds[0]
        loss1 = super(CriterionGALD, self).forward(scale_pred, target)
        scale_pred = preds[1]
        loss2 = super(CriterionGALD, self).forward(scale_pred, target)

        return loss1 + loss2 * 0.4


