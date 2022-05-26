import torch
from torch.autograd import Function
import numpy as np

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def MPA(pred,label):
    A = label.view(-1)
    B = pred.view(-1)
    TP = torch.dot(A,B) #pred =1&label=1
    FP,FN,TN=0,0,0
    if pred.is_cuda:
        FP = torch.dot(torch.ones(A.size()).cuda()-A,B) # pred=1 &label=0
        FN = torch.dot(torch.ones(B.size()).cuda()-B,A)
        TN = torch.dot(torch.ones(A.size()).cuda()-A,torch.ones(B.size()).cuda()-B)
    else:
        FP = torch.dot(torch.ones(A.size()) - A, B)  # pred=1 &label=0
        FN = torch.dot(torch.ones(B.size()) - B, A)
        TN = torch.dot(torch.ones(A.size())- A, torch.ones(B.size())- B)
    return 0.5*(TP/(TP+FP+0.0001) + TN/(FN+TN+0.0001))

def DiceAndMPA(pred,label):
    A = label.view(-1).float()
    B = pred.view(-1).float()
    TP = torch.dot(A,B) #pred =1&label=1
    FP,FN,TN=0,0,0
    if pred.is_cuda:
        FP = torch.dot(torch.ones(A.size()).cuda()-A,B) # pred=1 &label=0
        FN = torch.dot(torch.ones(B.size()).cuda()-B,A)
        TN = torch.dot(torch.ones(A.size()).cuda()-A,torch.ones(B.size()).cuda()-B)
    else:
        FP = torch.dot(torch.ones(A.size()) - A, B)  # pred=1 &label=0
        FN = torch.dot(torch.ones(B.size()) - B, A)
        TN = torch.dot(torch.ones(A.size())- A, torch.ones(B.size())- B)
    return 2*TP/(2*TP+FN+FP),0.5*(TP/(TP+FP+0.0001) + TN/(FN+TN+0.0001))