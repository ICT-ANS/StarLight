import torch
import torch.nn as nn

OPS = {
  "skip_connect" : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  "conv_3x1_1x3" : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,3), stride=(1, stride), padding=(0, 1), bias=False),
    nn.Conv2d(C, C, (3,1), stride=(stride, 1), padding=(1, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  "conv_7x1_1x7" : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  "dil_conv_3x3" : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  "dil_conv_5x5" : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  "avg_pool_3x3" : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  "max_pool_3x3" : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  "max_pool_5x5" : lambda C, stride, affine: nn.MaxPool2d(5, stride=stride, padding=2),
  "max_pool_7x7" : lambda C, stride, affine: nn.MaxPool2d(7, stride=stride, padding=3),
  "conv 1x1" : lambda C, stride, affine: ReLUConvBN(C, C, kernel_size=1, stride=stride, padding=0, affine=affine),
  "conv 3x3" : lambda C, stride, affine: ReLUConvBN(C, C, kernel_size=3, stride=stride, padding=1, affine=affine),
  "sep_3x3" : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  "sep_5x5" : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  "sep_7x7" : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
}



class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)

class SepConv2(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv2, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
    )

class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

def numParameters(module):
  num = 0
  for x in module.parameters():
    num+=torch.numel(x)
  return num

if __name__=='__main__':
  for key,value in OPS.items():
    print(key,)
    print(numParameters(value(12,1,False)))


