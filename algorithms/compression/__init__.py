import sys
sys.path.append('algorithms/compression')
sys.path.append('algorithms/compression/nets/ResNet50_SSD')
sys.path.append('algorithms/compression/nets/ResNet50_SSD/SSD_Pytorch')
sys.path.append('algorithms/compression/nets/UNet/Pytorch-UNet-master')

from .nets.ResNet50_SSD.prune_model import *