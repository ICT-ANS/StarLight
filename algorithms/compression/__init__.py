import sys
sys.path.append('algorithms/compression')
sys.path.append('algorithms/compression/nets/ResNet50_SSD')
sys.path.append('algorithms/compression/nets/ResNet50_SSD/SSD_Pytorch')

from .nets.ResNet50_SSD.prune_model import *