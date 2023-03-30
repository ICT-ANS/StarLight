import os.path

import torch

from nas_vis.nas_models.model import NetworkCIFAR
from nas_vis.nas_models import genotypes

if __name__ == '__main__':
    method = 'GDAS'
    model_dir = '/Users/lushun/Documents/code/2022/star-light-repo/data/StarLight_Cache/nas.classification/%s/checkpoint' % method
    ckpt_path = os.path.join(model_dir, '%s_cifar10_2_retrain_best.pt' % method)
    model_path = os.path.join(model_dir, 'model.pth')
    model = NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=True,
                         genotype=eval('genotypes.%s' % method))
    pretrained_ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(pretrained_ckpt)
    print('Successfully load the pre-trained weights from: %s' % ckpt_path)
    torch.save(model, model_path)
