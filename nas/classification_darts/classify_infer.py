import os
import numpy as np
from config import C

import matplotlib.pyplot as plt
from thop import profile
import warnings

warnings.filterwarnings('ignore')
import torch
import torch.utils
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from nas.classification_darts import classify_genotypes
from nas.classification_darts.classify_model import NetworkCIFAR as Network


def infer(model_name):
    """
    Infer the searched model on the test set using the designated model.

    Parameters
    ----------
    model_name : str
        searched model name

    Returns
    -------
    None
    """
    genotype = eval("classify_genotypes.%s_cifar10_2" % model_name)
    model = Network(C=36, num_classes=10, layers=20, auxiliary=True, genotype=genotype)
    model.load_state_dict(torch.load(
        os.path.join(C.cache_dir, './checkpoint/%s_cifar10_2_retrain_best.pt' % model_name),
        map_location='cpu')
    )
    model.drop_path_prob = 0.2
    model.eval()
    inputs = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, (inputs, ), verbose=False)
    print('Model flops: %.3f M, params: %.3f M' % (flops/1e6, params/1e6))

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_data = dset.CIFAR10(root=os.path.join(C.cache_dir, './data/cifar10'),
                             train=False, download=False, transform=test_transform)
    test_queue = DataLoader(test_data, batch_size=1, pin_memory=False, num_workers=0, shuffle=False)

    visual_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    visual_data = dset.CIFAR10(root=os.path.join(C.cache_dir, './data/cifar10'),
                               train=False, download=False, transform=visual_transform)
    visual_queue = DataLoader(visual_data, batch_size=1, pin_memory=True, num_workers=0, shuffle=False)

    test_iter = iter(test_queue)
    visual_iter = iter(visual_queue)
    for i in range(3):
        input, target = next(test_iter)
        # print(input[0][0][0][:])
        pred = model(input)
        print("pred:%s, label:%s" % (torch.argmax(pred[0]).item(), target.item()))

        input, target = next(visual_iter)
        img = input.detach().numpy().squeeze()
        img = np.transpose(img, (1, 2, 0))
        # print(img.shape)
        # print(img)
        plt.figure()
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    infer('darts')
