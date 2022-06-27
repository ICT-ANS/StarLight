import torch
from nas.classification_darts.nas_classify_darts_quiver.searched_models.segmentation.auto_deeplab.build_autodeeplab import Retrain_Autodeeplab
from nas.classification_darts.nas_classify_darts_quiver.config.model_config import auto_deeplab_retrain_args


def model_builder(model_name):
    if model_name == 'resnet18':
        input_size = [250, 250]
        from torchvision import models
        model = models.resnet18()
    elif model_name in ['darts', 'pdarts', 'pcdarts',
                        'sgas_cri1', 'sgas_cri2', 'sdarts_rs', 'sdarts_pgd'
                        'rdarts_l2', 'rdarts_cutout']:
        input_size = [32, 32]
        from nas.classification_darts.nas_classify_darts_quiver.searched_models.classification.darts import genotypes
        from nas.classification_darts.nas_classify_darts_quiver.searched_models.classification.darts.model import NetworkCIFAR
        model = NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=False,
                             genotype=eval('genotypes.{}_cifar10_2'.format(model_name)))
    elif model_name in ['DARTS_V1', 'DARTS_cifar10_2']:
        input_size = [32, 32]
        from nas.classification_darts.nas_classify_darts_quiver.searched_models.classification.darts import genotypes
        from nas.classification_darts.nas_classify_darts_quiver.searched_models.classification.darts.model import NetworkCIFAR
        model = NetworkCIFAR(C=36, num_classes=10, layers=20, auxiliary=False,
                             genotype=eval('genotypes.{}'.format(model_name)))
    elif model_name == 'auto_deeplab':
        input_size = [513, 513]
        args = auto_deeplab_retrain_args()
        args.num_classes = 19
        model = Retrain_Autodeeplab(args)
        model.eval()
    else:
        raise ValueError('No Defined Model!')

    return model, input_size


if __name__ == '__main__':
    model = model_builder(model_name='resnet18')
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
