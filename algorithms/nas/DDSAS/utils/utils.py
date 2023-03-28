import os
import time
import json
import numpy as np
import torch
import shutil
from collections import namedtuple
import torchvision.transforms as transforms
from torch.autograd import Variable


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1, )):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms(args):
    if args.dataset == 'cifar10':
        DATA_MEAN = [0.49139968, 0.48215827, 0.44653124]
        DATA_STD = [0.24703233, 0.24348505, 0.26158768]
    elif args.dataset == 'cifar100':
        DATA_MEAN = [0.5071, 0.4867, 0.4408]
        DATA_STD = [0.2675, 0.2565, 0.2761]
    elif args.dataset == 'svhn':
        DATA_MEAN = [0.4377, 0.4438, 0.4728]
        DATA_STD = [0.1980, 0.2010, 0.1970]
    else:
        raise ValueError('No Defined Dataset!')
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, path):
    torch.save(state, path)


def save_checkpoint2(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def load_checkpoint(path):
    return torch.load(path, map_location='cpu')


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def convert_param(original_lists):
    support_types = ('str', 'int', 'bool', 'float', 'none')
    assert isinstance(original_lists, list), 'The type is not right : {:}'.format(original_lists)
    ctype, value = original_lists[0], original_lists[1]
    assert ctype in support_types, 'Ctype={:}, support={:}'.format(ctype, support_types)
    is_list = isinstance(value, list)
    if not is_list:
        value = [value]
    outs = []
    for x in value:
        if ctype == 'int':
            x = int(x)
        elif ctype == 'str':
            x = str(x)
        elif ctype == 'bool':
            x = bool(int(x))
        elif ctype == 'float':
            x = float(x)
        elif ctype == 'none':
            if x.lower() != 'none':
                raise ValueError('For the none type, the value must be none instead of {:}'.format(x))
            x = None
        else:
            raise TypeError('Does not know this type : {:}'.format(ctype))
        outs.append(x)
    if not is_list:
        outs = outs[0]
    return outs


def load_config(path, extra, logger):
    path = str(path)
    if hasattr(logger, 'log'): logger.log(path)
    assert os.path.exists(path), 'Can not find {:}'.format(path)
    # Reading data back
    with open(path, 'r') as f:
        data = json.load(f)
    content = {k: convert_param(v) for k, v in data.items()}
    assert extra is None or isinstance(extra, dict), 'invalid type of extra : {:}'.format(extra)
    if isinstance(extra, dict):
        content = {**content, **extra}
    Arguments = namedtuple('Configure', ' '.join(content.keys()))
    content = Arguments(**content)
    if hasattr(logger, 'log'):
        logger.log('{:}'.format(content))
    return content


def save_model(model, file_path):
    torch.save({'alphas_normal': model.alphas_normal, 'alphas_reduce': model.alphas_reduce, 'betas_normal': model.betas_normal, 'betas_reduce': model.betas_reduce, 'params': model.state_dict()}, file_path)


def write_to_file(str, file):
    with open(file, 'a') as f:
        f.write(str + '\n')
