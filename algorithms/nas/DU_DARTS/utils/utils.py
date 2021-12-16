import os
import time
import shutil
import numpy as np

import torch
from torch.autograd import Variable
import torchvision.transforms as transforms


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


def accuracy(output, target, topk=(1,)):
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

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def data_transforms_cifar10(args):
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


def data_transforms_cifar100(args):
    CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR_STD = [0.2673, 0.2564, 0.2762]

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


def save_checkpoint(state, is_best, save, exp, stage, epoch):
    filename = os.path.join(save, 'checkpoint_{}_{}_{}.pth.tar'.format(exp, stage, epoch))
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def print_dict(_dict, sort=False, info=None, accuracy=0):
    if info:
        print(info)
    if sort:
        _dict = dict(sorted(_dict.items(), key=lambda x: x[1], reverse=True))

    for _, (k, v) in enumerate(_dict.items()):
        if accuracy != 0:
            v = round(v, accuracy)
            print(k, v)
        else:
            print(k, v)
    print('')


def dict_normalize(_dict):
    total = 0
    for i in _dict.values():
        total += i
    for j in _dict.keys():
        _dict[j] /= total
    return _dict


def time_record(start):
    import logging
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    logging.info('Elapsed Time: %dh %dmin %ds' % (hour, minute, second))


def gpu_monitor(gpu, sec, used=100):
    import time
    import pynvml
    import logging

    wait_min = sec // 60
    divisor = 1024 * 1024
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if meminfo.used / divisor < used:
        logging.info('GPU-{} is free, start runing!'.format(gpu))
        return False
    else:
        logging.info('GPU-{}, Memory: total={}MB used={}MB free={}MB, waiting {}min...'.format(
            gpu,
            meminfo.total / divisor,
            meminfo.used / divisor,
            meminfo.free / divisor,
            wait_min)
        )
        time.sleep(sec)
        return True


def get_one_hot_tensor(alpha, num_edges, num_ops):
    alpha_argmax = torch.argmax(alpha, dim=-1).cpu().numpy().tolist()
    alpha_one_hot = torch.zeros((num_edges, num_ops))
    for i, j in enumerate(alpha_argmax):
        alpha_one_hot[i][j] = 1
    return alpha_one_hot


def cal_maximum_entropy(num_edges, num_ops):
    p = (1 / num_ops) * torch.ones(num_edges, num_ops).cpu()
    max_info_entrypy = torch.sum(-p * torch.log2(p))
    return max_info_entrypy.item()


def cal_aux_loss(normal_alpha, reduce_alpha, normal_beta, reduce_beta):
    from darts_info_entropy.search_config import args
    if args.del_none:
        num_edges = 14
        num_ops = 7
    else:
        num_edges = 14
        num_ops = 8

    if args.beta_loss and args.loss_type == 'entropy':
        max_info_entrypy = cal_maximum_entropy(num_edges, num_ops)
        normal_info = torch.sum(-normal_alpha * torch.log2(normal_alpha), dim=-1) / max_info_entrypy
        reduce_info = torch.sum(-reduce_alpha * torch.log2(reduce_alpha), dim=-1) / max_info_entrypy
        normal_info = normal_info / normal_beta
        reduce_info = reduce_info / reduce_beta
        aux_loss = torch.sum(normal_info) + torch.sum(reduce_info)

    elif args.beta_loss and args.loss_type == 'info_amount':
        normal_info_amount = torch.sum(-torch.log2(torch.max(normal_alpha, dim=-1)[0])/normal_beta)/num_edges
        reduce_info_amount = torch.sum(-torch.log2(torch.max(reduce_alpha, dim=-1)[0])/reduce_beta)/num_edges
        aux_loss = normal_info_amount + reduce_info_amount

    elif args.beta_loss and args.loss_type == 'mae':
        normal_alpha_one_hot = get_one_hot_tensor(normal_alpha, num_edges, num_ops)
        reduce_alpha_one_hot = get_one_hot_tensor(reduce_alpha, num_edges, num_ops)
        normal_alpha_error = torch.sum(torch.abs(torch.sub(normal_alpha, normal_alpha_one_hot.cuda())), dim=-1) \
                             / (normal_beta * num_edges)
        reduce_alpha_error = torch.sum(torch.abs(torch.sub(reduce_alpha, reduce_alpha_one_hot.cuda())), dim=-1) \
                             / (reduce_beta * num_edges)
        aux_loss = torch.sum(normal_alpha_error) + torch.sum(reduce_alpha_error)

    elif args.beta_loss and args.loss_type == 'mse':
        normal_alpha_one_hot = get_one_hot_tensor(normal_alpha, num_edges, num_ops)
        reduce_alpha_one_hot = get_one_hot_tensor(reduce_alpha, num_edges, num_ops)
        normal_alpha_error = torch.sum(torch.sub(normal_alpha, normal_alpha_one_hot.cuda()).pow(2), dim=-1)\
                             / (normal_beta * num_edges)
        reduce_alpha_error = torch.sum(torch.sub(reduce_alpha, reduce_alpha_one_hot.cuda()).pow(2), dim=-1)\
                             / (reduce_beta * num_edges)
        aux_loss = torch.sum(normal_alpha_error) + torch.sum(reduce_alpha_error)

    elif args.beta_loss and args.loss_type == 'rmse':
        normal_alpha_one_hot = get_one_hot_tensor(normal_alpha, num_edges, num_ops)
        reduce_alpha_one_hot = get_one_hot_tensor(reduce_alpha, num_edges, num_ops)
        normal_alpha_error = torch.sum(
            torch.sqrt(torch.sub(normal_alpha, normal_alpha_one_hot.cuda()).pow(2)/num_edges), dim=-1) / normal_beta
        reduce_alpha_error = torch.sum(
            torch.sqrt(torch.sub(reduce_alpha, reduce_alpha_one_hot.cuda()).pow(2)/num_edges), dim=-1) / reduce_beta
        aux_loss = torch.sum(normal_alpha_error) + torch.sum(reduce_alpha_error)

    elif not args.beta_loss and args.loss_type == 'entropy':
        max_info_entrypy = cal_maximum_entropy(num_edges, num_ops)
        normal_info = torch.sum(-normal_alpha * torch.log2(normal_alpha))
        reduce_info = torch.sum(-reduce_alpha * torch.log2(reduce_alpha))
        aux_loss = (normal_info + reduce_info) / (2 * max_info_entrypy)

    else:
        raise ValueError('No Defined Loss Type!!!')

    return aux_loss


def run_func(args, main):
    if torch.cuda.is_available():
        while gpu_monitor(args.gpu_id, sec=900, used=500):
            pass
    start_time = time.time()
    result = main()
    time_record(start_time)


def set_seed(seed):
    """
        fix all seeds
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

