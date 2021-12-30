from __future__ import division

import logging
import os
import random
import shutil

import hdf5storage
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    """Print the results in the log file."""
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s',"%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger


def save_checkpoint(model_path, epoch, iteration, miou, model, optimizer):
    """Save the checkpoint."""
    state = {
        # 'epoch': epoch,
        # 'iter': iteration,
        # 'miou': miou,
        'state_dict': model.state_dict(),
        # 'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'model.pth'))


def save_checkpoint_model(model_path, epoch, iteration, miou, model, optimizer):
    """Save the checkpoint."""
    state = {
        'epoch': epoch,
        'iter': iteration,
        'miou': miou,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

    state = {
        # 'epoch': epoch,
        # 'iter': iteration,
        # 'miou': miou,
        'state_dict': model.state_dict(),
        # 'optimizer': optimizer.state_dict(),
    }
    torch.save(state, os.path.join(model_path, 'net_%depoch_model.pth' % epoch))


def save_checkpoint_model_iteration(model_path, epoch, iteration, miou, model, optimizer):
    """Save the checkpoint."""
    # state = {
    #     'epoch': epoch,
    #     'iter': iteration,
    #     'miou': miou,
    #     'state_dict': model.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    # }
    #
    # torch.save(state, os.path.join(model_path, 'net_%depoch%diteration.pth' % (epoch, iteration)))

    state = {
        # 'epoch': epoch,
        # 'iter': iteration,
        # 'miou': miou,
        'state_dict': model.state_dict(),
        # 'optimizer': optimizer.state_dict(),
    }
    torch.save(state, os.path.join(model_path, 'model.pth'))


def save_matv73(mat_name, var_name, var):
    hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)


def record_loss(loss_csv, epoch, iteration, epoch_time, lr, tra_loss, tra_acc, tra_acc_cls,
                    tra_miou, tra_fwavacc, val_loss, val_acc, val_acc_cls,
                    val_miou, val_fwavacc):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, tra_loss, tra_acc, tra_acc_cls,
                    tra_miou, tra_fwavacc, val_loss, val_acc, val_acc_cls,
                    val_miou, val_fwavacc))
    loss_csv.flush()    
    loss_csv.close


def get_reconstruction(input, num_split, dimension, model):
    """As the limited GPU memory split the input."""
    input_split = torch.split(input,  int(input.shape[3]/num_split), dim=dimension)
    output_split = []
    for i in range(num_split):
        var_input = Variable(input_split[i].cuda(), volatile=True)
        var_output = model(var_input)
        output_split.append(var_output.data)
        if i == 0:
            output = output_split[i]
        else:
            output = torch.cat((output, output_split[i]), dim=dimension)
    
    return output


def reconstruction(rgb,model):
    """Output the final reconstructed hyperspectral images."""
    img_res = get_reconstruction(torch.from_numpy(rgb).float(), 1, 3, model)
    img_res = img_res.cpu().numpy()*4095
    img_res = np.transpose(np.squeeze(img_res))
    img_res_limits = np.minimum(img_res, 4095)
    img_res_limits = np.maximum(img_res_limits, 0)
    return img_res_limits


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def get_evaluation_score(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc


def _fast_hist_torch(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = torch.bincount(
        num_classes * label_true[mask] +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate_torch(predictions, gts, num_classes):
    hist = torch.zeros((num_classes, num_classes)).cuda()
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist_torch(lp.flatten(), lt.flatten(), num_classes).cuda().float()
    # axis 0: gt, axis 1: prediction
    # acc = torch.diag(hist).sum() / hist.sum()
    # acc_cls = torch.diag(hist) / hist.sum(dim=1)
    # acc_cls = torch.mean(acc_cls)
    iu = torch.diag(hist) / (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist))
    # mean_iu = torch.mean(iu)
    mean_iu_noclass0 = torch.mean(iu[1:])
    # freq = hist.sum(dim=1) / hist.sum()
    # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    hist_false = hist[1:, 1:]
    iu_false = torch.diag(hist_false) / (hist_false.sum(dim=1) + hist_false.sum(dim=0) - torch.diag(hist_false))
    mean_iu_noclass0_false = torch.mean(iu_false)
    return mean_iu_noclass0, mean_iu_noclass0_false, iu[1], iu[2], iu[3], iu[4]


def set_seed(seed):
    """
        fix all seeds as setting values
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


def time_record(start):
    import logging
    import time
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


def run_func(args, main):
    import time
    if torch.cuda.is_available():
        while gpu_monitor(args.gpu_id, sec=300, used=1000):
            pass
    start_time = time.time()
    result = main()
    time_record(start_time)
    # email_sender(result=result, config=args)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
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


def data_transforms_cifar10(cutout=False, cutout_length=16):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if cutout:
        train_transform.transforms.append(Cutout(cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform
