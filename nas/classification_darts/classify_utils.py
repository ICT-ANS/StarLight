import os
import glob
from pdf2image import convert_from_path

import torch
from torch.autograd import Variable


def convertPath(path: str) -> str:
    """
    This function solves the problem of path separator in windows.

    Parameters
    ----------
    path : str
        Path before processing

    Returns
    -------
    str
        Path after solving the problem of path separator in Windows
    """
    return path.replace(os.sep, "/")


def drop_path(x, drop_prob):
    """
    This function randomly drops information from the input x with the probability drop_prob.

    Parameters
    ----------
    x : tensor
        input to be dropped
    drop_prob : float
        the probability to drop information

    Returns
    -------
    tensor
        x after dropping information
    """
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def from_darts_read_log(method, log_name, key_words):
    """
    This function reads information from the log.

    Parameters
    ----------
    method : str
        corresponding NAS method for the log
    log_name : str
        the name of each log
    key_words : list
        the expected data name

    Returns
    -------
    dict
        this method data_info extracted from log
    """
    info = dict()
    for key in key_words:
        info[key] = []
    # info['epoch'].append('0')
    with open(log_name) as f:
        if method == 'DARTS':
            for line in f:
                if 'train_acc' in line:
                    line = line.strip().split()
                    info['train_acc'].append(float(line[4]))
                    info['train_loss'].append(float(line[6]))
                elif 'test_acc' in line or 'valid_acc' in line:
                    line = line.strip().split()
                    info['valid_acc'].append(float(line[4]))
                    info['valid_loss'].append(float(line[6]))
                elif 'genotype' in line:
                    line = line.strip().split('genotype = ')
                    info['genotype'].append(line[1])
                elif 'epoch' in line and 'lr' in line:
                    line = line.strip().split(' ')
                    info['epoch'].append(line[4])
                    info['lr'].append(float(line[-1]))
        elif method == 'GDAS':
            for line in f:
                if 'train_acc' in line:
                    line = line.strip().split()
                    info['train_acc'].append(float(line[3]))
                elif 'train_loss' in line:
                    line = line.strip().split()
                    info['train_loss'].append(float(line[3]))
                elif 'valid_acc' in line:
                    line = line.strip().split()
                    info['valid_acc'].append(float(line[3]))
                elif 'valid_loss' in line:
                    line = line.strip().split()
                    info['valid_loss'].append(float(line[3]))
                elif 'genotype' in line:
                    line = line.strip().split('genotype = ')
                    info['genotype'].append(line[1])
                elif 'tau' in line and 'LR' in line:
                    line = line.strip().split(' ')
                    info['epoch'].append(int(line[2].split('-')[0]))
                    info['lr'].append(float(line[-1].split('=')[-1]))
        else:
            raise NotImplementedError

    return info


def convert_pdf_to_img(inputDir, outputDir):
    """
    This function converts pdf to image.

    Parameters
    ----------
    inputDir : str
        path of the input pdf file
    outputDir : str
        path of the output images

    Returns
    -------
    str
        convert pdf to image
    """
    img_path_list = glob.glob('{}/*.pdf'.format(inputDir))
    img_name_list = []
    for _img_path in img_path_list:
        _img_name = _img_path.split('/')[-1].split('.')[-2]
        img_name_list.append(_img_name)

    for i, img_path in enumerate(img_path_list):
        name = img_name_list[i].split('.')[0]
        assert name in img_path
        img = convert_from_path(img_path)
        img[0].save('%s/%s.png' % (outputDir, name))
        os.remove(img_path)


def read_offline_log_to_string(log_path):
    """
    This function extracts information from the log.

    Parameters
    ----------
    log_path : str
        location of log path

    Returns
    -------
    list
        log information content

    """
    log_content = []
    with open(log_path) as f:
        for line in f:
            log_content.append(line)
    return log_content


def write_online_log_to_file(log_path, log_info, log_pointer, write_length, recreate=False):
    """
    This function saves the online information to a designated file.

    Parameters
    ----------
    log_path : str
        path location of the log
    log_info : list
        contents in the log
    log_pointer : int
        start point to write the log
    write_length : int
        length of content
    recreate : int
        recreate the log or False

    Returns
    -------
    str
        file name
    """
    if os.path.exists(log_path) and recreate:
        os.remove(log_path)
    with open(log_path, 'a+') as f:
        for i in range(write_length*(log_pointer-1), write_length*log_pointer):
            try:
                f.write(log_info[i])
            except:
                break
