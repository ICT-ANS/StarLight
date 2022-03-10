import gc
import glob
import os
import time

import cv2
import logging
import numpy as np
import torch
from torch.nn import functional as F

from models.EfficientNet.deeplab_efficiennetb1 import DeepLabv3_plus as deeplab_efficiennetb3


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def evaluate_score(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.000001)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.000001)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc


def evaluate(args, model):
    start_time = time.time()
    model.eval()
    image_path = os.path.join(args.data_root, 'Test/image')
    label_path = os.path.join(args.data_root, 'Test/label')
    image_name = glob.glob(os.path.join(image_path, '*.jpg'))
    image_name.sort()

    gts_all, pts_all = [], []
    time_all = 0
    print('Start evaluation on the test set !!!')
    for k in range(len(image_name)):
        image = cv2.imread(image_name[k])
        label = cv2.imread(os.path.join(label_path, image_name[k].split('/')[-1][:-4]+'.png'), -1)
        # if k % 1 == 0:
        #     print("%d/%d, %s" % (k+1, len(image_name), image_name[k].split('/')[-1]))

        # acquire results
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image) / 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0).copy()
        image = torch.from_numpy(image)
        image = image.cuda()

        # eval
        st_time = time.time()
        with torch.no_grad():
            output = model(image)
            output = F.log_softmax(output, dim=1)
        prediction = output.data.squeeze(0).max(0)[1].cpu().numpy()
        ed_time = time.time()
        time_all += ed_time - st_time

        gts_all.append(label)
        pts_all.append(prediction)

    miou = evaluate_score(pts_all, gts_all, args.num_classes)
    avg_time = time_all/len(image_name)

    del gts_all, pts_all
    gc.collect()
    end_time = time.time()
    print('Evaluation elapse: %.2f s' % (end_time - start_time))
    return miou, avg_time


def evaluate_quan(args, model, is_trt=True):
    if not is_trt:
        model.eval()
    image_path = os.path.join(args.data_root, 'Test/image')
    label_path = os.path.join(args.data_root, 'Test/label')
    image_name = glob.glob(os.path.join(image_path, '*.jpg'))
    image_name.sort()

    gts_all, pts_all = [], []
    time_all = 0
    logging.info('Start testing ...')
    for k in range(len(image_name)):
        image = cv2.imread(image_name[k])
        label = cv2.imread(os.path.join(label_path, image_name[k].split('/')[-1][:-4]+'.png'), -1)
        if k % 1 == 0:
            logging.info("%d/%d, %s" % (k+1, len(image_name), image_name[k].split('/')[-1]))

        # pre-process
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image) / 255.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0).copy()
        image = torch.from_numpy(image.astype(np.float32))
        image = image.cuda()
        # print(image.shape)
        # inference
        with torch.no_grad():
            if is_trt:
                # image = image.unsqueeze(0)
                # print('image_shape', image.shape)
                output, trt_infer_time = model.inference(image)
                output = output.reshape(-1, args.num_classes, 2048, 2048)
                # output = output.reshape(-1, 256, 64, 64)
                # output = output.reshape(-1, 1536, 16, 16)
                # output = output.reshape(-1, 32, 512, 512)
                # output = output.reshape(-1, 24, 1024, 1024)
                # output = output.reshape(-1, 40, 1024, 1024)
                time_all += trt_infer_time
                # print('output_shape after quan:', output.shape)
            else:
                st_time = time.time()
                # print('image_shape', image.shape)
                output = model(image)
                ed_time = time.time()
                time_all += ed_time - st_time
                # print('output_shape before quan:', output.shape)

        # # debug
        # output = output.flatten()
        # print(output[:10])
        # return output.cpu().numpy()

        output = F.log_softmax(output, dim=1)
        prediction = output.data.squeeze(0).max(0)[1].cpu().numpy()
        gts_all.append(label)
        pts_all.append(prediction)
    miou = evaluate_score(pts_all, gts_all, args.num_classes)
    avg_time = time_all / len(image_name)
    fps = len(image_name) / time_all

    del gts_all, pts_all
    gc.collect()
    return miou, avg_time, fps


if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    numClasses = 6

    model_path = './pretrained/model.pth'
    model = deeplab_efficiennetb3(nInputChannels=3, n_classes=numClasses, os=16, pretrained=False, _print=False)

    # load pretrained weights
    save_point = torch.load('./pretrained/model.pth')
    model_param = model.state_dict()
    state_dict_param = save_point['state_dict']
    for key in model_param.keys():
        if key == 'efficientnet_features._blocks.0._depthwise_conv.conv.weight':
            print(key)
        if key in state_dict_param.keys():
            model_param[key] = state_dict_param[key]
        else:
            new_key = key.split('.')
            del new_key[-2]
            new_key = ".".join(new_key)
            assert new_key in state_dict_param.keys()
            model_param[key] = state_dict_param[new_key]
    model.load_state_dict(model_param)

    model = model.cuda()

    miou, avg_time = evaluate(model)
    print("[Test MIoU = %.4f] [Avg eval time = %.4f s]" % (miou, avg_time))
