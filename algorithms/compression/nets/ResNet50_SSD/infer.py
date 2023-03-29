import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from SSD_Pytorch.data_ import COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from SSD_Pytorch.layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from SSD_Pytorch.layers.functions import Detect
from SSD_Pytorch.utils.nms_wrapper import nms, soft_nms
from configs.config import cfg, cfg_from_file
import numpy as np
import time
import os
import sys
import pickle
import datetime
from SSD_Pytorch.models.model_builder import SSD
import yaml

from thop import profile
from thop import clever_format

from prune_model import SSD as PruneSSD
from quan_model import SSD as QuanSSD
from lib.compression.pytorch import ModelSpeedup
from lib.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT


def arg_parse():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
    parser.add_argument('--weights', default='SSD_Pytorch/weights/ssd_res50_epoch_250_300.pth', type=str, help='Trained state_dict file path to open')
    parser.add_argument('--cfg', dest='cfg_file', default='SSD_Pytorch/configs/ssd_res50_voc.yaml', help='Config file for training (and optionally testing)')
    parser.add_argument('--save_folder', default='logs/infer/test/', type=str, help='File path to save results')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--retest', default=False, type=bool, help='test cache results')

    parser.add_argument('--mode', default='baseline', help='baseline prune quan prune_quan', type=str)
    parser.add_argument('--prune_eval_path', default=None, type=str, metavar='PATH', help='path to eval pruned model')
    parser.add_argument('--quan_mode', default='fp16', help='fp16 int8 best', type=str)
    parser.add_argument('--quan_eval_path', default=None, type=str, metavar='PATH', help='path to eval quan model')
    args = parser.parse_args()
    return args


def count_flops_params(net):
    # flops and params
    torch.set_default_tensor_type('torch.FloatTensor')
    input = torch.randn(1, 3, 300, 300).cuda()
    macs, params = profile(net, inputs=(input, ), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('FLOPs: {}, params: {}'.format(macs, params))
    for m in net.modules():
        if hasattr(m, 'total_params'):
            del m.total_params
        if hasattr(m, 'total_ops'):
            del m.total_ops


def eval_net(val_dataset, val_loader, net, detector, cfg, transform, eval_save_folder, max_per_image=300, thresh=0.01, batch_size=1):
    net.eval()
    num_images = len(val_dataset)
    num_classes = cfg.MODEL.NUM_CLASSES
    # eval_save_folder = "./eval/"
    if not os.path.exists(eval_save_folder):
        os.mkdir(eval_save_folder)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    det_file = os.path.join(eval_save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        val_dataset.evaluate_detections(all_boxes, eval_save_folder)
        return

    for idx, (imgs, _, img_info) in enumerate(val_loader):
        with torch.no_grad():
            t1 = time.time()
            x = imgs
            x = x.cuda()
            if hasattr(net, 'engine'):
                output = net.inference(x)
                forward_time = output[2]
            else:
                output = net(x)
                forward_time = time.time() - t1
            output = (output[0].cuda(), output[1].cuda(), net.priors)
            t4 = time.time()
            boxes, scores = detector.forward(output)
            t2 = time.time()
            for k in range(boxes.size(0)):
                i = idx * batch_size + k
                boxes_ = boxes[k]
                scores_ = scores[k]
                boxes_ = boxes_.cpu().numpy()
                scores_ = scores_.cpu().numpy()
                img_wh = img_info[k]
                scale = np.array([img_wh[0], img_wh[1], img_wh[0], img_wh[1]])
                boxes_ *= scale
                for j in range(1, num_classes):
                    inds = np.where(scores_[:, j] > thresh)[0]
                    if len(inds) == 0:
                        all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_bboxes = boxes_[inds]
                    c_scores = scores_[inds, j]
                    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = nms(c_dets, cfg.TEST.NMS_OVERLAP, force_cpu=True)
                    keep = keep[:50]
                    c_dets = c_dets[keep, :]
                    all_boxes[j][i] = c_dets
            t3 = time.time()
            detect_time = t2 - t1
            nms_time = t3 - t2
            forward_time = t4 - t1
            if idx % 10 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s'.format(i + 1, num_images, forward_time, detect_time, nms_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    val_dataset.evaluate_detections(all_boxes, eval_save_folder)
    print("detect time: ", time.time() - st)


def main():
    global args
    args = arg_parse()
    cfg_from_file(args.cfg_file)
    bgr_means = cfg.TRAIN.BGR_MEAN
    dataset_name = cfg.DATASETS.DATA_TYPE
    batch_size = cfg.TEST.BATCH_SIZE
    num_workers = args.num_workers
    if cfg.DATASETS.DATA_TYPE == 'VOC':
        trainvalDataset = VOCDetection
        top_k = 200
    else:
        trainvalDataset = COCODetection
        top_k = 300
    dataroot = cfg.DATASETS.DATAROOT
    if cfg.MODEL.SIZE == '300':
        size_cfg = cfg.SMALL
    else:
        size_cfg = cfg.BIG
    valSet = cfg.DATASETS.VAL_TYPE
    num_classes = cfg.MODEL.NUM_CLASSES
    save_folder = args.save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cfg.TRAIN.TRAIN_ON = False
    net = SSD(cfg)

    checkpoint = torch.load(args.weights)
    state_dict = checkpoint['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    detector = Detect(cfg)
    ValTransform = BaseTransform(size_cfg.IMG_WH, bgr_means, (2, 0, 1))
    val_dataset = trainvalDataset(dataroot, valSet, ValTransform, "val")
    val_loader = data.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=detection_collate)
    top_k = 300
    thresh = cfg.TEST.CONFIDENCE_THRESH

    device = torch.device('cuda')
    if args.mode == 'baseline':
        count_flops_params(net)
    elif args.mode == 'prune':
        assert os.path.isdir(args.prune_eval_path)
        prune_net = PruneSSD(cfg).to(device)
        print("=> loading pruned model '{}'".format(args.prune_eval_path))
        prune_net.load_state_dict(torch.load(os.path.join(args.prune_eval_path, 'model_masked.pth')))
        masks_file = os.path.join(args.prune_eval_path, 'mask.pth')
        m_speedup = ModelSpeedup(prune_net, torch.randn(1, 3, 300, 300).to(device), masks_file, device)
        m_speedup.speedup_model()
        prune_net.load_state_dict(torch.load(os.path.join(args.prune_eval_path, 'export_pruned_model.pth'))['model'])
        print("=> loaded pruned model")
        prune_net.eval().to(device)
        net = prune_net
        count_flops_params(net)
    elif args.mode == 'quan' or args.mode == 'prune_quan':
        input_shape = (1, 3, 300, 300)
        input_names = ["actual_input_1"]
        output_names = ["output1", "output2"]

        onnx_path = os.path.join(args.quan_eval_path, '{}.onnx'.format(args.quan_mode))
        trt_path = os.path.join(args.quan_eval_path, '{}.trt'.format(args.quan_mode))
        cache_path = os.path.join(args.quan_eval_path, '{}.cache'.format(args.quan_mode))

        if args.quan_mode == "int8":
            extra_layer_bit = 8
        elif args.quan_mode == "fp16":
            extra_layer_bit = 16
        elif args.quan_mode == "best":
            extra_layer_bit = -1
        else:
            extra_layer_bit = 32

        engine = ModelSpeedupTensorRT(
            net,
            input_shape,
            config=None,
            calib_data_loader=None,
            batchsize=1,
            onnx_path=onnx_path,
            calibration_cache=cache_path,
            extra_layer_bit=extra_layer_bit,
            input_names=input_names,
            output_names=output_names,
        )
        engine.load_quantized_model(trt_path)
        net = QuanSSD(cfg).to(device)
        net.load_engine(engine)
    else:
        raise NotImplementedError

    eval_net(val_dataset, val_loader, net, detector, cfg, ValTransform, os.path.join(args.save_folder, 'eval'), top_k, thresh=thresh, batch_size=batch_size)


if __name__ == '__main__':
    st = time.time()
    main()
    print("final time", time.time() - st)
