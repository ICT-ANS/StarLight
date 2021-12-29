import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.enabled = True

import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from layers.modules import MultiBoxLoss, RefineMultiBoxLoss
from layers.functions import Detect
from utils.nms_wrapper import nms, soft_nms
from configs.config import cfg, cfg_from_file
import numpy as np
import time
import os
import sys
import pickle
import datetime
from models.model_builder import SSD
import yaml
from thop import profile

from lib.compression.pytorch import ModelSpeedup
from lib.algorithms.pytorch.pruning import (TaylorFOWeightFilterPruner, FPGMPruner, AGPPruner)

def arg_parse():
    parser = argparse.ArgumentParser(
        description='Single Shot MultiBox Detection')
    parser.add_argument(
        '--weights',
        default='weights/ssd_vgg_epoch_250_300_2.pth',
        type=str,
        help='Trained state_dict file path to open')
    parser.add_argument(
        '--cfg',
        default='./configs/ssd_vgg_voc.yaml',
        dest='cfg_file',
        #required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--save_folder',
        default='./eval/',
        type=str,
        help='File path to save results')
    parser.add_argument(
        '--num_workers',
        default=8,
        type=int,
        help='Number of workers used in dataloading')
    parser.add_argument(
        '--retest', default=False, type=bool, help='test cache results')

    parser.add_argument('--pruner', default='fpgm', type=str, help='pruner: agp|taylor|fpgm')
    parser.add_argument('--sparsity', default=0.2, type=float, metavar='LR', help='prune sparsity')
    parser.add_argument('--save_dir', default='./prune', help='The directory used to save the trained models', type=str)

    args = parser.parse_args()
    return args


def eval_net(val_dataset,
             val_loader,
             net,
             detector,
             cfg,
             transform,
             max_per_image=300,
             thresh=0.01,
             batch_size=1):
    net.eval()
    num_images = len(val_dataset)
    num_classes = cfg.MODEL.NUM_CLASSES
    eval_save_folder = "./eval/"
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

    total_forward_time = 0
    total_detect_time = 0
    total_nms_time = 0
    total_id = 0
    for idx, (imgs, _, img_info) in enumerate(val_loader):
        with torch.no_grad():
            t1 = time.time()
            x = imgs
            x = x.cuda()
            # output = net(x)

            output = net(x)
            output = (output[0], output[1], net.priors)

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
                    c_dets = np.hstack((c_bboxes,
                                        c_scores[:, np.newaxis])).astype(
                                            np.float32, copy=False)
                    # keep = nms(c_dets, cfg.TEST.NMS_OVERLAP, force_cpu=True)
                    keep = nms(c_dets, cfg.TEST.NMS_OVERLAP, force_cpu=True)

                    keep = keep[:50]
                    c_dets = c_dets[keep, :]
                    all_boxes[j][i] = c_dets
            t3 = time.time()
            detect_time = t2 - t1
            nms_time = t3 - t2
            forward_time = t4 - t1
            if idx % 10 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s {:.3f}s'.format(
                    i + 1, num_images, forward_time, detect_time, nms_time))

            total_forward_time += forward_time
            total_detect_time += detect_time
            total_nms_time += nms_time
            total_id += 1
    print('im_detect: average_forward_time {:.3f}s average_detect_time {:.3f}s average_nms_time {:.3f}s'.format(
                    total_forward_time/total_id, total_detect_time/total_id, total_nms_time/total_id))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    val_dataset.evaluate_detections(all_boxes, eval_save_folder)
    print("detect time: ", time.time() - st)


def adjust_learning_rate(optimizer, epoch, step_epoch, gamma, epoch_size,
                         iteration):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    ## warmup
    if epoch <= cfg.TRAIN.WARMUP_EPOCH:
        if cfg.TRAIN.WARMUP:
            iteration += (epoch_size * (epoch - 1))
            lr = 1e-6 + (cfg.SOLVER.BASE_LR - 1e-6) * iteration / (
                epoch_size * cfg.TRAIN.WARMUP_EPOCH)
        else:
            lr = cfg.SOLVER.BASE_LR
    else:
        div = 0
        if epoch > step_epoch[-1]:
            div = len(step_epoch) - 1
        else:
            for idx, v in enumerate(step_epoch):
                if epoch > step_epoch[idx] and epoch <= step_epoch[idx + 1]:
                    div = idx
                    break
        lr = cfg.SOLVER.BASE_LR * (gamma**div)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(train_loader, net, criterion, optimizer, epoch, epoch_step, gamma,
          end_epoch, cfg):
    net.train()
    begin = time.time()
    epoch_size = len(train_loader)
    for iteration, (imgs, targets, _) in enumerate(train_loader):
        t0 = time.time()
        lr = adjust_learning_rate(optimizer, epoch, epoch_step, gamma,
                                  epoch_size, iteration)
        imgs = imgs.cuda()
        imgs.requires_grad_()
        with torch.no_grad():
            targets = [anno.cuda() for anno in targets]
        
        # output = net(imgs)
        output = net(imgs)
        output = (output[0], output[1], net.priors)
        
        optimizer.zero_grad()
        if not cfg.MODEL.REFINE:
            ssd_criterion = criterion[0]
            loss_l, loss_c = ssd_criterion(output, targets)
            loss = loss_l + loss_c
        else:
            arm_criterion = criterion[0]
            odm_criterion = criterion[1]
            arm_loss_l, arm_loss_c = arm_criterion(output, targets)
            odm_loss_l, odm_loss_c = odm_criterion(
                output, targets, use_arm=True, filter_object=True)
            loss = arm_loss_l + arm_loss_c + odm_loss_l + odm_loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        iteration_time = t1 - t0
        all_time = ((end_epoch - epoch) * epoch_size +
                    (epoch_size - iteration)) * iteration_time
        eta = str(datetime.timedelta(seconds=int(all_time)))
        if iteration % 10 == 0:
            if not cfg.MODEL.REFINE:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' +
                      repr(iteration % epoch_size) + '/' + repr(epoch_size) +
                      ' || L: %.4f C: %.4f||' %
                      (loss_l.item(), loss_c.item()) +
                      'iteration time: %.4f sec. ||' % (t1 - t0) +
                      'LR: %.5f' % (lr) + ' || eta time: {}'.format(eta))
            else:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' +
                      repr(iteration % epoch_size) + '/' + repr(epoch_size) +
                      '|| arm_L: %.4f arm_C: %.4f||' %
                      (arm_loss_l.item(), arm_loss_c.item()) +
                      ' odm_L: %.4f odm_C: %.4f||' %
                      (odm_loss_l.item(), odm_loss_c.item()) +
                      ' loss: %.4f||' % (loss.item()) +
                      'iteration time: %.4f sec. ||' % (t1 - t0) +
                      'LR: %.5f' % (lr) + ' || eta time: {}'.format(eta))


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
        os.mkdir(save_folder)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    cfg.TRAIN.TRAIN_ON = False
    net = SSD(cfg)
    print(net)

    # 统计原始模型参数量Params和计算量FLOPs
    torch.set_default_tensor_type('torch.FloatTensor')
    input = torch.randn(1, 3, 300, 300).cuda()
    flops, params = profile(net, (input,))
    print('Original flops: %.2f G, params: %.2f M' % (flops/1e9, params/1e6))
    for m in net.modules():
        if hasattr (m, 'total_params'):
            del m.total_params
        if hasattr (m, 'total_ops'):
            del m.total_ops

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
    
# extractor.vgg.0
# extractor.vgg.2
# extractor.vgg.5
# extractor.vgg.7
# extractor.vgg.10
# extractor.vgg.12
# extractor.vgg.14
# extractor.vgg.17
# extractor.vgg.19
# extractor.vgg.21
# extractor.vgg.24
# extractor.vgg.26
# extractor.vgg.28
# extractor.vgg.31
# extractor.vgg.33
# extractor.extras.0
# extractor.extras.1
# extractor.extras.2
# extractor.extras.3
# extractor.extras.4
# extractor.extras.5
# extractor.extras.6
# extractor.extras.7
# arm_loc.0
# arm_loc.1
# arm_loc.2
# arm_loc.3
# arm_loc.4
# arm_loc.5
# arm_conf.0
# arm_conf.1
# arm_conf.2
# arm_conf.3
# arm_conf.4
# arm_conf.5
    op_names = [
                'extractor.vgg.0',
                'extractor.vgg.2',
                'extractor.vgg.5',
                'extractor.vgg.7',
                'extractor.vgg.10',
                'extractor.vgg.12',
                'extractor.vgg.14',
                'extractor.vgg.17',
                'extractor.vgg.19',
                #'extractor.vgg.21', #解决BN与conv不一致
                'extractor.vgg.24',
                'extractor.vgg.26',
                'extractor.vgg.28',
                'extractor.vgg.31',
                'extractor.vgg.33',
                'extractor.extras.0',
                'extractor.extras.1',
                'extractor.extras.2',
                'extractor.extras.3',
                'extractor.extras.4',
                'extractor.extras.5',
                'extractor.extras.6',
                'extractor.extras.7',
                ]
    if args.pruner == 'fpgm':
        config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d'],
                        'op_names':op_names}]
        pruner = FPGMPruner(
            net,
            config_list,
            #optimizer,
            dummy_input=torch.rand(1, 3, 300, 300).cuda(),
        )
    else:
        raise NotImplementedError
    
    pruned_model = pruner.compress()

    pruner.export_model(os.path.join(args.save_dir, 'model_masked.pth'), 
                                os.path.join(args.save_dir, 'mask.pth'))
    
    # export pruned model
    print('Speed up masked model...')
    model = SSD(cfg).cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_masked.pth')))
    masks_file = os.path.join(args.save_dir, 'mask.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_speedup = ModelSpeedup(model, torch.rand(1, 3, 300, 300).cuda(), masks_file, device)
    m_speedup.speedup_model()

    #保存导出的剪枝模型
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_speed_up.pth'))

    #统计剪枝模型参数量Params和计算量FLOPs
    torch.set_default_tensor_type('torch.FloatTensor')
    input = torch.randn(1, 3, 300, 300).cuda()
    flops, params = profile(model, (input,))
    print('Pruned flops: %.2f G, params: %.2f M' % (flops/1e9, params/1e6))
    for m in model.modules():
        if hasattr (m, 'total_params'):
            del m.total_params
        if hasattr (m, 'total_ops'):
            del m.total_ops

    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_speed_up.pth')))

    detector = Detect(cfg)
    ValTransform = BaseTransform(size_cfg.IMG_WH, bgr_means, (2, 0, 1))
    val_dataset = trainvalDataset(dataroot, valSet, ValTransform, "val")
    val_loader = data.DataLoader(
        val_dataset,
        batch_size,
        #8,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate)
    
    ## Val pruned model without finetuning ##
    # top_k = 300
    # thresh = cfg.TEST.CONFIDENCE_THRESH
    # eval_net(
    #     val_dataset,
    #     val_loader,
    #     model,
    #     detector,
    #     cfg,
    #     ValTransform,
    #     top_k,
    #     thresh=thresh,
    #     batch_size=batch_size)
    
    ## Finetune Pruned Model ##
    trainSet = cfg.DATASETS.TRAIN_TYPE
    p = 0.6
    TrainTransform = preproc(size_cfg.IMG_WH, bgr_means, p)
    thresh = cfg.TEST.CONFIDENCE_THRESH
    criterion = list()
    if cfg.MODEL.REFINE:
        detector = Detect(cfg)
        arm_criterion = RefineMultiBoxLoss(cfg, 2)
        odm_criterion = RefineMultiBoxLoss(cfg, cfg.MODEL.NUM_CLASSES)
        criterion.append(arm_criterion)
        criterion.append(odm_criterion)
    else:
        detector = Detect(cfg)
        ssd_criterion = MultiBoxLoss(cfg)
        criterion.append(ssd_criterion)
    momentum = cfg.SOLVER.MOMENTUM
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        momentum=momentum,
        weight_decay=weight_decay)
    epoch_step = cfg.SOLVER.EPOCH_STEPS
    end_epoch = cfg.SOLVER.END_EPOCH
    gamma = cfg.SOLVER.GAMMA

    # for epoch in range(start_epoch + 1, end_epoch + 1):
    for epoch in range(1, 11):
        train_dataset = trainvalDataset(dataroot, trainSet, TrainTransform,
                                        dataset_name)
        epoch_size = len(train_dataset)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=detection_collate)
        train(train_loader, model, criterion, optimizer, epoch, epoch_step,
              gamma, end_epoch, cfg)
        
        #保存微调后的剪枝模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_pruned_finetune_10.pth'))

        #加载微调后的剪枝模型
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_pruned_finetune_10.pth')))
        
        eval_net(val_dataset,
                val_loader,
                model,
                detector,
                cfg,
                ValTransform,
                top_k,
                thresh=thresh,
                batch_size=batch_size)
    
    # top_k = 300
    # thresh = cfg.TEST.CONFIDENCE_THRESH
    # eval_net(
    #     val_dataset,
    #     val_loader,
    #     model,
    #     detector,
    #     cfg,
    #     ValTransform,
    #     top_k,
    #     thresh=thresh,
    #     batch_size=batch_size)


if __name__ == '__main__':
    st = time.time()
    main()
    print("final time", time.time() - st)
