import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
from SSD_Pytorch.utils_.nms_wrapper import nms, soft_nms
from SSD_Pytorch.configs.config import cfg, cfg_from_file
import numpy as np
import time
import os
import sys
import pickle
import datetime
import yaml

from thop import profile
from thop import clever_format
from prune_model import SSD
from lib.compression.pytorch import ModelSpeedup
from lib.algorithms.pytorch.pruning import (TaylorFOWeightFilterPruner, FPGMPruner, AGPPruner)


def arg_parse():
    parser = argparse.ArgumentParser(description='SSD Training')
    parser.add_argument('--cfg', dest='cfg_file', default='algorithms/compression/nets/ResNet50_SSD/configs/prune.yaml', help='Config file for training (and optionally testing)')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--ngpu', default=1, type=int, help='gpus')
    parser.add_argument('--resume_net', default="./checkpoint/origin.pth", help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')

    parser.add_argument('--save_folder', default='logs/prune/test', help='Location to save checkpoint models')
    # for pruning
    parser.add_argument('--baseline', action='store_true', default=False, help='evaluate model on validation set')
    parser.add_argument('--pruner', default='fpgm', type=str, help='pruner: agp|taylor|fpgm')
    parser.add_argument('--sparsity', default=0.2, type=float, metavar='LR', help='prune sparsity')

    parser.add_argument('--data', default='', type=str, help='dataset path')
    parser.add_argument('--finetune_epochs', default=10, type=int, metavar='N', help='number of epochs for exported model')
    parser.add_argument('--finetune_lr', default=0.001, type=float, metavar='N', help='number of lr for exported model')
    parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--write_yaml', action='store_true', default=False, help='write yaml file')
    parser.add_argument('--no_write_yaml_after_prune', action='store_true', default=False, help='')
    args = parser.parse_args()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    return args


def adjust_learning_rate(optimizer, epoch, step_epoch, gamma, epoch_size, iteration):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    ## warmup
    if epoch <= cfg.TRAIN.WARMUP_EPOCH:
        if cfg.TRAIN.WARMUP:
            iteration += (epoch_size * (epoch - 1))
            lr = 1e-6 + (cfg.SOLVER.BASE_LR - 1e-6) * iteration / (epoch_size * cfg.TRAIN.WARMUP_EPOCH)
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


def train(train_loader, net, criterion, optimizer, epoch, epoch_step, gamma, end_epoch, cfg):
    net.train()
    begin = time.time()
    epoch_size = len(train_loader)
    for iteration, (imgs, targets, _) in enumerate(train_loader):
        t0 = time.time()
        lr = adjust_learning_rate(optimizer, epoch, epoch_step, gamma, epoch_size, iteration)
        imgs = imgs.cuda()
        imgs.requires_grad_()
        with torch.no_grad():
            targets = [anno.cuda() for anno in targets]
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
            odm_loss_l, odm_loss_c = odm_criterion(output, targets, use_arm=True, filter_object=True)
            loss = arm_loss_l + arm_loss_c + odm_loss_l + odm_loss_c
        loss.backward()
        optimizer.step()
        t1 = time.time()
        iteration_time = t1 - t0
        all_time = ((end_epoch - epoch) * epoch_size + (epoch_size - iteration)) * iteration_time
        eta = str(datetime.timedelta(seconds=int(all_time)))
        if iteration % 10 == 0:
            if not cfg.MODEL.REFINE:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) + ' || L: %.4f C: %.4f||' % (loss_l.item(), loss_c.item()) + 'iteration time: %.4f sec. ||' % (t1 - t0) + 'LR: %.5f' % (lr) +
                      ' || eta time: {}'.format(eta))
            else:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size) + '|| arm_L: %.4f arm_C: %.4f||' % (arm_loss_l.item(), arm_loss_c.item()) + ' odm_L: %.4f odm_C: %.4f||' %
                      (odm_loss_l.item(), odm_loss_c.item()) + ' loss: %.4f||' % (loss.item()) + 'iteration time: %.4f sec. ||' % (t1 - t0) + 'LR: %.5f' % (lr) + ' || eta time: {}'.format(eta))


def save_checkpoint(name, net):
    save_name = os.path.join(args.save_folder, name)
    torch.save({'model': net.state_dict()}, save_name)


def eval_net(val_dataset, val_loader, net, detector, cfg, transform, eval_save_folder, max_per_image=300, thresh=0.01, batch_size=1):
    with torch.no_grad():
        net.eval()
        num_images = len(val_dataset)
        num_classes = cfg.MODEL.NUM_CLASSES
        if not os.path.exists(eval_save_folder):
            os.mkdir(eval_save_folder)
        all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
        det_file = os.path.join(eval_save_folder, 'detections.pkl')
        st = time.time()
        total_infer_time = 0
        count = 0
        for idx, (imgs, _, img_info) in enumerate(val_loader):
            with torch.no_grad():
                t1 = time.time()
                x = imgs
                x = x.cuda()
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
                        # print(idx, k, j)
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
                total_infer_time += forward_time
                count += 1
        print("detect time: ", time.time() - st)
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        print('Evaluating detections')
        mAP = val_dataset.evaluate_detections(all_boxes, eval_save_folder)
        return float(mAP), total_infer_time / count


def count_flops_params(net):
    # flops and params
    with torch.no_grad():
        torch.set_default_tensor_type('torch.FloatTensor')
        input = torch.randn(1, 3, 300, 300).cuda()
        macs, params = profile(net, inputs=(input, ), verbose=False)
        macs_f, params_f = clever_format([macs, params], "%.3f")
        print('FLOPs: {}, params: {}'.format(macs_f, params_f))
        for m in net.modules():
            if hasattr(m, 'total_params'):
                del m.total_params
            if hasattr(m, 'total_ops'):
                del m.total_ops
        return macs, params


def main():
    global args
    args = arg_parse()
    cfg_from_file(args.cfg_file)
    save_folder = args.save_folder
    batch_size = args.batch_size
    print('\n\n\n')
    print(batch_size)
    print('\n\n\n')
    bgr_means = cfg.TRAIN.BGR_MEAN
    p = 0.6
    gamma = cfg.SOLVER.GAMMA
    momentum = cfg.SOLVER.MOMENTUM
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    size = cfg.MODEL.SIZE
    thresh = cfg.TEST.CONFIDENCE_THRESH
    if cfg.DATASETS.DATA_TYPE == 'VOC':
        trainvalDataset = VOCDetection
        top_k = 200
    else:
        trainvalDataset = COCODetection
        top_k = 300
    dataset_name = cfg.DATASETS.DATA_TYPE
    #dataroot = args.data
    dataroot = cfg.DATASETS.DATAROOT
    trainSet = cfg.DATASETS.TRAIN_TYPE
    valSet = cfg.DATASETS.VAL_TYPE
    num_classes = cfg.MODEL.NUM_CLASSES
    start_epoch = args.resume_epoch
    epoch_step = cfg.SOLVER.EPOCH_STEPS
    end_epoch = args.finetune_epochs
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net = SSD(cfg)
    print(net)
    if cfg.MODEL.SIZE == '300':
        size_cfg = cfg.SMALL
    else:
        size_cfg = cfg.BIG
    optimizer = optim.SGD(net.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=momentum, weight_decay=weight_decay)
    if args.resume_net != None:
        checkpoint = torch.load(args.resume_net)
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
        print('Loading resume network...')
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)
    net.cuda()
    cudnn.benchmark = True

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

    TrainTransform = preproc(size_cfg.IMG_WH, bgr_means, p)
    ValTransform = BaseTransform(size_cfg.IMG_WH, bgr_means, (2, 0, 1))

    val_dataset = trainvalDataset(dataroot, valSet, ValTransform, dataset_name)
    val_loader = data.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=detection_collate)

    ######################## prune start ########################
    def trainer(model, optimizer, criterion, epoch):
        train_dataset = trainvalDataset(dataroot, trainSet, TrainTransform, dataset_name)
        epoch_size = len(train_dataset)
        train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate)
        train(train_loader, model, criterion, optimizer, epoch, epoch_step, 1, 100, cfg)

    device = torch.device('cuda')

    # torch.save(net, '/home/xingxing/projects/StarLight/data/compression/model_vis/VOC-ResNet50SSD/model.pth')
    # exit()

    # baseline
    print("\nInfer before pruning:")
    flops, params = count_flops_params(net)
    if args.baseline:
        eval_net(val_dataset, val_loader, net, detector, cfg, ValTransform, os.path.join(args.save_folder, 'baseline'), top_k, thresh=thresh, batch_size=batch_size)

    if args.write_yaml:
        flops, params = flops, params
        mAP, infer_time = eval_net(val_dataset, val_loader, net, detector, cfg, ValTransform, os.path.join(args.save_folder, 'baseline'), top_k, thresh=thresh, batch_size=batch_size)
        storage = os.path.getsize(args.resume_net)
        with open(os.path.join(args.save_folder, 'logs.yaml'), 'w') as f:
            yaml_data = {
                'mAP': {'baseline': round(mAP*100, 2), 'method': None},
                'FLOPs': {'baseline': round(flops/1e9, 2), 'method': None},
                'Parameters': {'baseline': round(params/1e6, 2), 'method': None},
                'Infer_times': {'baseline': round(infer_time*1e3, 2), 'method': None},
                'Storage': {'baseline': round(storage/1e6, 2), 'method': None},
            }
            yaml.dump(yaml_data, f)

    op_names = [
        'extractor.conv1',
        'extractor.layer1.0.conv1',
        'extractor.layer1.0.conv2',
        'extractor.layer1.0.conv3',
        'extractor.layer1.0.downsample.0',
        'extractor.layer1.1.conv1',
        'extractor.layer1.1.conv2',
        'extractor.layer1.1.conv3',
        'extractor.layer1.2.conv1',
        'extractor.layer1.2.conv2',
        'extractor.layer1.2.conv3',
        'extractor.layer2.0.conv1',
        'extractor.layer2.0.conv2',
        'extractor.layer2.0.conv3',
        'extractor.layer2.0.downsample.0',
        'extractor.layer2.1.conv1',
        'extractor.layer2.1.conv2',
        'extractor.layer2.1.conv3',
        'extractor.layer2.2.conv1',
        'extractor.layer2.2.conv2',
        'extractor.layer2.2.conv3',
        'extractor.layer2.3.conv1',
        'extractor.layer2.3.conv2',
        'extractor.layer2.3.conv3',
        'extractor.layer3.0.conv1',
        'extractor.layer3.0.conv2',
        'extractor.layer3.0.conv3',
        'extractor.layer3.0.downsample.0',
        'extractor.layer3.1.conv1',
        'extractor.layer3.1.conv2',
        'extractor.layer3.1.conv3',
        'extractor.layer3.2.conv1',
        'extractor.layer3.2.conv2',
        'extractor.layer3.2.conv3',
        'extractor.layer3.3.conv1',
        'extractor.layer3.3.conv2',
        'extractor.layer3.3.conv3',
        'extractor.layer3.4.conv1',
        'extractor.layer3.4.conv2',
        'extractor.layer3.4.conv3',
        'extractor.layer3.5.conv1',
        'extractor.layer3.5.conv2',
        'extractor.layer3.5.conv3',
        'extractor.layer4.0.conv1',
        'extractor.layer4.0.conv2',
        'extractor.layer4.0.conv3',
        'extractor.layer4.0.downsample.0',
        'extractor.layer4.1.conv1',
        'extractor.layer4.1.conv2',
        'extractor.layer4.1.conv3',
        'extractor.layer4.2.conv1',
        'extractor.layer4.2.conv2',
        'extractor.layer4.2.conv3',
        'extractor.extras.0',
        'extractor.extras.1',
        'extractor.extras.2',
        'extractor.extras.3',
        'extractor.extras.4',
        'extractor.extras.5',
        'extractor.smooth1',
    ]

    if args.pruner == 'agp':
        config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d'], 'op_names': op_names}]  # , 'op_names': op_names
        pruner = AGPPruner(
            net,
            config_list,
            optimizer,
            trainer,
            criterion=criterion,
            num_iterations=3,
            epochs_per_iteration=1,
            pruning_algorithm='l1',
        )
    elif args.pruner == 'taylor':
        config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d'], 'op_names': op_names}]  # , 'op_names': op_names
        pruner = TaylorFOWeightFilterPruner(
            net,
            config_list,
            optimizer,
            trainer,
            criterion=None,
            sparsifying_training_batches=1,
        )
    elif args.pruner == 'fpgm':
        config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d'], 'op_names': op_names}]  # , 'op_names': op_names
        pruner = FPGMPruner(
            net,
            config_list,
            optimizer,
            dummy_input=torch.randn(2, 3, 300, 300).to(device),
        )
    else:
        raise NotImplementedError
    pruner.compress()
    # print("\nInfer after pruning:")
    # eval_net(val_dataset, val_loader, net, detector, cfg, ValTransform, os.path.join(args.save_folder, 'after_prune'), top_k, thresh=thresh, batch_size=batch_size)

    # save masked pruned model (model + mask)
    pruner.export_model(os.path.join(args.save_folder, 'model_masked.pth'), os.path.join(args.save_folder, 'mask.pth'))

    # export pruned model
    print('Speed up masked model...')
    net = SSD(cfg).to(device)
    net.load_state_dict(torch.load(os.path.join(args.save_folder, 'model_masked.pth')))
    masks_file = os.path.join(args.save_folder, 'mask.pth')

    m_speedup = ModelSpeedup(net, torch.randn(2, 3, 300, 300).to(device), masks_file, device)
    m_speedup.speedup_model()
    net = net.to(device)
    net.priors = net.priors.to(device)

    # finetune
    optimizer = optim.SGD(net.parameters(), lr=args.finetune_lr, momentum=momentum, weight_decay=weight_decay)
    for epoch in range(start_epoch + 1, end_epoch + 1):
        train_dataset = trainvalDataset(dataroot, trainSet, TrainTransform, dataset_name)
        epoch_size = len(train_dataset)
        train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate)
        train(train_loader, net, criterion, optimizer, epoch, epoch_step, gamma, end_epoch, cfg)
    print("\nInfer after finetune:")
    flops, params = count_flops_params(net)
    mAP, infer_time = eval_net(val_dataset, val_loader, net, detector, cfg, ValTransform, os.path.join(args.save_folder, 'export'), top_k, thresh=thresh, batch_size=batch_size)
    save_checkpoint("export_pruned_model.pth", net)

    if args.write_yaml and not args.no_write_yaml_after_prune:
        storage = os.path.getsize(os.path.join(args.save_folder, 'export_pruned_model.pth'))
        with open(os.path.join(args.save_folder, 'logs.yaml'), 'w') as f:
            yaml_data = {
                'mAP': {'baseline': yaml_data['mAP']['baseline'], 'method': round(mAP*100, 2)},
                'FLOPs': {'baseline': yaml_data['FLOPs']['baseline'], 'method': round(flops/1e9, 2)},
                'Parameters': {'baseline': yaml_data['Parameters']['baseline'], 'method': round(params/1e6, 2)},
                'Infer_times': {'baseline': yaml_data['Infer_times']['baseline'], 'method': round(infer_time*1e3, 2)},
                'Storage': {'baseline': yaml_data['Storage']['baseline'], 'method': round(storage/1e6, 2)},
                'Output_file': os.path.join(args.save_folder, 'model_speed_up_finetuned.pth'),
            }
            yaml.dump(yaml_data, f)
        torch.save(net, \
            os.path.join(
                args.save_folder, \
                '../../..', \
                'model_vis/VOC-ResNet50SSD', \
                'online-{}.pth'.format(args.pruner)))

    ######################## prune end   ########################


if __name__ == '__main__':
    main()
