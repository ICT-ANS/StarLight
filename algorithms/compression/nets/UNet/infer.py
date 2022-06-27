import argparse
import logging
import os
from os import listdir
import numpy as np
import torch
import torch.nn as nn
from glob import glob
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from os.path import splitext
from unet import UNet
# from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from utils.mars_dataset import MarsDataset
from utils.mars_img_dataset import MarsImgDataset
from dice_loss import dice_coeff
from dice_loss import DiceAndMPA
import time
from thop import profile
from thop import clever_format
from lib.compression.pytorch import ModelSpeedup
from lib.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT
import time

infer_times = []


def newMPA(pred, label, threshold=0.05):
    """compute MPA metric

    Parameters
    ----------
    pred : torch.tensor
        predict data
    label : torch.tensor
        ground truth
    threshold : float, optional
        default 0.05

    Returns
    -------
    float
        MPA
    """    
    A = label.view(-1).float()
    B = pred.view(-1).float()
    TP = np.fabs(A - B) > threshold + 0.
    l = A.shape[0]
    return (l - float(TP.sum())) / l


def predict_img(net, root_dir, idx, device, scale_factor=1, out_threshold=0.5):
    img_dir_list = [
        root_dir + 'ortho_slope/',
        # root_dir + 'ortho_mosaic/',
        root_dir + 'ortho_depth/',
        root_dir + 'ortho_rough/'
    ]
    input_tensor_list = []
    for channel_dir in img_dir_list:
        txt_file = glob(channel_dir + idx)
        input_channel = Image.open(txt_file[0])
        input_channel = MarsImgDataset.preprocess(input_channel, scale_factor)
        input_channel_tensor = torch.from_numpy(input_channel).type(torch.FloatTensor).unsqueeze(0)
        input_tensor_list.append(input_channel_tensor)

    input_tensor = torch.cat(input_tensor_list, dim=0)
    # from torchvision import transforms
    # pic = transforms.ToPILImage()(input_tensor)
    # pic.save(f'{idx}.jpg')
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device=device, dtype=torch.float32)

    global infer_times

    with torch.no_grad():
        t = time.time()
        if isinstance(net, nn.Module):
            output = net(input_tensor)
        else:  # for trt
            output, _ = net.inference(input_tensor)
            output = output.reshape(-1, 1, 150, 150)
            output = output.to(input_tensor.device)
        infer_time = time.time() - t
        infer_times.append(infer_time)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(full_img.size[1]),
        #         transforms.ToTensor()
        #     ]
        # )

        # probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    # return full_mask > out_threshold
    return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_path', default='Pytorch-UNet-master/data/data/predict/')
    parser.add_argument('--model_path', default='checkpoints/origin.pth')
    parser.add_argument('--output_path', default='logs/infer_origin/')
    parser.add_argument('--prune_eval_path', default='')
    parser.add_argument('--quan_path', default='logs/quan/quan_fp16')
    parser.add_argument('--quan_mode', default='fp16')
    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


maxDice = 0
'''
tips: predict test dataset in virtual dataset.
'''

# def img_predict(net, device, root_path='Pytorch-UNet-master/data/data/predict/', out_path='logs/tmp/'):
#     if not os.path.exists(out_path):
#         os.makedirs(out_path)
#     global maxDice, infer_times
#     maxDice = 0
#     infer_times = []

#     if hasattr(net, 'eval'):
#         net.eval()

#     doc = open(out_path + 'log.txt', 'w')
#     dice = 0
#     mpa = 0
#     data_dir = listdir((root_path + 'ortho_depth/'))
#     l = len(data_dir)
#     for fn in listdir(root_path + 'ortho_depth/'):
#         idx = splitext(fn)[0]

#         mask = predict_img(net=net, root_dir=root_path, idx=fn, scale_factor=1.0, device=device, out_threshold=0.5)

#         result = mask_to_image(mask)

#         result.save(out_path + idx + '.jpg')
#         pred = torch.from_numpy(mask + 0).unsqueeze(0).unsqueeze(0)

#         label_img_file = root_path + 'ortho_traver/' + fn
#         label_channel = Image.open(label_img_file)
#         label_channel = MarsImgDataset.preprocess(label_channel, 1.0, istraver=1)
#         # label_channel = MarsImgDataset.preprocess(label_channel, 1.0)
#         label = torch.from_numpy(label_channel).type(torch.int).unsqueeze(0).unsqueeze(0)
#         # coeff = dice_coeff(pred, label)
#         coeff, m = DiceAndMPA(pred, label)
#         dice += coeff
#         mpa += m
#         print('i = ', idx, file=doc)
#         print(coeff, file=doc)
#         delta = (label - pred).numpy()
#         print(np.sum(delta == -1), np.sum(delta == 1), np.sum(delta == 0), file=doc)
#     print('aver dice:', dice / l, file=doc)
#     print('MPA:', mpa / l, file=doc)
#     if maxDice < dice / l:
#         maxDice = dice / l
#         print('max dice:', maxDice)
#     doc.close()

#     print('infer_time/s:', sum(infer_times) / len(infer_times))

#     return maxDice


def img_predict(net, device, root_path='Pytorch-UNet-master/data/data/predict/', out_path='logs/tmp/'):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    global maxDice, infer_times
    maxDice = 0
    infer_times = []

    doc = open(out_path + 'log.txt', 'w')
    dice = 0
    mpa = 0
    epoch_loss = 0
    data_dir = listdir((root_path + 'ortho_depth/'))
    # l = len(data_dir)
    l = min(len(data_dir), 100)
    count = 0
    for fn in listdir(root_path + 'ortho_depth/'):
        idx = splitext(fn)[0]

        mask = predict_img(net=net, root_dir=root_path, idx=fn, scale_factor=1.0, device=device, out_threshold=0.5)

        result = mask_to_image(mask)

        result.save(out_path + idx + '.jpg')
        pred = torch.from_numpy(mask).type(torch.float).unsqueeze(0).unsqueeze(0)

        label_img_file = root_path + 'ortho_traver/' + fn
        label_channel = Image.open(label_img_file)
        label_channel = MarsImgDataset.preprocess(label_channel, 1.0)
        # label_channel = MarsImgDataset.preprocess(label_channel, 1.0)
        label = torch.from_numpy(label_channel).type(torch.float).unsqueeze(0).unsqueeze(0)
        # coeff = dice_coeff(pred, label)
        coeff, m = DiceAndMPA(pred, label)
        dice += coeff
        mpa += m
        loss = newMPA(pred, label, threshold=0.02)
        epoch_loss += loss

        print('i = ', idx, file=doc)
        print(coeff, file=doc)
        delta = (label - pred).numpy()
        print(np.sum(delta == -1), np.sum(delta == 1), np.sum(delta == 0), file=doc)
        count += 1
        if count == 100:
            break
    # print('aver dice:', dice / l, file=doc)
    print('MPA:', mpa / l, file=doc)
    if maxDice < dice / l:
        maxDice = dice / l
        maxMSE = epoch_loss / l
        print('max mse:', maxMSE)
    doc.close()
    print('MPA:', epoch_loss / l)
    print('infer_time/s:', sum(infer_times) / len(infer_times))

    return epoch_loss / l, sum(infer_times) / len(infer_times)


def count_flops_params(net, device):
    """count net FLOPs and parameters

    Parameters
    ----------
    net : torch.Module
        network
    device : torch.device
        device

    Returns
    -------
    tuple
        FLOPs, Params
    """    
    input = torch.randn(1, 3, 150, 150).to(device)
    net = net.eval()
    macs, params = profile(net, inputs=(input, ), verbose=False)
    c_macs, c_params = clever_format([macs, params], "%.3f")
    print(c_macs, c_params)

    for m in net.modules():
        if hasattr(m, 'total_params'):
            del m.total_params
        if hasattr(m, 'total_ops'):
            del m.total_ops
    
    return macs, params


if __name__ == "__main__":
    # txt_predict()
    args = get_args()
    root_path = args.root_path
    model_path = args.model_path
    out_path = args.output_path

    if args.prune_eval_path:  # eval prune
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.isdir(args.prune_eval_path):
            prune_net = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
            prune_net.set_pad(150)
            print("=> loading pruned model '{}'".format(args.prune_eval_path))
            prune_net.load_state_dict(torch.load(os.path.join(args.prune_eval_path, 'model_masked.pth')))
            masks_file = os.path.join(args.prune_eval_path, 'mask.pth')
            m_speedup = ModelSpeedup(prune_net, torch.randn(1, 3, 150, 150).to(device), masks_file, device)
            m_speedup.speedup_model()
            prune_net.load_state_dict(torch.load(os.path.join(args.prune_eval_path, 'export_pruned_model.pth'))['state_dict'])
            print("=> loaded pruned model")
            net = prune_net.to(device)
            img_predict(net, device, root_path=root_path, out_path=out_path)
            count_flops_params(net, device)
    elif args.quan_path:  # eval quan
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
        net.set_pad(150)

        calib_loader = None

        input_shape = [
            (1, 3, 150, 150),
        ]

        onnx_path = os.path.join(args.quan_path, '{}.onnx'.format(args.quan_mode))
        trt_path = os.path.join(args.quan_path, '{}.trt'.format(args.quan_mode))
        cache_path = os.path.join(args.quan_path, '{}.cache'.format(args.quan_mode))

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
            calib_data_loader=calib_loader,
            batchsize=1,
            onnx_path=onnx_path,
            calibration_cache=cache_path,
            extra_layer_bit=extra_layer_bit,
        )
        if not os.path.exists(trt_path):
            engine.compress()
            engine.export_quantized_model(trt_path)
        else:
            engine.load_quantized_model(trt_path)

        # origin
        engine.n_classes = 1
        img_predict(engine, device)
    else:  # eval origin
        net = UNet(n_channels=3, n_classes=1, bilinear=False)
        net.set_pad(150)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net.to(device=device)
        net.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        # torch.save(net.state_dict(), 'checkpoints/origin0.pth')

        img_predict(net, device, root_path=root_path, out_path=out_path)
        count_flops_params(net, device)
