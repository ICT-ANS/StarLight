import argparse
import logging
import os
from os import listdir
import numpy as np
import torch
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
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(input_tensor)

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

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true', help="Visualize the images as they are processed", default=False)
    parser.add_argument('--no-save', '-n', action='store_true', help="Do not save the output masks", default=False)
    parser.add_argument('--mask-threshold', '-t', type=float, help="Minimum probability value to consider a mask pixel white", default=0.5)
    parser.add_argument('--scale', '-s', type=float, help="Scale factor for the input images", default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def mars_prediect(net, root_path, idx, scale_factor, device, geometry_only=False, out_threshold=0.5):
    if geometry_only:
        channel_names = ['elevation/', 'slope/']
    else:
        channel_names = ['elevation/', 'roughness/', 'gap/', 'granularity/', 'slope/']
    channel_list = []
    for channel_name in channel_names:
        channel_file = root_path + channel_name + idx
        channel = MarsDataset.channel_loader(channel_file)
        channel_list.append(channel)
    img = torch.cat(channel_list, dim=0)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        time_start = time.time()
        output = net(img)
        time_end = time.time()
        print('Time cost = %fs' % (time_end - time_start))
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)
        # print(probs)
        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(100),
        #         transforms.ToTensor()
        #     ]
        # )
        #
        # probs = tf(probs.cpu())
        # print(probs)
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask


def vis_travers(root_path):
    for fn in listdir(root_path + 'elevation/'):
        idx = splitext(fn)[0]
        label_img_file = root_path + 'traversability/' + fn
        label = MarsDataset.channel_loader(label_img_file).numpy().squeeze(0)
        result = mask_to_image(label)
        result.save(root_path + 'traversability/vis/' + idx + '.jpg')


def smoothstep(low, high, x):
    k = (x - low) / (high - low)
    if k < 0:
        k = 0
    if k > 1:
        k = 1
    return k * k * (3 - 2 * k)


def SlopeAndHighRisk():
    file_name = 'slope.txt'
    with open(file_name) as f:
        file_lines = f.readlines()
    data_list = []
    for line in file_lines:
        line.strip()
        line = (line.replace("\n", " ")).split(" ")[0:-1]
        line = np.array(line).astype(np.float)
        read_line = line.tolist()
        data_list.append(read_line)
    slope = data_list
    slope_mapped = []
    for row in slope:
        slope_mapped_row = []
        for col in row:
            val = smoothstep(0.3, 0.0, col)
            slope_mapped_row.append(val)
        slope_mapped.append(slope_mapped_row)
    tensor = torch.from_numpy(np.array(slope_mapped)).type(torch.FloatTensor).unsqueeze(0)
    tensor = tensor.numpy().squeeze(0)
    result = mask_to_image(tensor)
    result.save('traver.jpg')


maxDice = 0
'''
tips: predict test dataset in virtual dataset.
'''


def img_predict(root_path='', model_path='CP_epoch40.pth', out_path='traver_predict/'):

    net = UNet(n_channels=3, n_classes=1, bilinear=False)

    # logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    logging.info("Model loaded !")
    doc = open(out_path + 'log.txt', 'w')
    dice = 0
    mpa = 0
    data_dir = listdir((root_path + 'ortho_depth/'))
    l = len(data_dir)
    for fn in listdir(root_path + 'ortho_depth/'):
        idx = splitext(fn)[0]

        mask = predict_img(net=net, root_dir=root_path, idx=fn, scale_factor=1.0, device=device, out_threshold=0.5)

        result = mask_to_image(mask)

        result.save(out_path + idx + '.jpg')
        pred = torch.from_numpy(mask + 0).unsqueeze(0).unsqueeze(0)

        label_img_file = root_path + 'ortho_traver/' + fn
        label_channel = Image.open(label_img_file)
        label_channel = MarsImgDataset.preprocess(label_channel, 1.0, istraver=1)
        # label_channel = MarsImgDataset.preprocess(label_channel, 1.0)
        label = torch.from_numpy(label_channel).type(torch.int).unsqueeze(0).unsqueeze(0)
        # coeff = dice_coeff(pred, label)
        coeff, m = DiceAndMPA(pred, label)
        dice += coeff
        mpa += m
        print('i = ', idx, file=doc)
        print(coeff, file=doc)
        delta = (label - pred).numpy()
        print(np.sum(delta == -1), np.sum(delta == 1), np.sum(delta == 0), file=doc)
    print('aver dice:', dice / l, file=doc)
    print('MPA:', mpa / l, file=doc)
    global maxDice
    if maxDice < dice / l:
        maxDice = dice / l
        print('max dice:', maxDice, model_path)
    doc.close()


'''
tips: predict test dataset in actual dataset.
'''


def txt_predict():
    # args = get_args()
    root_path = 'data/th50_nonoise_world1_mask/predict/'
    model_path = root_path + 'CP_epoch60.pth'
    # root_path = args.input
    out_path = root_path + 'traver_predict/'

    net = UNet(n_channels=5, n_classes=1)

    # logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    logging.info("Model loaded !")
    length = len(listdir((root_path + 'elevation/')))
    doc = open(out_path + 'log.txt', 'w')
    dice = 0
    mpa = 0
    for fn in listdir(root_path + 'elevation/'):
        idx = splitext(fn)[0]

        mask = mars_prediect(net=net, root_path=root_path, idx=fn, scale_factor=1.0, device=device, out_threshold=0.5)

        result = mask_to_image(mask)

        result.save(out_path + idx + '.jpg')
        pred = torch.from_numpy(mask + 0).unsqueeze(0).unsqueeze(0)

        label_img_file = root_path + 'traversability/' + fn
        label = MarsDataset.channel_loader(label_img_file).int().unsqueeze(0)
        coeff, m = DiceAndMPA(pred, label)
        # mpa += MPA(pred,label)
        dice += coeff
        mpa += m
        print('i = ', idx, file=doc)
        print(coeff, file=doc)
        delta = (label - pred).numpy()
        print(np.sum(delta == -1), np.sum(delta == 1), np.sum(delta == 0), file=doc)
    print('aver dice:', dice / length, file=doc)
    print('MPA:', mpa / length, file=doc)
    doc.close()


def FindBestPth():
    root_path = 'data/data/predict/'
    list_path = root_path + 'model_list/'
    # root_path = args.input
    out_path = root_path + 'traver_predict/'
    length = 40
    for i in range(1, length):
        model_name = 'CP_epoch' + str(i) + '.pth'
        model_path = list_path + model_name
        img_predict(root_path=root_path, model_path=model_path, out_path=out_path)


if __name__ == "__main__":
    # txt_predict()
    root_path = 'data/data/predict/'
    model_path = root_path + 'CP_epoch50.pth'
    out_path = root_path + 'traver_predict/'
    img_predict(root_path=root_path, model_path=model_path, out_path=out_path)
