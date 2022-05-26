import argparse
import logging
import os
import sys

import yaml
import copy
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval import eval_net
from unet import UNet
from torch.utils.tensorboard import SummaryWriter

from utils.mars_dataset import MarsDataset
from utils.mars_img_dataset import MarsImgDataset
from torch.utils.data import DataLoader, random_split, Subset

from infer import img_predict, count_flops_params
from lib.compression.pytorch import ModelSpeedup
from lib.algorithms.pytorch.pruning import (TaylorFOWeightFilterPruner, FPGMPruner, AGPPruner)

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

dir_checkpoint = 'virtual_checkpoints/'
root_dir = 'Pytorch-UNet-master/data/data/train/'
mark_name = 'data'


def train_net(net, device, epochs, train_loader, val_loader, optimizer, scheduler, data_path, save_cp=True, n_train=0, batch_size=0, writer=None):
    # tips: here change MarsImgDataset or MarsDataset to select img input or txt input.

    max_dice = 0
    max_dice_net = None
    global_step = 0
    # img_predict(net, device)
    for epoch in range(epochs):
        net.train()
        net.set_pad(100)

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='txt') as pbar:
            for batch in train_loader:
                # if global_step % 100 == 0:
                #     img_predict(net, device)
                imgs = batch['image']
                true_masks = batch['label']

                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                # writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                # if global_step % (n_train // (10 * batch_size)) == 0:
                #     for tag, value in net.named_parameters():
                #         tag = tag.replace('.', '/')
                #         writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                #         # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                #     val_score, mpa = eval_net(net, val_loader, device)
                #     scheduler.step(val_score)
                #     writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                #     if net.n_classes > 1:
                #         logging.info('Validation cross entropy: {}'.format(val_score))
                #         writer.add_scalar('Loss/test', val_score, global_step)
                #     else:
                #         logging.info('Validation Dice Coeff: {}'.format(val_score))
                #         logging.info('Validation mpa: {}'.format(mpa))
                #         writer.add_scalar('Dice/test', val_score, global_step)
                #         writer.add_scalar('mpa/test', mpa, global_step)

                #     # writer.add_images('images', imgs, global_step)
                #     if net.n_classes == 1:
                #         writer.add_images('masks/true', true_masks, global_step)
                #         writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
            epoch_loss /= len(train_loader)
            # writer.add_scalar('Loss epoch/train', epoch_loss, epoch)
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if epoch % 1 == 0:
                torch.save(net.state_dict(), dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

        net.set_pad(150)
        dice, _ = img_predict(net, device, root_path=os.path.join(args.data_path, 'predict/'))
        if dice > max_dice:
            max_dice = dice
            del max_dice_net
            max_dice_net = copy.deepcopy(net)

    # writer.close()
    return max_dice_net


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batchsize')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-8, help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default="checkpoints/origin.pth", help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--sparsity', default=0.5, type=float, metavar='LR', help='prune sparsity')
    parser.add_argument('--save_dir', type=str, default='logs/prune/test', help='model name')
    parser.add_argument('--data_path', type=str, default="", help='dataset path')

    parser.add_argument('--write_yaml', action='store_true', default=False, help='write yaml file')
    parser.add_argument('--no_write_yaml_after_prune', action='store_true', default=False, help='')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    global dir_checkpoint
    dir_checkpoint = os.path.join(args.save_dir, 'checkpoint')
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)

    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    name = torch.cuda.get_device_name(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    logging.info(f'Network:\n' f'\t{net.n_channels} input channels\n' f'\t{net.n_classes} output channels (classes)\n' f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device), strict=False)
        logging.info(f'Model loaded from {args.load}')

    # origin
    net.set_pad(150)
    logging.info(f'Origin Net:')
    img_predict(net, device, root_path=os.path.join(args.data_path, 'predict/'))
    count_flops_params(net, device)

    if args.write_yaml:
        flops, params = count_flops_params(net, device)
        mpa, infer_time = img_predict(net, device, root_path=os.path.join(args.data_path, 'predict/'))
        storage = os.path.getsize(args.load)
        with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
            yaml_data = {
                'MPA': {'baseline': round(mpa*100, 2), 'method': None},
                'FLOPs': {'baseline': round(flops/1e6, 2), 'method': None},
                'Parameters': {'baseline': round(params/1e3, 2), 'method': None},
                'Infer_times': {'baseline': round(infer_time*1e3, 2), 'method': None},
                'Storage': {'baseline': round(storage/1e6, 2), 'method': None},
            }
            yaml.dump(yaml_data, f)

    # dataloader
    dataset = MarsImgDataset(os.path.join(args.data_path, 'train/'), args.scale)
    n_val = int(len(dataset) * args.val / 100)
    n_train = len(dataset) - n_val
    random = False
    if random:
        train, val = random_split(dataset, [n_train, n_val])
    else:
        train = Subset(dataset, list(range(n_train)))
        val = Subset(dataset, list(range(n_train, len(dataset))))
    train_loader = DataLoader(train, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    # optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    '''tips: here change patience to control learning rate decay.'''
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=20, factor=0.9, eps=1e-11)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for name, module in net.named_modules():
        print(name)

    # prune
    op_names = [
        'inc.double_conv.0',
        'down1.maxpool_conv.1.double_conv.0',
        'down2.maxpool_conv.1.double_conv.0',
        'down3.maxpool_conv.1.double_conv.0',
        # 'down4.maxpool_conv.1.double_conv.0',
        # 'up1.conv.double_conv.0',
        'up2.conv.double_conv.0',
        'up3.conv.double_conv.0',
        'up4.conv.double_conv.0',
    ]
    config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d'], 'op_names': op_names}]
    pruner = FPGMPruner(
        net,
        config_list,
        optimizer,
        dummy_input=torch.randn(2, 3, 150, 150).to(device),
    )
    pruner.compress()
    logging.info("\nInfer after pruning:")

    # save masked pruned model (model + mask)
    pruner.export_model(os.path.join(args.save_dir, 'model_masked.pth'), os.path.join(args.save_dir, 'mask.pth'))

    # export pruned model
    logging.info('Speed up masked model...')
    net = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    net.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_masked.pth')))
    masks_file = os.path.join(args.save_dir, 'mask.pth')

    net.set_pad(150)
    m_speedup = ModelSpeedup(net, torch.randn(2, 3, 150, 150).to(device), masks_file, device)
    m_speedup.speedup_model()

    count_flops_params(net, device)
    img_predict(net, device, root_path=os.path.join(args.data_path, 'predict/'))

    # optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9)
    '''tips: here change patience to control learning rate decay.'''
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=20, factor=0.9)

    logging.info(f'''Starting finetune:
        Epochs:          {args.epochs}
        Batch size:      {args.batchsize}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {True}
        Device:          {device.type}
        Images scaling:  {args.scale}
    ''')
    # writer = SummaryWriter(comment=f'LR_{args.lr}_BS_{args.batchsize}_ep_{args.epochs}_cl_{net.n_channels}_data_{n_train}_{mark_name}')
    net.set_pad(100)
    net = train_net(net, device, args.epochs, train_loader, val_loader, optimizer, scheduler, args.data_path, save_cp=False, n_train=n_train, batch_size=args.batchsize, writer=None)

    logging.info("\nInfer after finetuning the export-prune model:")
    net.set_pad(150)
    count_flops_params(net, device)
    img_predict(net, device, root_path=os.path.join(args.data_path, 'predict/'))

    torch.save({'state_dict': net.state_dict()}, os.path.join(args.save_dir, 'export_pruned_model.pth'))

    if args.write_yaml and not args.no_write_yaml_after_prune:
        flops, params = count_flops_params(net, device)
        mpa, infer_time = img_predict(net, device, root_path=os.path.join(args.data_path, 'predict/'))
        storage = os.path.getsize(os.path.join(args.save_dir, 'export_pruned_model.pth'))
        with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
            yaml_data = {
                'MPA': {'baseline': yaml_data['MPA']['baseline'], 'method': round(mpa*100, 2)},
                'FLOPs': {'baseline': yaml_data['FLOPs']['baseline'], 'method': round(flops/1e6, 2)},
                'Parameters': {'baseline': yaml_data['Parameters']['baseline'], 'method': round(params/1e3, 2)},
                'Infer_times': {'baseline': yaml_data['Infer_times']['baseline'], 'method': round(infer_time*1e3, 2)},
                'Storage': {'baseline': yaml_data['Storage']['baseline'], 'method': round(storage/1e6, 2)},
                'Output_file': os.path.join(args.save_dir, 'export_pruned_model.pth'),
            }
            yaml.dump(yaml_data, f)
        torch.save(net, \
            os.path.join(
                args.save_dir, \
                '../../..', \
                'model_vis/KeTi3Dataset-UNet', \
                'online-fpgm.pth'))
