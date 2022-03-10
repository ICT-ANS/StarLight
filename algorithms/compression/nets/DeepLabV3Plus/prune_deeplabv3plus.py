import sys

import network

sys.path.append('.')
sys.path.append('..')
import torch.nn as nn
import argparse
import sys
import logging
import utils
import glob
import copy
from utils import *
from lib.compression.pytorch import ModelSpeedup
from torch.utils import data
from utils import ext_transforms as et
from datasets import Cityscapes
from lib.algorithms.pytorch.pruning import FPGMPruner
from metrics import StreamSegMetrics
from export_utils_deeplabv3plus import get_pruned_model

parser = argparse.ArgumentParser(description='DeepLabV3Plus for Cityscapes')
parser.add_argument('--model', default='deeplabv3plus_resnet101', type=str, help='model name')
parser.add_argument("--separable_conv", action='store_true', default=False,
                    help="apply separable conv to decoder and aspp")
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

parser.add_argument('--dataset', default='cityscapes', type=str, help='dataset name')
parser.add_argument('--data_root', default='/home/lushun/dataset/cityscapes', type=str, help='dataset path')
parser.add_argument('--index_start', default=0, type=int)
parser.add_argument('--index_step', default=0, type=int)
parser.add_argument('--index_split', default=5, type=int)
parser.add_argument('--zoom_factor', default=8, type=int)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--workers', default=32, type=int, metavar='N', help='number of data loading workers')
# Prune options
parser.add_argument('--pruner', default='fpgm', type=str, help='pruner: agp|taylor|fpgm')
parser.add_argument('--sparsity', default=0.5, type=float, metavar='LR', help='prune sparsity')
parser.add_argument('--finetune_epochs', default=100, type=int, metavar='N', help='number of epochs for exported model')
parser.add_argument('--finetune_lr', default=0.01, type=float, metavar='N', help='initial finetune learning rate')
# Train options
parser.add_argument("--total_itrs", type=int, default=30e3,
                    help="epoch number (default: 30k)")
parser.add_argument("--val_interval", type=int, default=100,
                    help="step interval for eval (default: 100)")
parser.add_argument('--batch_size', default=16, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--val_batch_size', default=8, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--base_size', default=2048, type=int, )
parser.add_argument("--crop_size", type=int, default=513)
parser.add_argument("--loss_type", type=str, default='cross_entropy',
                    choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                    help="learning rate scheduler policy")
parser.add_argument('--prune_lr', default=0.01, type=float, metavar='LR', help='initial prune learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--power', default=0.9, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--tra_sample_rate', default=0.001, type=float, )
parser.add_argument('--iteration', default=0, type=int, help='weight decay (default: 5e-4)')
parser.add_argument('--max_iter', default=10e10, type=int, help='weight decay (default: 5e-4)')
parser.add_argument('--epoch_num', default=100, type=int, help='weight decay (default: 5e-4)')
parser.add_argument('--num_classes', default=19, type=int)
parser.add_argument('--ignore_label', default=255, type=int)

parser.add_argument('--print_freq', default=50, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--ckpt', default='./checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar',
                    type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', default='./exp_log/', help='The directory used to save the trained models', type=str)
parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=5)
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--baseline', action='store_true', default=False, help='evaluate model on validation set')
parser.add_argument('--multiprocessing_distributed', action='store_true', default=False)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--ngpus_per_node', type=int, default=1)
parser.add_argument('--prune_eval_path', default='', type=str, metavar='PATH', help='path to eval pruned model')

args = parser.parse_args()
args.device = torch.device('cuda:%d' % args.gpu_id) if torch.cuda.is_available() else torch.device('cpu')

best_prec1 = 0
# args.inputs_shape = (1, 3, 1024, 2048)
args.inputs_shape = (1, 3, 513, 513)
args.save_dir = os.path.join(
    args.save_dir, 'prune_%s_%s_s%s_ft%s' % (args.model, args.pruner, args.sparsity, args.finetune_lr)
)
# if os.path.exists(args.save_dir):
#     raise ValueError('Repeated save dir !!!')

# Setup random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

create_exp_dir(args.save_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')


def compute_Params_FLOPs(model, device):
    from thop import profile
    model.eval()
    inputs = torch.rand(size=(1, 3, 1024, 2048)).to(device)
    flops, params = profile(model, inputs=(inputs,), verbose=False)
    logging.info("Thop: params = %.3f M, flops = %.3f G" % (params / 1e6, flops / 1e9))
    logging.info("sum(params): params = %.3f M" % (sum(_param.numel() for _param in model.parameters()) / 1e6))


def get_dataset():
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtRandomCrop(size=(args.crop_size, args.crop_size)),
        et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    train_dst = Cityscapes(root=args.data_root,
                           split='train', transform=train_transform)
    val_dst = Cityscapes(root=args.data_root,
                         split='val', transform=val_transform)
    return train_dst, val_dst


def main():
    train_dst, val_dst = get_dataset()
    train_loader = data.DataLoader(
        train_dst, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=args.val_batch_size, shuffle=True, num_workers=args.workers)
    logging.info("Dataset: %s, Train set: %d, Val set: %d" %
                 ('Cityscapes', len(train_dst), len(val_dst)))

    model = get_model(args.model).to(args.device)
    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    # model = nn.DataParallel(model)
    model.to(args.device)
    logging.info("Model restored from %s" % args.ckpt)
    del checkpoint  # free memory
    logging.info(
        "[Before Pruning] sum Param: %.3f" % (float(sum(_param.numel() for _param in model.parameters())) / 1e6))
    compute_Params_FLOPs(model, device=args.device)

    # Set up metrics
    metrics = StreamSegMetrics(args.num_classes)

    prune_optimizer = torch.optim.SGD(
        model.parameters(), lr=args.prune_lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # # evaluate before pruning
    # val_score, ret_samples = validate(
    #     model=model, loader=val_loader, device=args.device, metrics=metrics, is_trt=False)
    # logging.info(metrics.to_str(val_score))

    model_weights_with_mask = os.path.join(args.save_dir, '%s_with_mask.pth' % args.model)
    model_mask = os.path.join(args.save_dir, '%s_mask.pth' % args.model)
    if args.pruner == 'fpgm':
        config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d'],
                        'op_names': ['backbone.conv1', 'backbone.conv1', 'backbone.layer1.0.conv1',
                                     'backbone.layer1.0.conv2',
                                     'backbone.layer1.0.conv3', 'backbone.layer1.0.downsample.0',
                                     'backbone.layer1.1.conv1', 'backbone.layer1.1.conv2',
                                     'backbone.layer1.1.conv3', 'backbone.layer1.2.conv1',
                                     'backbone.layer1.2.conv2', 'backbone.layer1.2.conv3',
                                     'backbone.layer2.0.conv1', 'backbone.layer2.0.conv2',
                                     'backbone.layer2.0.conv3', 'backbone.layer2.0.downsample.0',
                                     'backbone.layer2.1.conv1', 'backbone.layer2.1.conv2',
                                     'backbone.layer2.1.conv3', 'backbone.layer2.2.conv1',
                                     'backbone.layer2.2.conv2', 'backbone.layer2.2.conv3',
                                     'backbone.layer2.3.conv1', 'backbone.layer2.3.conv2',
                                     'backbone.layer2.3.conv3', 'backbone.layer3.0.conv1',
                                     'backbone.layer3.0.conv2', 'backbone.layer3.0.conv3',
                                     'backbone.layer3.0.downsample.0', 'backbone.layer3.1.conv1',
                                     'backbone.layer3.1.conv2', 'backbone.layer3.1.conv3',
                                     'backbone.layer3.2.conv1', 'backbone.layer3.2.conv2',
                                     'backbone.layer3.2.conv3', 'backbone.layer3.3.conv1',
                                     'backbone.layer3.3.conv2', 'backbone.layer3.3.conv3',
                                     'backbone.layer3.4.conv1', 'backbone.layer3.4.conv2',
                                     'backbone.layer3.4.conv3', 'backbone.layer3.5.conv1',
                                     'backbone.layer3.5.conv2', 'backbone.layer3.5.conv3',
                                     'backbone.layer3.6.conv1', 'backbone.layer3.6.conv2',
                                     'backbone.layer3.6.conv3', 'backbone.layer3.7.conv1',
                                     'backbone.layer3.7.conv2', 'backbone.layer3.7.conv3',
                                     'backbone.layer3.8.conv1', 'backbone.layer3.8.conv2',
                                     'backbone.layer3.8.conv3', 'backbone.layer3.9.conv1',
                                     'backbone.layer3.9.conv2', 'backbone.layer3.9.conv3',
                                     'backbone.layer3.10.conv1', 'backbone.layer3.10.conv2',
                                     'backbone.layer3.10.conv3', 'backbone.layer3.11.conv1',
                                     'backbone.layer3.11.conv2', 'backbone.layer3.11.conv3',
                                     'backbone.layer3.12.conv1', 'backbone.layer3.12.conv2',
                                     'backbone.layer3.12.conv3', 'backbone.layer3.13.conv1',
                                     'backbone.layer3.13.conv2', 'backbone.layer3.13.conv3',
                                     'backbone.layer3.14.conv1', 'backbone.layer3.14.conv2',
                                     'backbone.layer3.14.conv3', 'backbone.layer3.15.conv1',
                                     'backbone.layer3.15.conv2', 'backbone.layer3.15.conv3',
                                     'backbone.layer3.16.conv1', 'backbone.layer3.16.conv2',
                                     'backbone.layer3.16.conv3', 'backbone.layer3.17.conv1',
                                     'backbone.layer3.17.conv2', 'backbone.layer3.17.conv3',
                                     'backbone.layer3.18.conv1', 'backbone.layer3.18.conv2',
                                     'backbone.layer3.18.conv3', 'backbone.layer3.19.conv1',
                                     'backbone.layer3.19.conv2', 'backbone.layer3.19.conv3',
                                     'backbone.layer3.20.conv1', 'backbone.layer3.20.conv2',
                                     'backbone.layer3.20.conv3', 'backbone.layer3.21.conv1',
                                     'backbone.layer3.21.conv2', 'backbone.layer3.21.conv3',
                                     'backbone.layer3.22.conv1', 'backbone.layer3.22.conv2',
                                     'backbone.layer3.22.conv3', 'backbone.layer4.0.conv1',
                                     'backbone.layer4.0.conv2', 'backbone.layer4.0.conv3',
                                     'backbone.layer4.0.downsample.0', 'backbone.layer4.1.conv1',
                                     'backbone.layer4.1.conv2', 'backbone.layer4.1.conv3',
                                     'backbone.layer4.2.conv1', 'backbone.layer4.2.conv2',
                                     'backbone.layer4.2.conv3', ]
                        }]
        pruner = FPGMPruner(
            model,
            config_list,
            prune_optimizer,
            dummy_input=torch.rand(size=args.inputs_shape).to(args.device),
        )
    else:
        raise NotImplementedError

    if not os.path.exists(model_mask):
        logging.info('Start to generate mask from scratch ...')
        pruner.compress()
        # Save masked pruned model (model + mask)
        pruner.export_model(model_weights_with_mask, model_mask)

    model = get_pruned_model(args, pruner, model, model_mask)

    logging.info(
        "[After Pruning] sum Param: %.3f" % (float(sum(_param.numel() for _param in model.parameters())) / 1e6))
    compute_Params_FLOPs(model, device=args.device)

    # # evaluate after pruning
    # val_score, ret_samples = validate(
    #     model=model, loader=val_loader, device=args.device, metrics=metrics, is_trt=False)
    # logging.info(metrics.to_str(val_score))

    # Finetune pruned model
    logging.info('Finetuning exported pruned model...')

    # Set up criterion
    # criterion = utils.get_loss(args.loss_type)
    if args.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif args.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    else:
        raise NotImplementedError

    # Set up finetune optimizer
    finetune_optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * args.finetune_lr},
        {'params': model.classifier.parameters(), 'lr': args.finetune_lr},
    ], lr=args.finetune_lr, momentum=0.9, weight_decay=args.weight_decay)
    if args.lr_policy == 'poly':
        scheduler = utils.PolyLR(finetune_optimizer, args.total_itrs, power=0.9)
    elif args.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(finetune_optimizer, step_size=args.step_size, gamma=0.1)
    else:
        raise NotImplementedError

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": finetune_optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        logging.info("Model saved as %s" % path)

    interval_loss = 0
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    while True:  # cur_itrs < args.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(args.device, dtype=torch.float32)
            labels = labels.to(args.device, dtype=torch.long)

            finetune_optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            finetune_optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if cur_itrs == 1 or cur_itrs % 10 == 0:
                interval_loss = interval_loss / 10
                logging.info("Epoch %d, Itrs %d/%d, Loss=%f" %
                             (cur_epochs, cur_itrs, args.total_itrs, interval_loss))
                interval_loss = 0.0

            if cur_itrs % args.val_interval == 0:
                print('\n')
                save_ckpt(os.path.join(args.save_dir, 'latest_%s_%s_os%d.pth' %
                                       (args.model, args.dataset, args.output_stride)))
                logging.info("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    model=model, loader=val_loader, device=args.device, metrics=metrics, is_trt=False)
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt(os.path.join(args.save_dir, 'best_%s_%s_os%d.pth' %
                                           (args.model, args.dataset, args.output_stride)))
                logging.info(metrics.to_str(val_score))
                logging.info('Best_mIoU: %.6f' % best_score)
                model.train()
                print('\n')
            scheduler.step()

            if cur_itrs >= args.total_itrs:
                return


def get_model(model_name):
    if model_name == 'deeplabv3plus_resnet101':
        # Set up model (all models are 'constructed at network.modeling)
        model = network.modeling.__dict__[args.model](num_classes=args.num_classes, output_stride=args.output_stride)
        if args.separable_conv and 'plus' in args.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
    else:
        raise NotImplementedError('Not Support {}'.format(model_name))
    return model


def validate(model, loader, device, metrics, is_trt=False):
    """Do validation and return specified samples"""
    logging.info('Do validation.')
    metrics.reset()
    ret_samples = []
    if not is_trt:
        model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            # logging.info('images:', images.shape, 'labels:', labels.shape)
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            if is_trt:
                outputs, trt_infer_time = model.inference(images)
                outputs = torch.tensor(outputs)
                outputs = outputs.reshape(4, 19, 1024, 2048)
            else:
                outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

            if i % 20 == 0 or i == (len(loader) - 1):
                logging.info('Step: %d/%d' % (i + 1, len(loader)))

        score = metrics.get_results()
    return score, ret_samples


if __name__ == "__main__":
    main()
