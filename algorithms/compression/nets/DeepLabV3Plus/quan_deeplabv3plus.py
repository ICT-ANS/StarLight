import sys

sys.path.append('..')
import network

import argparse
import sys
import logging
from utils import ext_transforms as et
from torch.utils import data
from datasets import Cityscapes
import utils
from metrics import StreamSegMetrics

from utils import *
from lib.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT


parser = argparse.ArgumentParser(description='DeepLabV3Plus for Cityscapes')
parser.add_argument('--model', default='deeplabv3plus_resnet101', type=str, help='model name')
parser.add_argument("--separable_conv", action='store_true', default=False,
                    help="apply separable conv to decoder and aspp")
parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

parser.add_argument('--data_root', default='/home/lushun/dataset/cityscapes', type=str, help='dataset path')
parser.add_argument('--index_start', default=0, type=int)
parser.add_argument('--index_step', default=0, type=int)
parser.add_argument('--index_split', default=5, type=int)
parser.add_argument('--zoom_factor', default=8, type=int)
parser.add_argument('--gpu_id', default=0, type=int)
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

parser.add_argument('--pruner', default='fpgm', type=str, help='pruner: agp|taylor|fpgm')
parser.add_argument('--sparsity', default=0.5, type=float, metavar='LR', help='prune sparsity')
parser.add_argument('--finetune_epochs', default=100, type=int, metavar='N', help='number of epochs for exported model')
parser.add_argument('--finetune_lr', default=0.01, type=float, metavar='N', help='initial finetune learning rate')

parser.add_argument('--batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--val_batch_size', default=4, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--base_size', default=2048, type=int, )
parser.add_argument("--crop_size", type=int, default=513)
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

parser.add_argument('--quan_mode', default='fp16', help='fp16 int8 best', type=str)
parser.add_argument('--baseline', action='store_true', default=False, help='evaluate model on validation set')
parser.add_argument('--multiprocessing_distributed', action='store_true', default=False)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--ngpus_per_node', type=int, default=1)
parser.add_argument('--prune_eval_path', default='', type=str, metavar='PATH', help='path to eval pruned model')

args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

best_prec1 = 0
# args.inputs_shape = (args.batch_size, 3, 513, 513)
args.inputs_shape = (args.batch_size, 3, 1024, 2048)

args.save_dir += 'quan_%s' % args.model
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)


def main():
    global args, best_prec1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True
    cudnn.deterministic = True

    model = get_model(args.model).cuda()

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    # model = nn.DataParallel(model)
    model.to(args.device)
    print("Model restored from %s" % args.ckpt)
    del checkpoint  # free memory

    model.eval()

    train_dst, val_dst = get_dataset(args)
    train_loader = data.DataLoader(
        train_dst, batch_size=args.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=args.val_batch_size, shuffle=True, num_workers=8)
    logging.info("Dataset: %s, Train set: %d, Val set: %d" %
                 ('Cityscapes', len(train_dst), len(val_dst)))

    logging.info('Before Quantization !')
    # Set up metrics
    metrics = StreamSegMetrics(args.num_classes)
    # val_score, ret_samples = validate(
    #     model=model, loader=val_loader, device=args.device, metrics=metrics, is_trt=False)
    # print(metrics.to_str(val_score))
    print('\n')

    if 'sparse' in args.model:
        onnx_path = os.path.join(args.save_dir, '{}_s{}_{}.onnx'.format(args.model, args.sparsity, args.quan_mode))
        trt_path = os.path.join(args.save_dir, '{}_s{}_{}.trt'.format(args.model, args.sparsity, args.quan_mode))
        cache_path = os.path.join(args.save_dir, '{}_s{}_{}.cache'.format(args.model, args.sparsity, args.quan_mode))
    else:
        onnx_path = os.path.join(args.save_dir, '{}_{}.onnx'.format(args.model, args.quan_mode))
        trt_path = os.path.join(args.save_dir, '{}_{}.trt'.format(args.model, args.quan_mode))
        cache_path = os.path.join(args.save_dir, '{}_{}.cache'.format(args.model, args.quan_mode))

    if args.quan_mode == "int8":
        extra_layer_bit = 8
    elif args.quan_mode == "fp16":
        extra_layer_bit = 16
    elif args.quan_mode == "best":
        extra_layer_bit = -1
    else:
        raise NotImplementedError

    engine = ModelSpeedupTensorRT(
        model,
        args.inputs_shape,
        config=None,
        calib_data_loader=val_loader,
        batchsize=args.inputs_shape[0],  # error
        onnx_path=onnx_path,
        calibration_cache=cache_path,
        extra_layer_bit=extra_layer_bit,
    )
    if not os.path.exists(trt_path):
        engine.compress()
        engine.export_quantized_model(trt_path)
    else:
        engine.load_quantized_model(trt_path)
        # engine = common.load_engine(trt_path)
        logging.info('Directly load quantized model from %s' % trt_path)

    logging.info('After Quantization !')
    val_score, ret_samples = validate(
        model=engine, loader=val_loader, device=args.device, metrics=metrics, is_trt=True)
    print(metrics.to_str(val_score))


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
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

    train_dst = Cityscapes(root=opts.data_root,
                           split='train', transform=train_transform)
    val_dst = Cityscapes(root=opts.data_root,
                         split='val', transform=val_transform)
    return train_dst, val_dst


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
    metrics.reset()
    ret_samples = []
    if not is_trt:
        model.eval()

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            # print('images:', images.shape, 'labels:', labels.shape)
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

            if i % 10 == 0:
                logging.info('Step: %d/%d' % (i+1, len(loader)))

        score = metrics.get_results()
    return score, ret_samples


if __name__ == "__main__":
    main()
