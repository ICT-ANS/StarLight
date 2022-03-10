import sys

sys.path.append('')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import argparse
import sys
import copy
import logging
from dataReader.dataset import Reader
from torch.utils.data import DataLoader
from torch.autograd import Variable

import glob
from utils.utils import *
from lib.compression.pytorch.utils.counter import count_flops_params

from lib.compression.pytorch import ModelSpeedup
from lib.algorithms.pytorch.pruning import (TaylorFOWeightFilterPruner, FPGMPruner, AGPPruner)
from export_utils_seg_deeplab import get_pruned_model
from eval_mars import evaluate

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--data_root', default='./dataset/Mars_Seg_1119/Data', type=str, help='dataset path')
parser.add_argument('--model', default='seg_deeplab_efficientnetb3', type=str, help='model name')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')

parser.add_argument('--pruner', default='agp', type=str, help='pruner: agp|taylor|fpgm')
parser.add_argument('--sparsity', default=0.5, type=float, metavar='LR', help='prune sparsity')
parser.add_argument('--finetune_epochs', default=100, type=int, metavar='N', help='number of epochs for exported model')
parser.add_argument('--finetune_lr', default=2e-4, type=float, metavar='N', help='initial finetune learning rate')
parser.add_argument('--finetune_momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--finetune_weight_decay', default=1e-4, type=float, metavar='W',
                    help='weight decay (default: 5e-4)')

parser.add_argument('--batch_size', default=6, type=int, metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--prune_lr', default=2e-5, type=float, metavar='LR', help='initial prune learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--power', default=0.9, type=float, help='weight decay (default: 5e-4)')
parser.add_argument('--train_sample_rate', default=0.001, type=float, )
parser.add_argument('--iteration', default=0, type=int, help='weight decay (default: 5e-4)')
parser.add_argument('--max_iter', default=10e10, type=int, help='weight decay (default: 5e-4)')
parser.add_argument('--epoch_num', default=100, type=int, help='weight decay (default: 5e-4)')
parser.add_argument('--num_classes', default=8, type=int, help='weight decay (default: 5e-4)')

parser.add_argument('--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='./checkpoint/resnet50.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_dir', default='./exp_log/', help='The directory used to save the trained models', type=str)
parser.add_argument('--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=5)
parser.add_argument('--seed', type=int, default=2, help='random seed')

parser.add_argument('--baseline', action='store_true', default=False, help='evaluate model on validation set')
parser.add_argument('--debug', default=True, help='debug mode')
parser.add_argument('--prune_eval_path', default='', type=str, metavar='PATH', help='path to eval pruned model')

args = parser.parse_args()
args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

best_prec1 = 0
args.inputs_shape = (1, 3, 512, 512)
args.save_dir = os.path.join(
    args.save_dir, '%s_%s_%s_p%s_f%s' % (args.model, args.pruner, args.sparsity, args.prune_lr, args.finetune_lr)
)
if os.path.exists(args.save_dir) and not args.debug:
    raise ValueError('Repeated save dir !!!')

create_exp_dir(args.save_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


# logging.info(args)


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion > max_iter:
        return init_lr

    lr = init_lr * (1 - iteraion / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def compute_Params_FLOPs(model, device):
    from thop import profile
    model.eval()
    inputs = torch.rand(size=(2, 3, 512, 512)).to(device)
    flops, params = profile(model, inputs=(inputs,), verbose=False)
    logging.info("Thop: params = {}M, flops = {}M".format(params / 1e6, flops / 1e6))
    logging.info("sum(params): params = {}M".format(sum(_param.numel() for _param in model.parameters()) / 1e6))


def main():
    train_loader, val_loader = get_data_loader(args)
    args.max_iter = len(train_loader) * args.epoch_num
    logging.info(args)

    model = get_model(args.model).to(args.device)
    logging.info('Parameters number is: %f' % sum(param.numel() for param in model.parameters()))
    compute_Params_FLOPs(copy.deepcopy(model), device=args.device)
    print('\n')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)
    prune_optimizer = optim.Adam(model.parameters(), lr=args.prune_lr,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
    # evaluate
    miou, avg_time = evaluate(args, model)
    logging.info("Before pruning: [Test MIoU = %.4f] [Avg eval time = %.4f s]" % (miou, avg_time))
    print('\n')

    # val_loss, val_acc, val_acc_cls, val_miou, val_fwavacc = validate(model, val_loader, criterion)
    # flops, params, _ = count_flops_params(model, args.inputs_shape, verbose=False)
    # logging.info(
    #     "Before pruning\t val_loss:%.4f\t val_acc:%.4f \tval_acc_cls:%.4f \tval_miou:%.4f \tval_fwavacc:%.4f"
    #     % (val_loss, val_acc, val_acc_cls, val_miou, val_fwavacc)
    # )

    # only eval pruned model
    if args.prune_eval_path:
        model = get_model(args.model).to(args.device)
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_masked.pth')))
        masks_file = os.path.join(args.save_dir, 'mask.pth')
        m_speedup = ModelSpeedup(model, torch.rand(size=args.inputs_shape).to(args.device), masks_file, args.device)
        m_speedup.speedup_model()
        model.load_state_dict(args.prune_eval_path)

        flops, params, _ = count_flops_params(model, args.inputs_shape, verbose=False)
        loss, top1, pre_time, infer_time, post_time = validate(model, val_loader, criterion)
        logging.info(
            "Evaluation result : [Top1: {:.2f}%][FLops: {:.5f}M][Params: {:.5f}M][Infer: {:.2f}ms]\n".format(top1,
                                                                                                             flops / 1e6,
                                                                                                             params / 1e6,
                                                                                                             infer_time * 1000))
        exit(0)

    def trainer(model, optimizer, criterion, epoch):
        result = train(train_loader, model, criterion, optimizer, epoch, args.prune_lr, args)
        return result

    # get pruner, agp|taylor|fpgm
    if args.pruner == 'agp':
        config_list = [{'sparsity': args.sparsity,
                        'op_types': ['Conv2d'],
                        'op_names':
                            ['efficientnet_features._conv_stem.conv',
                             'efficientnet_features._blocks.0._depthwise_conv.conv',
                             'efficientnet_features._blocks.0._se_reduce.conv',
                             'efficientnet_features._blocks.0._project_conv.conv',
                             'efficientnet_features._blocks.1._depthwise_conv.conv',
                             'efficientnet_features._blocks.1._se_reduce.conv',
                             'efficientnet_features._blocks.1._project_conv.conv',
                             'efficientnet_features._blocks.2._expand_conv.conv',
                             'efficientnet_features._blocks.2._depthwise_conv.conv',
                             'efficientnet_features._blocks.2._se_reduce.conv',
                             'efficientnet_features._blocks.2._project_conv.conv',
                             'efficientnet_features._blocks.3._expand_conv.conv',
                             'efficientnet_features._blocks.3._depthwise_conv.conv',
                             'efficientnet_features._blocks.3._se_reduce.conv',
                             'efficientnet_features._blocks.3._project_conv.conv',
                             'efficientnet_features._blocks.4._expand_conv.conv',
                             'efficientnet_features._blocks.4._depthwise_conv.conv',
                             'efficientnet_features._blocks.4._se_reduce.conv',
                             'efficientnet_features._blocks.4._project_conv.conv',
                             'efficientnet_features._blocks.5._expand_conv.conv',
                             'efficientnet_features._blocks.5._depthwise_conv.conv',
                             'efficientnet_features._blocks.5._se_reduce.conv',
                             'efficientnet_features._blocks.5._project_conv.conv',
                             'efficientnet_features._blocks.6._expand_conv.conv',
                             'efficientnet_features._blocks.6._depthwise_conv.conv',
                             'efficientnet_features._blocks.6._se_reduce.conv',
                             'efficientnet_features._blocks.6._project_conv.conv',
                             'efficientnet_features._blocks.7._expand_conv.conv',
                             'efficientnet_features._blocks.7._depthwise_conv.conv',
                             'efficientnet_features._blocks.7._se_reduce.conv',
                             'efficientnet_features._blocks.7._project_conv.conv',
                             'efficientnet_features._blocks.8._expand_conv.conv',
                             'efficientnet_features._blocks.8._depthwise_conv.conv',
                             'efficientnet_features._blocks.8._se_reduce.conv',
                             'efficientnet_features._blocks.8._project_conv.conv',
                             'efficientnet_features._blocks.9._expand_conv.conv',
                             'efficientnet_features._blocks.9._depthwise_conv.conv',
                             'efficientnet_features._blocks.9._se_reduce.conv',
                             'efficientnet_features._blocks.9._project_conv.conv',
                             'efficientnet_features._blocks.10._expand_conv.conv',
                             'efficientnet_features._blocks.10._depthwise_conv.conv',
                             'efficientnet_features._blocks.10._se_reduce.conv',
                             'efficientnet_features._blocks.10._project_conv.conv',
                             'efficientnet_features._blocks.11._expand_conv.conv',
                             'efficientnet_features._blocks.11._depthwise_conv.conv',
                             'efficientnet_features._blocks.11._se_reduce.conv',
                             'efficientnet_features._blocks.11._project_conv.conv',
                             'efficientnet_features._blocks.12._expand_conv.conv',
                             'efficientnet_features._blocks.12._depthwise_conv.conv',
                             'efficientnet_features._blocks.12._se_reduce.conv',
                             'efficientnet_features._blocks.12._project_conv.conv',
                             'efficientnet_features._blocks.13._expand_conv.conv',
                             'efficientnet_features._blocks.13._depthwise_conv.conv',
                             'efficientnet_features._blocks.13._se_reduce.conv',
                             'efficientnet_features._blocks.13._project_conv.conv',
                             'efficientnet_features._blocks.14._expand_conv.conv',
                             'efficientnet_features._blocks.14._depthwise_conv.conv',
                             'efficientnet_features._blocks.14._se_reduce.conv',
                             'efficientnet_features._blocks.14._project_conv.conv',
                             'efficientnet_features._blocks.15._expand_conv.conv',
                             'efficientnet_features._blocks.15._depthwise_conv.conv',
                             'efficientnet_features._blocks.15._se_reduce.conv',
                             'efficientnet_features._blocks.15._project_conv.conv',
                             'efficientnet_features._blocks.16._expand_conv.conv',
                             'efficientnet_features._blocks.16._depthwise_conv.conv',
                             'efficientnet_features._blocks.16._se_reduce.conv',
                             'efficientnet_features._blocks.16._project_conv.conv',
                             'efficientnet_features._blocks.17._expand_conv.conv',
                             'efficientnet_features._blocks.17._depthwise_conv.conv',
                             'efficientnet_features._blocks.17._se_reduce.conv',
                             'efficientnet_features._blocks.17._project_conv.conv',
                             'efficientnet_features._blocks.18._expand_conv.conv',
                             'efficientnet_features._blocks.18._depthwise_conv.conv',
                             'efficientnet_features._blocks.18._se_reduce.conv',
                             'efficientnet_features._blocks.18._project_conv.conv',
                             'efficientnet_features._blocks.19._expand_conv.conv',
                             'efficientnet_features._blocks.19._depthwise_conv.conv',
                             'efficientnet_features._blocks.19._se_reduce.conv',
                             'efficientnet_features._blocks.19._project_conv.conv',
                             'efficientnet_features._blocks.20._expand_conv.conv',
                             'efficientnet_features._blocks.20._depthwise_conv.conv',
                             'efficientnet_features._blocks.20._se_reduce.conv',
                             'efficientnet_features._blocks.20._project_conv.conv',
                             'efficientnet_features._blocks.21._expand_conv.conv',
                             'efficientnet_features._blocks.21._depthwise_conv.conv',
                             'efficientnet_features._blocks.21._se_reduce.conv',
                             'efficientnet_features._blocks.21._project_conv.conv',
                             'efficientnet_features._blocks.22._expand_conv.conv',
                             'efficientnet_features._blocks.22._depthwise_conv.conv',
                             'efficientnet_features._blocks.22._se_reduce.conv',
                             'efficientnet_features._blocks.22._project_conv.conv',
                             'efficientnet_features._blocks.23._expand_conv.conv',
                             'efficientnet_features._blocks.23._depthwise_conv.conv',
                             'efficientnet_features._blocks.23._se_reduce.conv',
                             'efficientnet_features._blocks.23._project_conv.conv',
                             'efficientnet_features._blocks.24._expand_conv.conv',
                             'efficientnet_features._blocks.24._depthwise_conv.conv',
                             'efficientnet_features._blocks.24._se_reduce.conv',
                             'efficientnet_features._blocks.24._project_conv.conv',
                             'efficientnet_features._blocks.25._expand_conv.conv',
                             'efficientnet_features._blocks.25._depthwise_conv.conv',
                             'efficientnet_features._blocks.25._se_reduce.conv',
                             'efficientnet_features._blocks.25._project_conv.conv',
                             'efficientnet_features._conv_head.conv',
                             'efficientnet_features._conv_head', 'conv1', 'conv2', 'last_conv', 'last_conv.0',
                             'last_conv.1', 'last_conv.2', 'last_conv.3', 'last_conv.4', 'last_conv.5'
                             ]
                        }]
        pruner = AGPPruner(
            model,
            config_list,
            prune_optimizer,
            trainer,
            criterion,
            num_iterations=1,
            epochs_per_iteration=1,
            pruning_algorithm='taylorfo',
        )
    elif args.pruner == 'taylor':
        config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
        pruner = TaylorFOWeightFilterPruner(
            model,
            config_list,
            prune_optimizer,
            trainer,
            criterion,
            sparsifying_training_batches=1,
        )
    elif args.pruner == 'fpgm':
        config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
        pruner = FPGMPruner(
            model,
            config_list,
            prune_optimizer,
            dummy_input=torch.rand(size=args.inputs_shape).to(args.device),
        )
    else:
        raise NotImplementedError

    # pruner.export_model(os.path.join(args.save_dir, 'model_masked.pth'), os.path.join(args.save_dir, 'mask.pth'))
    # import sys
    # sys.exit()

    logging.info('Start pruning ...')
    pruner.compress()

    # save masked pruned model (model + mask)
    pruner.export_model(os.path.join(args.save_dir, 'model_masked.pth'), os.path.join(args.save_dir, 'mask.pth'))
    print('\n')

    # export pruned model
    logging.info('Speed up masked model ...')
    # model = get_model(args.model).to(args.device)
    # model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_masked.pth'), map_location=args.device))
    masks_file = os.path.join(args.save_dir, 'mask.pth')
    mask_pt = torch.load(masks_file, map_location=args.device)

    # m_speedup = ModelSpeedup(model, torch.rand(size=args.inputs_shape).to(args.device), masks_file, args.device)
    # m_speedup.speedup_model()
    model = get_pruned_model(pruner, model, mask_pt)
    compute_Params_FLOPs(copy.deepcopy(model), device=args.device)

    # val_loss, val_acc, val_acc_cls, val_miou, val_fwavacc = validate(model, val_loader, criterion)
    # logging.info(
    #     "After pruning\t val_loss:%.4f\t val_acc:%.4f \tval_acc_cls:%.4f \tval_miou:%.4f \tval_fwavacc:%.4f"
    #     % (val_loss, val_acc, val_acc_cls, val_miou, val_fwavacc)
    # )
    miou, avg_time = evaluate(args, model)
    logging.info("After pruning: [Test MIoU = %.4f] [Avg eval time = %.4f s]" % (miou, avg_time))
    print('\n')

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_speed_up.pth'))

    # finetune pruned model
    logging.info('Finetuning export pruned model...')
    finetune_optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr,
                                    betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)

    best_miou = 0.0
    args.iteration = 0
    for epoch in range(args.finetune_epochs):
        train_loss, train_acc, train_acc_cls, train_miou, train_fwavacc = \
            train(train_loader, model, criterion, finetune_optimizer, epoch, args.finetune_lr, args)
        logging.info(
            "Epoch[%d]\t train_loss:%.4f\t train_acc:%.4f\t train_acc_cls:%.4f\t train_miou:%.4f\t train_fwavacc:%.4f"
            % (epoch, train_loss, train_acc, train_acc_cls, train_miou, train_fwavacc)
        )

        # val_loss, val_acc, val_acc_cls, val_miou, val_fwavacc = validate(model, val_loader, criterion)
        # logging.info(
        #     "Epoch[%d]\t val_loss:%.4f\t val_acc:%.4f\t val_acc_cls:%.4f\t val_miou:%.4f\t val_fwavacc:%.4f"
        #       % (epoch, val_loss, val_acc, val_acc_cls, val_miou, val_fwavacc)
        # )
        test_miou, avg_time = evaluate(args, model)

        if epoch % args.save_every == 0 and epoch > (args.finetune_epochs // 2):
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'miou': test_miou,
            }, os.path.join(args.save_dir, 'model_pruned_epoch%s_miou%f.pth' % (epoch, test_miou)))
        if best_miou < test_miou:
            best_miou = test_miou
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'miou': best_miou,
            }, os.path.join(args.save_dir, 'model_pruned_best_miou.pth'))
        logging.info(
            "[Epoch = %d] [Test MIoU = %.4f] [Best MIoU = %.4f] [Avg eval time = %.4f s]\n" %
            (epoch, test_miou, best_miou, avg_time)
        )


def train(train_loader, model, criterion, optimizer, epoch, init_lr, args):
    model.train()
    losses = AverageMeter()
    gts_all, predictions_all = [], []
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.cuda()
        images = images.cuda()
        images = Variable(images)
        labels = Variable(labels)
        # Decaying Learning Rate
        lr = poly_lr_scheduler(optimizer, init_lr, args.iteration, args.max_iter, args.power)
        # Forward + Backward + Optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        #  record loss
        losses.update(loss.data)
        if random.random() < args.train_sample_rate or i == 0:
            predictions = outputs.data.max(1)[1].cpu().numpy()
            gts_all.append(labels.data.cpu().numpy())
            predictions_all.append(predictions)
        if i % 100 == 0:
            logging.info('[epoch %d],[step %04d/%04d],[iter %05d],[lr %.9f],[train_loss.avg %.5f]'
                         % (epoch, i, len(train_loader), args.iteration, lr, losses.avg))
        args.iteration = args.iteration + 1
    train_acc, train_acc_cls, train_miou, train_fwavacc = get_evaluation_score(predictions_all, gts_all,
                                                                               args.num_classes)
    return losses.avg, train_acc, train_acc_cls, train_miou, train_fwavacc


def validate(model, val_loader, criterion):
    model.eval()
    losses = AverageMeter()
    gts_all, predictions_all = [], []
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
        # record loss
        losses.update(loss.data)
        predictions = output.data.max(1)[1].cpu().numpy()
        gts_all.append(target.data.cpu().numpy())
        predictions_all.append(predictions)
    val_acc, val_acc_cls, val_miou, val_fwavacc = get_evaluation_score(predictions_all, gts_all, args.num_classes)
    return losses.avg, val_acc, val_acc_cls, val_miou, val_fwavacc


def get_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, 200)
    elif model_name == 'resnet101':
        model = models.resnet101()
        model.fc = nn.Linear(model.fc.in_features, 200)
    elif model_name == 'seg_deeplab_efficientnetb3':
        # model define
        from models.EfficientNet.deeplab_efficiennetb1 import DeepLabv3_plus as deeplab_efficiennetb3
        model = deeplab_efficiennetb3(nInputChannels=3, n_classes=args.num_classes, os=16,
                                      pretrained=False, _print=False)
        # load pretrained weights
        # save_point = torch.load('./pretrained/model.pth', map_location=args.device)
        save_point = torch.load('./pretrained/model_1119.pth', map_location=args.device)
        model_param = model.state_dict()
        pretrained_param = save_point['state_dict']
        model_keys = list(model_param.keys())
        pretrained_keys = list(pretrained_param.keys())
        for i, key in enumerate(model_keys):
            # if key == 'efficientnet_features._blocks.0._depthwise_conv.conv.weight':
            #     print(key)
            # if key in state_dict_param.keys():
            #     model_param[key] = state_dict_param[key]
            # else:
            #     new_key = key.split('.')
            #     del new_key[-2]
            #     new_key.insert(0, 'module')
            #     new_key = ".".join(new_key)
            #     assert new_key in state_dict_param.keys()
            #     model_param[key] = state_dict_param[new_key]
            assert model_param[model_keys[i]].shape == pretrained_param[pretrained_keys[i]].shape
            model_param[model_keys[i]] = pretrained_param[pretrained_keys[i]]
        model.load_state_dict(model_param)

        logging.info('Load model weights done!')
        model.eval()
    else:
        raise NotImplementedError('Not Support {}'.format(model_name))
    return model


def get_data_loader(args):
    print("loading dataset ...")
    train_data = Reader(args, mode='train')
    print("Train set samples: ", len(train_data))
    val_data = Reader(args, mode='test')
    print("Validation set samples: ", len(val_data))
    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                              drop_last=True, num_workers=args.workers if torch.cuda.is_available() else 0)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                            drop_last=False, num_workers=args.workers if torch.cuda.is_available() else 0)

    return train_loader, val_loader


if __name__ == "__main__":
    main()
