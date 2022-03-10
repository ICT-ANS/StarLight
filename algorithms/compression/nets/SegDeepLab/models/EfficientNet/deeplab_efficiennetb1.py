import sys

sys.path.append('')
sys.path.append('../../')
sys.path.append('./prune_seg')
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.EfficientNet.efficientnet import EfficientNet


class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=8, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        # Atrous Conv

        # self.efficientnet_features = EfficientNet.from_pretrained('efficientnet-b3',
            # weights_path='./prune_seg/models/EfficientNet/efficientnet-b3-5fb5a3c3.pth')
            # weights_path='/home/lushun/code/model_compress/model_compress_v1/prune_seg/models/EfficientNet/efficientnet-b3-5fb5a3c3.pth')
        # weights_path='/Users/lushun/Documents/code/model_compress/model_compress_v1/prune_seg/models/EfficientNet/efficientnet-b3-5fb5a3c3.pth')
        self.efficientnet_features = EfficientNet.from_name('efficientnet-b3')
        self.efficientnet_features.set_swish(memory_efficient=False)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        # WITHOUT ASPP
        # self.aspp1 = ASPP_module(1280, 256, rate=rates[0])
        # self.aspp2 = ASPP_module(1280, 256, rate=rates[1])
        # self.aspp3 = ASPP_module(1280, 256, rate=rates[2])
        # self.aspp4 = ASPP_module(1280, 256, rate=rates[3])

        self.relu = nn.ReLU()

        # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                                      nn.Conv2d(1280, 256, 1, stride=1, bias=False),
        #                                      nn.BatchNorm2d(256),
        #                                      nn.ReLU())

        self.conv1 = nn.Conv2d(1536, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(32, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):  # 1,3,512,512
        x, low_level_features = self.efficientnet_features(input)  # 1,2048,32,32; 1,256,128,128;
        input_size_ori = list(input.size())
        input_size = []
        for _size in input_size_ori:
            if torch.is_tensor(_size):
                input_size.append(_size.item())
            else:
                input_size.append(_size)

        # x = self.resnet_features(input)   #

        # ASPP 100-108
        # x1 = self.aspp1(x)  # # 1,256,32,32
        # x2 = self.aspp2(x)
        # x3 = self.aspp3(x)
        # x4 = self.aspp4(x)
        # x5 = self.global_avg_pool(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        # x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print('ERRRRRRRRRRRRRRRR: x_shape: %s, input_size: %s' % (x.shape, input_size))
        # size = list(input.size())
        # size1 = list(input.size())[-2] / 4
        # size2 = list(input.size())[-1] / 4
        x = F.interpolate(x, size=(int(math.ceil(input_size[-2] / 4)),
                                   int(math.ceil(input_size[-1] / 4))), mode='bilinear', align_corners=True)
        # x = F.interpolate(x, scale_factor=(8.0, 8.0), mode='bilinear', align_corners=True)

        # # for onnx
        # x = F.interpolate(x, size=(int(math.ceil(input_size[-2] / 4)),
        #                            int(math.ceil(input_size[-1] / 4))), mode='nearest')
        # x = F.interpolate(x, size=(int(math.ceil(input_size[-2] / 4)),
        #                            int(math.ceil(input_size[-1] / 4))), mode='bilinear', align_corners=False)
        # x = F.interpolate(x, scale_factor=(8, 8), mode='bilinear', align_corners=False)

        # x = F.interpolate(x, size=(128, 128), mode='nearest')

        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        # to avoid onnx error
        low_level_features = F.interpolate(low_level_features, size=(int(math.ceil(input_size[-2] / 4)),
                                   int(math.ceil(input_size[-1] / 4))), mode='bilinear', align_corners=True)

        # print('concat features:', x.shape, low_level_features.shape)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)

        x = F.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=True)
        # x = F.interpolate(x, scale_factor=(4.0, 4.0), mode='bilinear', align_corners=True)
        # # for onnx
        # x = F.interpolate(x, size=(512, 512), mode='nearest')
        # x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        # x = F.interpolate(x, scale_factor=(4, 4), mode='bilinear', align_corners=False)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_Backone(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=8, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus_Backone, self).__init__()

        self.efficientnet_features = EfficientNet.from_name('efficientnet-b3')
        self.efficientnet_features.set_swish(memory_efficient=False)

    def forward(self, input):  # 1,3,512,512
        x, low_level_features = self.efficientnet_features(input)  # 1,2048,32,32; 1,256,128,128;
        return x, low_level_features

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_debug(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=6, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus_debug, self).__init__()
        self.n_classes = n_classes
        # Atrous Conv

        # self.efficientnet_features = EfficientNet.from_pretrained('efficientnet-b3',
            # weights_path='./prune_seg/models/EfficientNet/efficientnet-b3-5fb5a3c3.pth')
            # weights_path='/home/lushun/code/model_compress/model_compress_v1/prune_seg/models/EfficientNet/efficientnet-b3-5fb5a3c3.pth')
        # weights_path='/Users/lushun/Documents/code/model_compress/model_compress_v1/prune_seg/models/EfficientNet/efficientnet-b3-5fb5a3c3.pth')
        self.efficientnet_features = EfficientNet.from_name('efficientnet-b3', image_size=2048)
        self.efficientnet_features.set_swish(memory_efficient=False)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        # WITHOUT ASPP
        # self.aspp1 = ASPP_module(1280, 256, rate=rates[0])
        # self.aspp2 = ASPP_module(1280, 256, rate=rates[1])
        # self.aspp3 = ASPP_module(1280, 256, rate=rates[2])
        # self.aspp4 = ASPP_module(1280, 256, rate=rates[3])

        self.relu = nn.ReLU()

        # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        #                                      nn.Conv2d(1280, 256, 1, stride=1, bias=False),
        #                                      nn.BatchNorm2d(256),
        #                                      nn.ReLU())

        self.conv1 = nn.Conv2d(1536, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2 = nn.Conv2d(32, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):  # 1,3,512,512
        x = self.efficientnet_features(input)  # 1,2048,32,32; 1,256,128,128;
        # input_size_ori = list(input.size())
        # input_size = [1, 3, 2048, 2048]
        # for _size in input_size_ori:
        #     if torch.is_tensor(_size):
        #         input_size.append(_size.item())
        #     else:
        #         input_size.append(_size)

        # x = self.resnet_features(input)   #

        # ASPP 100-108
        # x1 = self.aspp1(x)  # # 1,256,32,32
        # x2 = self.aspp2(x)
        # x3 = self.aspp3(x)
        # x4 = self.aspp4(x)
        # x5 = self.global_avg_pool(x)
        # x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        # x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # # print('ERRRRRRRRRRRRRRRR: x_shape: %s, input_size: %s' % (x.shape, input_size))
        # # size = list(input.size())
        # # size1 = list(input.size())[-2] / 4
        # # size2 = list(input.size())[-1] / 4
        # x = F.interpolate(x, size=(int(math.ceil(input_size[-2] / 4)),
        #                            int(math.ceil(input_size[-1] / 4))), mode='bilinear', align_corners=True)
        # x = x.reshape(-1, 256, 128, 128)
        #
        # # # x = F.interpolate(x, scale_factor=(8.0, 8.0), mode='bilinear', align_corners=True)
        # #
        # # # # for onnx
        # # # x = F.interpolate(x, size=(int(math.ceil(input_size[-2] / 4)),
        # # #                            int(math.ceil(input_size[-1] / 4))), mode='nearest')
        # # # x = F.interpolate(x, size=(int(math.ceil(input_size[-2] / 4)),
        # # #                            int(math.ceil(input_size[-1] / 4))), mode='bilinear', align_corners=False)
        # # # x = F.interpolate(x, scale_factor=(8, 8), mode='bilinear', align_corners=False)
        # #
        # # # x = F.interpolate(x, size=(128, 128), mode='nearest')
        # #
        # low_level_features = self.conv2(low_level_features)
        # low_level_features = self.bn2(low_level_features)
        # low_level_features = self.relu(low_level_features)
        # # low_level_features = F.interpolate(low_level_features, size=(int(math.ceil(input_size[-2] / 4)),
        # #                            int(math.ceil(input_size[-1] / 4))), mode='bilinear', align_corners=True)
        # low_level_features = low_level_features.reshape(-1, 48, 128, 128)
        #
        # #
        # # print('concat features:', x.shape, low_level_features.shape)
        # x = torch.cat((x, low_level_features), dim=1)
        # x = self.last_conv(x)
        # #
        # x = F.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=True)
        # x = x.reshape(-1, self.n_classes, input_size[-2], input_size[-1])
        # # x = F.interpolate(x, scale_factor=(4.0, 4.0), mode='bilinear', align_corners=True)
        # # # for onnx
        # # x = F.interpolate(x, size=(512, 512), mode='nearest')
        # # x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        # # x = F.interpolate(x, scale_factor=(4, 4), mode='bilinear', align_corners=False)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus_debug_v2(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=8, os=16, pretrained=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus_debug_v2, self).__init__()

        # Atrous Conv

        # self.efficientnet_features = EfficientNet.from_pretrained('efficientnet-b3',
            # weights_path='./prune_seg/models/EfficientNet/efficientnet-b3-5fb5a3c3.pth')
            # weights_path='/home/lushun/code/model_compress/model_compress_v1/prune_seg/models/EfficientNet/efficientnet-b3-5fb5a3c3.pth')
        # weights_path='/Users/lushun/Documents/code/model_compress/model_compress_v1/prune_seg/models/EfficientNet/efficientnet-b3-5fb5a3c3.pth')
        self.efficientnet_features = EfficientNet.from_name('efficientnet-b3')
        self.efficientnet_features.set_swish(memory_efficient=False)

        # # ASPP
        # if os == 16:
        #     rates = [1, 6, 12, 18]
        # elif os == 8:
        #     rates = [1, 12, 24, 36]
        # else:
        #     raise NotImplementedError
        # # WITHOUT ASPP
        # # self.aspp1 = ASPP_module(1280, 256, rate=rates[0])
        # # self.aspp2 = ASPP_module(1280, 256, rate=rates[1])
        # # self.aspp3 = ASPP_module(1280, 256, rate=rates[2])
        # # self.aspp4 = ASPP_module(1280, 256, rate=rates[3])
        #
        # self.relu = nn.ReLU()
        #
        # # self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
        # #                                      nn.Conv2d(1280, 256, 1, stride=1, bias=False),
        # #                                      nn.BatchNorm2d(256),
        # #                                      nn.ReLU())
        #
        # self.conv1 = nn.Conv2d(1536, 256, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(256)
        #
        # # adopt [1x1, 48] for channel reduction.
        # self.conv2 = nn.Conv2d(32, 48, 1, bias=False)
        # self.bn2 = nn.BatchNorm2d(48)
        #
        # self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                nn.BatchNorm2d(256),
        #                                nn.ReLU(),
        #                                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                nn.BatchNorm2d(256),
        #                                nn.ReLU(),
        #                                nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    def forward(self, input):  # 1,3,512,512
        x, low_level_features = self.efficientnet_features(input)  # 1,2048,32,32; 1,256,128,128;

        # # comment
        # input_size_ori = list(input.size())
        # input_size = []
        # for _size in input_size_ori:
        #     if torch.is_tensor(_size):
        #         input_size.append(_size.item())
        #     else:
        #         input_size.append(_size)
        #
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = F.interpolate(x, size=(int(math.ceil(input_size[-2] / 4)),
        #                            int(math.ceil(input_size[-1] / 4))), mode='bilinear', align_corners=True)
        #
        # low_level_features = self.conv2(low_level_features)
        # low_level_features = self.bn2(low_level_features)
        # low_level_features = self.relu(low_level_features)
        # # to avoid onnx error
        # low_level_features = F.interpolate(low_level_features, size=(int(math.ceil(input_size[-2] / 4)),
        #                            int(math.ceil(input_size[-1] / 4))), mode='bilinear', align_corners=True)
        #
        # x = torch.cat((x, low_level_features), dim=1)
        # x = self.last_conv(x)
        #
        # x = F.interpolate(x, size=input_size[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_1x_lr_params(model):
    """
    This generator returns all the parameters of the net except for
    the last classification layer. Note that for each batchnorm layer,
    requires_grad is set to False in deeplab_resnext.py, therefore this function does not return
    any batchnorm parameter
    """
    b = [model.resnet_features]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last layer of the net,
    which does the classification of pixel into classes
    """
    b = [model.aspp1, model.aspp2, model.aspp3, model.aspp4, model.conv1, model.conv2, model.last_conv]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


def compute_Params_FLOPs(model, device):
    from thop import profile
    model = model.to(device)
    model.eval()
    inputs = torch.rand(size=(2, 3, 512, 512)).to(device)
    flops, params = profile(model, inputs=(inputs,), verbose=False)
    print("Thop: params = {}M, flops = {}M".format(params / 1e6, flops / 1e6))
    print("sum(params): params = {}M".format(sum(_param.numel() for _param in model.parameters()) / 1e6))


if __name__ == "__main__":
    import os
    import logging

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    inputs_shape = (1, 3, 2048, 2048)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # model = DeepLabv3_plus(nInputChannels=3, n_classes=6, os=16, pretrained=False, _print=False)
    model = DeepLabv3_plus_debug(nInputChannels=3, n_classes=8, os=16, pretrained=False, _print=False)
    print('Parameters number is ', sum(param.numel() for param in model.parameters()))
    print(model)

    # # save_point = torch.load(
    # #     '/Users/lushun/Documents/code/model_compress/model_compress_v1/prune_seg/pretrained/model.pth',
    # #     map_location=device
    # # )
    # # model_param = model.state_dict()
    # # state_dict_param = save_point['state_dict']
    # # for key in model_param.keys():
    # #     if key == 'efficientnet_features._blocks.0._depthwise_conv.conv.weight':
    # #         print(key)
    # #     if key in state_dict_param.keys():
    # #         model_param[key] = state_dict_param[key]
    # #     else:
    # #         new_key = key.split('.')
    # #         del new_key[-2]
    # #         new_key = ".".join(new_key)
    # #         assert new_key in state_dict_param.keys()
    # #         model_param[key] = state_dict_param[new_key]
    # # model.load_state_dict(model_param)
    # # print('load weights done!')
    #
    # prune_name_list = []
    # for name, _, in model.named_modules():
    #     if ('conv' in name and 'static_padding' not in name) or ('_se_reduce' in name and 'static_padding' not in name):
    #     # if '.conv' in name or '_bn' in name:
    #     # if 'conv' in name or 'bn' in name:
    #         print(name)
    #         prune_name_list.append(name)
    # print(prune_name_list)
    # import sys
    # sys.exit()
    # # compute_Params_FLOPs(model, device)
    #
    model.eval()
    image = torch.randn(size=inputs_shape)
    image = image.to(device)
    model.to(device)
    with torch.no_grad():
        output = model.forward(image)
        print('Output size:', output.size())
    sys.exit()
    #
    # from lib.compression.pytorch import ModelSpeedup
    #
    # # masks_file = os.path.join(
    # #     '/Users/lushun/Documents/code/model_compress/model_compress_v1/prune_seg/models/EfficientNet', 'mask.pth'
    # # )
    # # masks_file = os.path.join(
    # #     '/home/lushun/code/model_compress/model_compress_v1/exp_log/seg_deeplab_efficiennetb3_agp_0.5/', 'mask.pth'
    # # )
    # # masks_file = '/home/lushun/.jupyter/results/mask_1.pth'
    # masks_file = './mask_1.pth'
    # mask_pt = torch.load(masks_file, map_location='cpu')
    # m_speedup = ModelSpeedup(model, torch.rand(size=inputs_shape).to(device), masks_file, device)
    # m_speedup.speedup_model()
    # compute_Params_FLOPs(model, device)
