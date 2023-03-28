import torch
from torch import nn
import torch.nn.functional as F

import models.prune_seg.models.PSPNet.resnet as models


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True,
                pretrained=False):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        # self.criterion = criterion

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim / len(bins)), bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        # if self.training:
        #     self.aux = nn.Sequential(
        #         nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout2d(p=dropout),
        #         nn.Conv2d(256, classes, kernel_size=1)
        #     )

    def forward(self, x):
        x_size = list(x.size())

        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) // 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) // 8 * self.zoom_factor + 1)
        # assert (x_size[2]) % 8 == 0 and (x_size[3]) % 8 == 0
        # h = int((x_size[2]) / 8 * self.zoom_factor)
        # w = int((x_size[3]) / 8 * self.zoom_factor)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x = self.layer4(x_tmp)
        if self.use_ppm:
            x = self.ppm(x)
        x = self.cls(x)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        return x
        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x


if __name__ == '__main__':
    import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
    # inputs = torch.rand(4, 3, 473, 473).cuda()
    # model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=21, zoom_factor=1, use_ppm=True, pretrained=True).cuda()
    inputs = torch.rand(2, 3, 713, 713)
    # inputs = torch.rand(2, 3, 512, 512)
    model = PSPNet(layers=50, classes=19, zoom_factor=8, pretrained=False)
    model.eval()

    # from thop import profile
    # flops, params = profile(model, inputs=(inputs,), verbose=False)
    # print("Thop: params = {}M, flops = {}M".format(params / 1e6, flops / 1e6))
    # print("sum(params): params = {}M".format(sum(_param.numel() for _param in model.parameters()) / 1e6))

    output = model(inputs)
    print('PSPNet', output.size())
    prune_name_list = []
    last_name = None
    for name, module, in model.named_modules():
        # if ('conv' in name and 'static_padding' not in name) or ('_se_reduce' in name and 'static_padding' not in name):

        # # print in_place dict
        # if isinstance(module, nn.Conv2d) or isinstance(module, nn.BatchNorm2d):
        #     # print(name, module)
        #     if last_name is None:
        #         print('\'%s\': (%s, ),' % (name, last_name))
        #     else:
        #         print('\'%s\': (\'%s\', ),' % (name, last_name))
        #     if isinstance(module, nn.Conv2d):
        #         last_name = name

        # print op_names dict
        if isinstance(module, nn.Conv2d):
            prune_name_list.append(name)

        # if 'conv' in name or 'bn' in name:
        # print(name, module)
        # if 'aux' not in name and name != '' and '.' in name:
        #     if 'conv' in name or 'Conv2d' in str(module):
        # prune_name_list.append(name)
    print(prune_name_list)
    print(len(prune_name_list))
    print(model)

