# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PoolFormer implementation
"""
import os
import copy
import torch
from torch.functional import norm
import torch.nn as nn
from arch_generate import arch2_weight, weight2_arch

from nas_vis.nas_burgerformer.timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from nas_vis.nas_burgerformer.timm.models.layers import DropPath, trunc_normal_
from nas_vis.nas_burgerformer.timm.models.registry import register_model
from nas_vis.nas_burgerformer.timm.models.layers.helpers import to_2tuple

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first")
    has_mmdet = False

from nas_vis.nas_burgerformer.search_block import SearchBlock
from nas_vis.nas_burgerformer.operation import GroupNorm


def _cfg(url='', **kwargs):
    return {'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None, 'crop_pct': .95, 'interpolation': 'bicubic', 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head', **kwargs}


default_cfgs = {
    'unifiedarch_s': _cfg(crop_pct=0.9),
}


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=None, norm_track=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()  # HardSwish(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ViTNASPatchEmbed(nn.Module):
    '''
        Use several conv for patch embedding
    '''
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=None, img_size=224, mid_chans=24):
        super(ViTNASPatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_grid = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patch_grid[0] * self.patch_grid[1]

        self.conv1 = ConvBnAct(in_channels=in_chans, out_channels=mid_chans, stride=(2, 2))
        self.conv2 = ConvBnAct(in_channels=mid_chans, out_channels=mid_chans)
        self.conv3 = ConvBnAct(in_channels=mid_chans, out_channels=mid_chans)

        assert self.patch_size[0] % 2 == 0
        assert self.patch_size[1] % 2 == 0
        self.conv_proj = nn.Conv2d(mid_chans, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]

        x = self.conv1(x)
        x_res = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + x_res
        x = self.conv_proj(x)

        return x


def basic_blocks(
    dim,
    H,
    W,
    index,
    layers,
    has_atten_mlp=True,
    num_heads=5,
    mlp_ratio=4.,
    expand_ratio=6,
    act_layer=nn.GELU,
    drop_rate=.0,
    drop_path_rate=0.,
    use_layer_scale=True,
    layer_scale_init_value=1e-5,
    norm_track=True,
):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks 
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(
            SearchBlock(
                dim,
                H,
                W,
                has_atten_mlp=has_atten_mlp,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate,
                drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                norm_track=norm_track,
                expand_ratio=expand_ratio,
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class UnifiedArch(nn.Module):
    """
    # todo
    PoolFormer, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, --pool_size: the embedding dims, mlp ratios and 
        pooling size for the 4 stages
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalizaiotn and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad: 
        specify the downsample (patch embed.)
    --fork_faat: whetehr output features of the 4 stages, for dense prediction
    --init_cfg，--pretrained: 
        for mmdetection and mmsegmentation to load pretrianfed weights
    """
    def __init__(self,
                 layers,
                 H=None,
                 W=None,
                 embed_dims=None,
                 mlp_ratios=None,
                 expand_ratios=None,
                 downsamples=None,
                 has_atten_mlp=None,
                 num_heads=None,
                 act_layer=nn.GELU,
                 norm_layer=GroupNorm,
                 num_classes=1000,
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=2,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        # self.patch_embed = PatchEmbed(patch_size=in_patch_size, stride=in_stride, padding=in_pad, in_chans=3, embed_dim=embed_dims[0], norm_layer=norm_layer)
        self.patch_embed = ViTNASPatchEmbed(in_stride, stride=in_stride, padding=in_pad, in_chans=3, embed_dim=embed_dims[0], norm_layer=norm_layer)

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i],
                                 H[i],
                                 W[i],
                                 i,
                                 layers,
                                 has_atten_mlp=has_atten_mlp[i],
                                 num_heads=num_heads[i],
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 norm_track=kwargs['norm_track'],
                                 expand_ratio=expand_ratios[i])

            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(PatchEmbed(patch_size=down_patch_size, stride=down_stride, padding=down_pad, in_chans=embed_dims[i], embed_dim=embed_dims[i + 1], norm_layer=norm_layer))

        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for ' f'{self.__class__.__name__}, ' f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            print('missing_keys: ', missing_keys)
            print('unexpected_keys: ', unexpected_keys)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x, skip_weightsss, norm_mixer_act_weightssss, depths, widths, ratios):
        outs = []
        k = -1
        for idx, block in enumerate(self.network):
            if isinstance(block, nn.Sequential):
                k += 1
                # mask = torch.zeros(1, x.size(1), 1, 1).to(x.device)
                # mask[:, :widths[k], :, :] = 1
                # x = x * mask
                for b in block[:depths[k]]:
                    x = b(x, skip_weightsss[k], norm_mixer_act_weightssss[k], widths[k], ratios[k])
            else:
                x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x, skip_weightsss, norm_mixer_act_weightssss, depths, widths, ratios):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x, skip_weightsss, norm_mixer_act_weightssss, depths, widths, ratios)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out


@register_model
def unifiedarch_s20(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [4, 4, 8, 4]
    embed_dims = [96, 128, 320, 512]
    H = [56, 28, 14, 7]
    W = [56, 28, 14, 7]
    mlp_ratios = [4, 4, 4, 4]
    expand_ratios = [6, 6, 6, 6]
    downsamples = [True, True, True, True]
    has_atten_mlp = [False, False, True, True]
    num_heads = [-1, -1, 4, 8]
    model = UnifiedArch(
        layers,
        H=H,
        W=W,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        expand_ratios=expand_ratios,
        downsamples=downsamples,
        has_atten_mlp=has_atten_mlp,
        num_heads=num_heads,
        **kwargs,
    )
    model.default_cfg = default_cfgs['unifiedarch_s']
    return model


@register_model
def unifiedarch_s24(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [6, 6, 6, 6]
    embed_dims = [128, 256, 448, 768]
    H = [56, 28, 14, 7]
    W = [56, 28, 14, 7]
    mlp_ratios = [4, 4, 4, 4]
    expand_ratios = [6, 6, 6, 6]
    downsamples = [True, True, True, True]
    has_atten_mlp = [False, False, True, True]
    num_heads = [-1, -1, 4, 8]
    model = UnifiedArch(
        layers,
        H=H,
        W=W,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        expand_ratios=expand_ratios,
        downsamples=downsamples,
        has_atten_mlp=has_atten_mlp,
        num_heads=num_heads,
        **kwargs,
    )
    model.default_cfg = default_cfgs['unifiedarch_s']
    return model


@register_model
def unifiedarch_s28(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [6, 6, 8, 8]
    embed_dims = [128, 256, 320, 512]
    H = [56, 28, 14, 7]
    W = [56, 28, 14, 7]
    mlp_ratios = [4, 4, 4, 4]
    expand_ratios = [6, 6, 6, 6]
    downsamples = [True, True, True, True]
    has_atten_mlp = [False, False, True, True]
    num_heads = [-1, -1, 4, 8]
    model = UnifiedArch(
        layers,
        H=H,
        W=W,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        expand_ratios=expand_ratios,
        downsamples=downsamples,
        has_atten_mlp=has_atten_mlp,
        num_heads=num_heads,
        **kwargs,
    )
    model.default_cfg = default_cfgs['unifiedarch_s']
    return model


@register_model
def unifiedarch_s8(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 2, 2]
    embed_dims = [64, 128, 320, 512]
    H = [56, 28, 14, 7]
    W = [56, 28, 14, 7]
    mlp_ratios = [4, 4, 4, 4]
    expand_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    has_atten_mlp = [False, False, True, True]
    num_heads = [-1, -1, 4, 8]
    model = UnifiedArch(
        layers,
        H=H,
        W=W,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        expand_ratios=expand_ratios,
        downsamples=downsamples,
        has_atten_mlp=has_atten_mlp,
        num_heads=num_heads,
        **kwargs,
    )
    model.default_cfg = default_cfgs['unifiedarch_s']
    return model


@register_model
def unifiedarch_s12(pretrained=False, **kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    --layers: [x,x,x,x], numbers of layers for the four stages
    --embed_dims, --mlp_ratios: 
        embedding dims and mlp ratios for the four stages
    --downsamples: flags to apply downsampling or not in four blocks
    """
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    H = [56, 28, 14, 7]
    W = [56, 28, 14, 7]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    has_atten_mlp = [False, False, True, True]
    num_heads = [-1, -1, 5, 8]
    model = UnifiedArch(
        layers,
        H=H,
        W=W,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        has_atten_mlp=has_atten_mlp,
        num_heads=num_heads,
        **kwargs,
    )
    model.default_cfg = default_cfgs['unifiedarch_s']
    return model


if has_mmseg and has_mmdet:
    """
    The following models are for dense prediction based on 
    mmdetection and mmsegmentation
    """
    @seg_BACKBONES.register_module()
    @det_BACKBONES.register_module()
    class unifiedarch_s8_feat(UnifiedArch):
        """
        PoolFormer-S12 model, Params: 12M
        """
        def __init__(self, **kwargs):
            layers = [2, 2, 2, 2]
            embed_dims = [64, 128, 320, 512]
            mlp_ratios = [4, 4, 4, 4]
            downsamples = [True, True, True, True]
            has_atten_mlp = [False, False, True, True]
            num_heads = [-1, -1, 5, 8]
            super().__init__(
                layers,
                embed_dims=embed_dims,
                mlp_ratios=mlp_ratios,
                downsamples=downsamples,
                has_atten_mlp=has_atten_mlp,
                num_heads=num_heads,
                **kwargs,
            )


if __name__ == "__main__":
    supernet = unifiedarch_s28(norm_track=True).cuda()
    num_params = 0
    for param in supernet.parameters():
        num_params += param.numel()
    print(num_params / 1e6, 'M')

    x = torch.randn(32, 3, 224, 224).cuda()

    # skip_weightsss = [torch.ones(6, 2), torch.ones(6, 2), torch.ones(6, 2), torch.ones(6, 2)]
    # norm_mixer_act_weightssss = [[
    #     [torch.ones(4), torch.ones(4), torch.ones(3), torch.ones(4), torch.ones(4)],
    #     [torch.ones(4), torch.ones(4), torch.ones(3), torch.ones(4), torch.ones(4)],
    #     [torch.ones(4), torch.ones(4), torch.ones(3), torch.ones(4), torch.ones(4)],
    # ], [
    #     [torch.ones(4), torch.ones(4), torch.ones(3), torch.ones(4), torch.ones(4)],
    #     [torch.ones(4), torch.ones(4), torch.ones(3), torch.ones(4), torch.ones(4)],
    #     [torch.ones(4), torch.ones(4), torch.ones(3), torch.ones(4), torch.ones(4)],
    # ], [
    #     [torch.ones(4), torch.ones(4), torch.ones(5), torch.ones(4), torch.ones(4)],
    #     [torch.ones(4), torch.ones(4), torch.ones(5), torch.ones(4), torch.ones(4)],
    #     [torch.ones(4), torch.ones(4), torch.ones(5), torch.ones(4), torch.ones(4)],
    # ], [
    #     [torch.ones(4), torch.ones(4), torch.ones(5), torch.ones(4), torch.ones(4)],
    #     [torch.ones(4), torch.ones(4), torch.ones(5), torch.ones(4), torch.ones(4)],
    #     [torch.ones(4), torch.ones(4), torch.ones(5), torch.ones(4), torch.ones(4)],
    # ]]

    from arch_generate import arch_generate
    skip_weightsss, norm_mixer_act_weightssss, depths, widths, ratios = arch_generate(x.device)
    print(skip_weightsss)
    print(norm_mixer_act_weightssss)
    print(depths, widths, ratios)
    print(weight2_arch(skip_weightsss, norm_mixer_act_weightssss, depths, widths, ratios))
    y = supernet(x, skip_weightsss, norm_mixer_act_weightssss, depths, widths, ratios)
    print(y.size())
