import torch
import torch.nn as nn
from itertools import repeat
from nas_vis.nas_burgerformer.timm.models.layers import trunc_normal_
import collections.abc

OPS = {
    'skip': lambda kwargs: nn.Identity(),
    'none': lambda kwargs: Zero(),
    'BN': lambda kwargs: nn.BatchNorm2d(num_features=kwargs['dim'], track_running_stats=kwargs['norm_track']),
    'GN': lambda kwargs: GroupNorm(num_channels=kwargs['dim']),
    'LN': lambda kwargs: LayerNormChannel(num_channels=kwargs['dim'], eps=1e-6),
    'relu6': lambda kwargs: nn.ReLU6(),
    'gelu': lambda kwargs: nn.GELU(),
    'silu': lambda kwargs: nn.SiLU(),
    'conv_1x1': lambda kwargs: nn.Conv2d(kwargs['in_channels'], kwargs['out_channels'], kernel_size=1, stride=1),
    'dwise_3x3': lambda kwargs: nn.Conv2d(kwargs['in_channels'], kwargs['out_channels'], kernel_size=3, stride=1, padding=1, groups=kwargs['groups']),
    'avgpool': lambda kwargs: Pooling(),
    'self_atten': lambda kwargs: Attention(kwargs['dim'], kwargs['head']),
    'spatial_mlp': lambda kwargs: Mlp(kwargs['seq_len']),  # Mlp(kwargs['in_features'], kwargs['hidden_features'], kwargs['out_features'], kwargs['act_layer'], drop=kwargs['drop']),
    'channel_mlp': lambda kwargs: ChannelMlp(kwargs['in_features'], kwargs['hidden_features'], kwargs['out_features'], kwargs['act_layer'], drop=kwargs['drop']),
}


class Zero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mul(0.)


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, head_dim_ratio=1., attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = round(dim // num_heads * head_dim_ratio)
#         self.head_dim = head_dim
#         self.scale = head_dim ** -0.5
#         self.qkv = nn.Conv2d(dim, head_dim * num_heads * 3, 1, stride=1, padding=0, bias=False)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Conv2d(self.head_dim * self.num_heads, dim, 1, stride=1, padding=0, bias=False)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, -1).permute(1, 0, 2, 4, 3)
#         q, k, v = x[0], x[1], x[2]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x = attn @ v

#         x = x.permute(0, 1, 3, 2).reshape(B, -1, H, W)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
        return x


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """ spatial mlp
    """
    def __init__(self, seq_len):
        super().__init__()

        self.fc = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2)

        x = self.fc(x)

        x = x.reshape(B, C, H, W).contiguous()
        return x


# class Mlp(nn.Module):
#     """ MLP as used in Vision Transformer, MLP-Mixer and related networks
#     """
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         drop_probs = to_2tuple(drop)

#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop_probs[0])
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop2 = nn.Dropout(drop_probs[1])

#     def forward(self, x):
#         B, C, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)  # B, N, C

#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)

#         x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
#         return x


class ChannelMlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x