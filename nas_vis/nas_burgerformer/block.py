import torch
import torch.nn as nn
from nas_vis.nas_burgerformer.search_config import CONFIG
from nas_vis.nas_burgerformer.operation import OPS
from nas_vis.nas_burgerformer.timm.models.layers import DropPath
from nas_vis.nas_burgerformer.operation import *


class SkipBlock(nn.Module):
    def __init__(self, micro_arch_str):
        super().__init__()
        self.op = OPS[micro_arch_str]({})

    def forward(self, x):
        out = self.op(x)
        return out


class NormMixerActBlock(nn.Module):
    def __init__(
        self,
        micro_arch_str,
        dim,
        H,
        W,
        num_heads=5,
        mlp_ratio=4,
        act_layer=nn.GELU,
        drop=0.,
        drop_path=0.,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        use_expand=False,
        expand_ratio=4,
    ):
        # num_heads: stage 3:5 stage 4:8
        # mlp_ratio: stage 3 4 stage 4
        super().__init__()

        self.use_expand = use_expand
        if self.use_expand:
            self.expand_start = nn.Sequential(OPS['GN']({'dim': dim}), nn.Conv2d(dim, expand_ratio * dim, 1), act_layer())
            self.expand_end = nn.Sequential(nn.Conv2d(expand_ratio * dim, dim, 1))
            dim = expand_ratio * dim

        ops_list = []
        ops_str = micro_arch_str.split('-')
        for op in ops_str:
            if op in ['skip']:
                continue
            if op in CONFIG['normopact_module']['act']:
                ops_list.append(OPS[op]({}))
            if op in CONFIG['normopact_module']['norm']:
                ops_list.append(OPS[op]({'dim': dim, 'norm_track': True}))
            if op in CONFIG['normopact_module']['meat_op' if use_expand else 'bread_op'] or op == 'channel_mlp':
                ops_list.append(OPS[op]({
                    'in_channels': dim,  # conv_1x1, dwise_3x3
                    'out_channels': dim,  # conv_1x1, dwise_3x3
                    'groups': dim,  # dwise_3x3
                    'dim': dim,  # self-atten
                    'head': num_heads,  # self-atten
                    'seq_len': int(H * W),  # spatial_mlp
                    'in_features': dim,  # channel_mlp
                    'hidden_features': mlp_ratio * dim,  # channel_mlp,
                    'out_features': dim,  # channel_mlp
                    'act_layer': act_layer,  # channel_mlp
                    'drop': drop,  # channel_mlp
                }))
        if len(ops_list) == 0:
            ops_list.append(OPS['skip']({}))
        self.ops = nn.Sequential(*ops_list)

        if self.use_expand:
            dim = dim // expand_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale = nn.Identity()

    def forward(self, x):
        if self.use_expand:
            x = self.expand_start(x)
        x = self.ops(x)
        if self.use_expand:
            x = self.expand_end(x)
        if self.use_layer_scale:
            x = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x
        x = self.drop_path(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        micro_arch,
        dim,
        H,
        W,
        num_heads=5,
        mlp_ratio=4,
        act_layer=nn.GELU,
        drop=0.,
        drop_path=0.,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        multiplier=3,
        expand_ratio=4,
    ):
        super().__init__()

        norm_mixer_act_block_list = []
        skip_block_list = []

        for i in range(multiplier):
            norm_mixer_act_block_list.append(
                NormMixerActBlock(
                    micro_arch["{}".format(i + 1)],
                    dim,
                    H,
                    W,
                    num_heads,
                    mlp_ratio,
                    act_layer,
                    int(micro_arch["is_drop"][i]) * drop,
                    int(micro_arch["is_drop"][i]) * drop_path,
                    micro_arch["use_layer_scale"][i],
                    layer_scale_init_value,
                    use_expand=True if i == 1 else False,
                    expand_ratio=expand_ratio,
                ))
            for j in range(i + 1):
                skip_block_list.append(SkipBlock(micro_arch["{}->{}".format(j, i + 1)]))

        self.norm_mixer_act_blocks = nn.ModuleList(norm_mixer_act_block_list)
        self.skip_blocks = nn.ModuleList(skip_block_list)
        self.multiplier = multiplier

    def forward(self, x, cal_flops=False):
        if cal_flops:
            flops = 0

        outs = [x]
        pos = 0
        for i in range(self.multiplier):
            out = self.norm_mixer_act_blocks[i](outs[i])

            if cal_flops:
                flops += self.flops(self.norm_mixer_act_blocks[i], outs[i].size(), out.size())

            for j in range(i + 1):
                if isinstance(self.skip_blocks[pos + j].op, nn.Identity):
                    out += self.skip_blocks[pos + j](outs[j])
            pos += i + 1
            outs.append(out)

        if cal_flops:
            return outs[-1], flops
        return outs[-1]

    def flops(self, norm_mixer_act_block, input_size, output_size):
        assert input_size == output_size
        flops = 0
        for b in norm_mixer_act_block.ops:
            if isinstance(b, nn.Identity):
                flops += 0
            elif isinstance(b, nn.BatchNorm2d) or isinstance(b, GroupNorm) or isinstance(b, LayerNormChannel):
                flops += 2 * input_size[1] * input_size[2] * input_size[3]
            elif isinstance(b, nn.ReLU6) or isinstance(b, nn.GELU) or isinstance(b, nn.SiLU):  # approximation
                flops += input_size[1] * input_size[2] * input_size[3]
            elif isinstance(b, Pooling):
                flops += output_size[1] * output_size[2] * output_size[3]
            elif isinstance(b, nn.Conv2d):
                flops += output_size[1] * output_size[2] * output_size[3] * (b.in_channels / b.groups * b.kernel_size[0] * b.kernel_size[1] + (1 if b.bias is not None else 0))
            elif isinstance(b, Attention):
                embed_dim = input_size[1]
                n_seq = input_size[2] * input_size[3]
                num_heads = b.num_heads
                head_dim = embed_dim // num_heads
                # embedding -> qkv
                flops += embed_dim * num_heads * head_dim * 3 * n_seq
                # attention score
                flops += n_seq * n_seq * num_heads * head_dim
                flops += n_seq * n_seq * head_dim * num_heads  # weighted average
                flops += n_seq * (head_dim * num_heads * embed_dim)
            elif isinstance(b, Mlp):
                embed_dim = input_size[1]
                n_seq = input_size[2] * input_size[3]
                hidden_size = b.fc1.out_features
                flops += n_seq * embed_dim * hidden_size
                flops += n_seq * embed_dim * hidden_size
            elif isinstance(b, ChannelMlp):
                flops += b.fc1.in_channels * b.fc1.out_channels * output_size[2] * output_size[3]
                flops += b.fc2.in_channels * b.fc2.out_channels * output_size[2] * output_size[3]
        return flops


if __name__ == "__main__":
    micro_arch = {
        "1": "skip-GN-avgpool-skip-skip",
        "2": "skip-GN-conv1x1-skip-gelu",
        "3": "skip-skip-conv1x1-skip-skip",
        "0->1": "skip",
        "0->2": "none",
        "1->2": "none",
        "0->3": "none",
        "1->3": "skip",
        "2->3": "none",
    }
    a = Block(micro_arch, 320)
    x = torch.randn(128, 320, 14, 14)

    b = a(x)
    print(b.size())