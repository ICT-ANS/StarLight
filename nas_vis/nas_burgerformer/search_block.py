import torch
import torch.nn as nn
from nas_vis.nas_burgerformer.search_config import CONFIG
from nas_vis.nas_burgerformer.operation import OPS
from nas_vis.nas_burgerformer.timm.models.layers import DropPath

SKIP_INDEX = 0


class SkipBlock(nn.Module):
    def __init__(self):
        super().__init__()
        ops = []
        for op in CONFIG['skip_module']:
            ops.append(OPS[op]({}))
        self.ops = nn.ModuleList(ops)

    def forward(self, x, weights):
        out = 0
        for w, op in zip(weights, self.ops):
            if w != 0:
                out += w * op(x)
        return out


class NormMixerActBlock(nn.Module):
    def __init__(
        self,
        dim,
        H,
        W,
        has_atten_mlp=True,
        num_heads=5,
        mlp_ratio=4,
        act_layer=nn.GELU,
        drop=0.,
        drop_path=0.,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        norm_track=True,
        use_expand=False,
        expand_ratio=4,
    ):
        # num_heads: stage 3:5 stage 4:8
        # mlp_ratio: stage 3 4 stage 4
        super().__init__()
        self.has_atten_mlp = has_atten_mlp

        self.use_expand = use_expand
        if self.use_expand:
            self.expand_ratio = expand_ratio
            self.expand_start = nn.Sequential(OPS['GN']({'dim': dim}), nn.Conv2d(dim, expand_ratio * dim, 1), act_layer())
            self.expand_end = nn.Sequential(nn.Conv2d(expand_ratio * dim, dim, 1))
            dim = expand_ratio * dim

        act_layer_0 = []
        norm_layer_1 = []
        mixer_layer_2 = []
        norm_layer_3 = []
        act_layer_4 = []

        for op in CONFIG['normopact_module']['act']:
            act_layer_0.append(OPS[op]({}))
            act_layer_4.append(OPS[op]({}))
        for op in CONFIG['normopact_module']['norm']:
            norm_layer_1.append(OPS[op]({'dim': dim, 'norm_track': norm_track}))
            norm_layer_3.append(OPS[op]({'dim': dim, 'norm_track': norm_track}))
        for op in CONFIG['normopact_module']['meat_op' if use_expand else 'bread_op']:
            if ('atten' in op or 'spatial_mlp' in op) and not has_atten_mlp:
                continue
            mixer_layer_2.append(OPS[op]({
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

        self.act_layer_0 = nn.ModuleList(act_layer_0)
        self.norm_layer_1 = nn.ModuleList(norm_layer_1)
        self.mixer_layer_2 = nn.ModuleList(mixer_layer_2)
        self.norm_layer_3 = nn.ModuleList(norm_layer_3)
        self.act_layer_4 = nn.ModuleList(act_layer_4)
        self.layers = [self.act_layer_0, self.norm_layer_1, self.mixer_layer_2, self.norm_layer_3, self.act_layer_4]

        if self.use_expand:
            dim = dim // expand_ratio

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.layer_scale = nn.Identity()

    def forward(self, x, weightss, width, ratio):
        mask = torch.zeros(1, x.size(1), 1, 1).to(x.device)
        mask[:, :width, :, :] = 1
        x = x * mask

        if self.use_expand:
            new_mask = torch.zeros(1, x.size(1) * self.expand_ratio, 1, 1).to(x.device)
            new_mask[:, :width * ratio, :, :] = 1
        else:
            new_mask = mask

        assert len(weightss) == 5
        if self.use_expand:
            x = self.expand_start(x)
            x = x * new_mask
        for weights, layers in zip(weightss, self.layers):
            out = 0
            for w, l in zip(weights, layers):
                if w != 0:
                    out += w * l(x) * new_mask
            x = out
        if self.use_expand:
            x = self.expand_end(x)
            x = x * mask

        if self.use_expand or (not weightss[2][SKIP_INDEX] == 1):  # skip op not use self.use_layer_scale
            if self.use_layer_scale:
                x = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * x
            x = self.drop_path(x)
        return x * mask


COUNT = 0


class SearchBlock(nn.Module):
    def __init__(
        self,
        dim,
        H,
        W,
        has_atten_mlp=True,
        num_heads=5,
        mlp_ratio=4,
        act_layer=nn.GELU,
        drop=0.,
        drop_path=0.,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        multiplier=3,
        norm_track=True,
        expand_ratio=4,
    ):
        super().__init__()

        norm_op_act_block_list = []
        skip_block_list = []

        for i in range(multiplier):
            norm_op_act_block_list.append(
                NormMixerActBlock(
                    dim,
                    H,
                    W,
                    has_atten_mlp,
                    num_heads,
                    mlp_ratio,
                    act_layer,
                    drop,
                    drop_path,
                    use_layer_scale,
                    layer_scale_init_value,
                    norm_track,
                    use_expand=True if i == 1 else False,
                    expand_ratio=expand_ratio,
                ))
            for _ in range(i + 1):
                skip_block_list.append(SkipBlock())

        self.norm_op_act_blocks = nn.ModuleList(norm_op_act_block_list)
        self.skip_blocks = nn.ModuleList(skip_block_list)
        self.multiplier = multiplier

    def forward(self, x, skip_weightss, norm_op_act_weightsss, width, ratio):
        mask = torch.zeros(1, x.size(1), 1, 1).to(x.device)
        mask[:, :width, :, :] = 1
        x = x * mask

        outs = [x]
        pos = 0
        for i in range(self.multiplier):
            out = self.norm_op_act_blocks[i](outs[i], norm_op_act_weightsss[i], width, ratio)
            for j in range(i + 1):
                out += self.skip_blocks[pos + j](outs[j], skip_weightss[pos + j])
            # global COUNT
            # if COUNT < 5:
            #     print(out.mean(), out.std())
            #     COUNT += 1
            pos += i + 1
            outs.append(out * mask)

        return outs[-1] * mask


if __name__ == "__main__":
    a = SearchBlock(320, 14, 14)
    x = torch.randn(128, 320, 14, 14)
    skip_weightss = torch.ones(6, 2)
    norm_op_act_weightss = [
        [torch.ones(4), torch.ones(4), torch.ones(6), torch.ones(4), torch.ones(4)],
        [torch.ones(4), torch.ones(4), torch.ones(2), torch.ones(4), torch.ones(4)],
        [torch.ones(4), torch.ones(4), torch.ones(6), torch.ones(4), torch.ones(4)],
    ]
    b = a(x, skip_weightss, norm_op_act_weightss, width=160, ratio=2)
    print(b.size())