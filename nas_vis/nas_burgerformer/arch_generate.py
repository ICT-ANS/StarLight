from os import path
import torch
import random
from torch._C import Argument
from torch.functional import norm
from nas_vis.nas_burgerformer.search_config import CONFIG

SKIP_INDEX = 0


def is_residual(weights, norm_op_act_weightsss=None):
    def is_residual_edge(edge):
        if edge in [0, 5] and norm_op_act_weightsss is not None:  # 0->1, 2->3
            return weights[edge][SKIP_INDEX] == 1 or norm_op_act_weightsss[0 if edge == 0 else 2][2][SKIP_INDEX] == 1  # op is skip
        else:
            return weights[edge][SKIP_INDEX] == 1

    edge, ops = weights.size()
    assert edge == 6 and ops == 2
    if is_residual_edge(3):  # 0->3
        return True
    if is_residual_edge(0) and is_residual_edge(4):  # 0->1->3
        return True
    if is_residual_edge(1) and is_residual_edge(5):  # 0->2->3
        return True
    if is_residual_edge(0) and is_residual_edge(2) and is_residual_edge(5):  # 0->1->2->3
        return True
    return False


def is_good_arch(network_def):
    skip_weightsss, norm_op_act_weightssss, _, _, _ = network_def
    for skip_weightss, norm_op_act_weightsss in zip(skip_weightsss, norm_op_act_weightssss):  # has residual
        if not is_residual(skip_weightss, norm_op_act_weightsss):
            return False
    return True


def arch_generate(device, generate_micro=True, generate_macro=True, stage=4, multiplier=3, has_atten_mlp=CONFIG['normopact_module']['has_atten_mlp'], search_residual=False, pre_defind_arch=None, min_macro=False, max_macro=False):
    with torch.no_grad():
        assert not min_macro or not max_macro

        skip_edge_num = sum([i + 1 for i in range(multiplier)])
        skip_op_num = len(CONFIG['skip_module'])
        norm_op_num = len(CONFIG['normopact_module']['norm'])
        bread_op_num = len(CONFIG['normopact_module']['bread_op'])
        meat_op_num = len(CONFIG['normopact_module']['meat_op'])
        act_op_num = len(CONFIG['normopact_module']['act'])

        skip_weightsss = []
        norm_op_act_weightssss = []
        widths = []
        depths = []
        ratios = []

        if not generate_micro or not generate_macro:
            assert pre_defind_arch is not None
            pre_defind_skip_weightsss, pre_defind_norm_op_act_weightssss, pre_defind_depths, pre_defind_widths, pre_defind_ratios = arch2_weight(pre_defind_arch, device)
        if not generate_micro:
            skip_weightsss = pre_defind_skip_weightsss
            norm_op_act_weightssss = pre_defind_norm_op_act_weightssss
        if not generate_macro:
            widths = pre_defind_widths
            depths = pre_defind_depths
            ratios = pre_defind_ratios

        for i in range(stage):
            if generate_macro:
                # depth
                depth_candidate = CONFIG['depth']['stage-{}'.format(i + 1)]
                if min_macro:
                    depths.append(min(depth_candidate))
                elif max_macro:
                    depths.append(max(depth_candidate))
                else:
                    idx = torch.randint(0, len(depth_candidate), []).item()
                    depths.append(depth_candidate[idx])

                # width
                width_candidate = CONFIG['width']['stage-{}'.format(i + 1)]
                if min_macro:
                    widths.append(min(width_candidate))
                elif max_macro:
                    widths.append(max(width_candidate))
                else:
                    idx = torch.randint(0, len(width_candidate), []).item()
                    widths.append(width_candidate[idx])

                # ratio
                ratio_candidate = CONFIG['ratio']['stage-{}'.format(i + 1)]
                if min_macro:
                    ratios.append(min(ratio_candidate))
                elif max_macro:
                    ratios.append(max(ratio_candidate))
                else:
                    idx = torch.randint(0, len(ratio_candidate), []).item()
                    ratios.append(ratio_candidate[idx])

            if generate_micro:
                if not search_residual:  # skip (fixed pattern)
                    skip_weightss = torch.zeros(skip_edge_num, skip_op_num).to(device)
                    skip_weightss[0][SKIP_INDEX] = 1
                    skip_weightss[2][SKIP_INDEX] = 1
                    skip_weightss[5][SKIP_INDEX] = 1
                    NONE_INDEX = 1
                    skip_weightss[1][NONE_INDEX] = 1
                    skip_weightss[3][NONE_INDEX] = 1
                    skip_weightss[4][NONE_INDEX] = 1
                else:  # skip (dynamic)
                    while True:
                        skip_weightss = torch.zeros(skip_edge_num, skip_op_num).to(device)
                        for j in range(skip_edge_num):
                            idx = torch.randint(0, skip_op_num, []).item()
                            skip_weightss[j][idx] = 1
                        if is_residual(skip_weightss):
                            break

                norm_op_act_weightsss = []
                for j in range(multiplier):
                    if j == 1:
                        op_op_num = meat_op_num
                    elif not has_atten_mlp[i]:
                        op_op_num = bread_op_num - 2
                    else:
                        op_op_num = bread_op_num
                    norm_op_act_weightss = [
                        torch.zeros(act_op_num).to(device),
                        torch.zeros(norm_op_num).to(device),
                        torch.zeros(op_op_num).to(device),
                        torch.zeros(norm_op_num).to(device),
                        torch.zeros(act_op_num).to(device),
                    ]
                    # select op
                    op_idx = torch.randint(0, op_op_num, []).item()
                    norm_op_act_weightss[2][op_idx] = 1
                    if op_idx == SKIP_INDEX and j != 1:  # if op is skip, no need residual except meat part
                        skip_weightss[[0, 2, 5][j]][0] = 0
                        skip_weightss[[0, 2, 5][j]][1] = 1

                    # select norm
                    if op_idx == SKIP_INDEX:
                        norm_op_act_weightss[1][SKIP_INDEX] = 1
                        norm_op_act_weightss[3][SKIP_INDEX] = 1
                    else:
                        randint = torch.randint(0, 2, []).item()
                        norm_select, norm_not_select = [1, 3][randint], [3, 1][randint]

                        while True:
                            norm_idx = torch.randint(0, norm_op_num, []).item()
                            if norm_idx != SKIP_INDEX:
                                break

                        norm_op_act_weightss[norm_select][norm_idx] = 1
                        norm_op_act_weightss[norm_not_select][SKIP_INDEX] = 1

                    # select act
                    norm_op_act_weightss[0][SKIP_INDEX] = 1  # first act layer is not used
                    if op_idx == SKIP_INDEX:
                        norm_op_act_weightss[4][SKIP_INDEX] = 1
                    else:
                        act_idx = torch.randint(0, act_op_num, []).item()
                        norm_op_act_weightss[4][act_idx] = 1

                    norm_op_act_weightsss.append(norm_op_act_weightss)

                skip_weightsss.append(skip_weightss)
                norm_op_act_weightssss.append(norm_op_act_weightsss)

        return skip_weightsss, norm_op_act_weightssss, depths, widths, ratios


def weight2_arch(skip_weightsss, norm_op_act_weightssss, depths, widths, ratios, stage=4, multiplier=3):
    with torch.no_grad():
        arch = {}
        depth = depths
        width = widths
        for i in range(stage):
            stage_dict = {
                'macro': {
                    'depth': depth[i],
                    'width': width[i],
                    'ratio': ratios[i]
                },
                'micro': {
                    "use_layer_scale": [True, True, True],  # [True, False, True]
                    "is_drop": [True, True, True],  # [True, False, True]
                }
            }
            skip_weightss = skip_weightsss[i]
            norm_op_act_weightsss = norm_op_act_weightssss[i]
            for j in range(multiplier):
                norm_op_act_weightss = norm_op_act_weightsss[j]
                act0_id = norm_op_act_weightss[0].argmax().item()
                norm1_id = norm_op_act_weightss[1].argmax().item()
                op2_id = norm_op_act_weightss[2].argmax().item()
                norm3_id = norm_op_act_weightss[3].argmax().item()
                act4_id = norm_op_act_weightss[4].argmax().item()
                stage_dict['micro'][str(j + 1)] = '{}-{}-{}-{}-{}'.format(
                    CONFIG['normopact_module']['act'][act0_id],
                    CONFIG['normopact_module']['norm'][norm1_id],
                    CONFIG['normopact_module']['bread_op' if j != 1 else 'meat_op'][op2_id],
                    CONFIG['normopact_module']['norm'][norm3_id],
                    CONFIG['normopact_module']['act'][act4_id],
                )
                if stage_dict['micro'][str(j + 1)] == 'skip-skip-skip-skip-skip' and j != 1:
                    stage_dict['micro']["use_layer_scale"][j] = False
                    stage_dict['micro']["is_drop"][j] = False
            pos = 0
            for j in range(1, multiplier + 1):
                for k in range(j):
                    idx = skip_weightss[pos].argmax().item()
                    pos += 1
                    stage_dict['micro']['{}->{}'.format(k, j)] = CONFIG['skip_module'][idx]
            arch['stage-{}'.format(i + 1)] = stage_dict

        return arch


def arch2_weight(arch, device, stage=4, multiplier=3, has_atten_mlp=CONFIG['normopact_module']['has_atten_mlp']):
    skip_weightsss, norm_op_act_weightssss = [], []
    depths, widths, ratios = [], [], []

    skip_edge_num = sum([i + 1 for i in range(multiplier)])
    skip_op_num = len(CONFIG['skip_module'])
    norm_op_num = len(CONFIG['normopact_module']['norm'])
    bread_op_num = len(CONFIG['normopact_module']['bread_op'])
    meat_op_num = len(CONFIG['normopact_module']['meat_op'])
    act_op_num = len(CONFIG['normopact_module']['act'])

    for i in range(stage):
        skip_weightss = torch.zeros(skip_edge_num, skip_op_num).to(device)
        norm_op_act_weightsss = []

        depths.append(arch['stage-{}'.format(i + 1)]['macro']['depth'])
        widths.append(arch['stage-{}'.format(i + 1)]['macro']['width'])
        ratios.append(arch['stage-{}'.format(i + 1)]['macro']['ratio'])
        micro_arch = arch['stage-{}'.format(i + 1)]['micro']
        pos = 0
        for j in range(multiplier):
            if j == 1:
                op_op_num = meat_op_num
            elif not has_atten_mlp[i]:
                op_op_num = bread_op_num - 2
            else:
                op_op_num = bread_op_num
            norm_op_act_weightsss.append([
                torch.zeros(act_op_num).to(device),
                torch.zeros(norm_op_num).to(device),
                torch.zeros(op_op_num).to(device),
                torch.zeros(norm_op_num).to(device),
                torch.zeros(act_op_num).to(device),
            ])

            ops = micro_arch['{}'.format(j + 1)].split('-')
            act0_id = CONFIG['normopact_module']['act'].index(ops[0])
            norm1_id = CONFIG['normopact_module']['norm'].index(ops[1])
            op2_id = CONFIG['normopact_module']['bread_op' if j != 1 else 'meat_op'].index(ops[2])
            norm3_id = CONFIG['normopact_module']['norm'].index(ops[3])
            act4_id = CONFIG['normopact_module']['act'].index(ops[4])

            norm_op_act_weightsss[-1][0][act0_id] = 1
            norm_op_act_weightsss[-1][1][norm1_id] = 1
            norm_op_act_weightsss[-1][2][op2_id] = 1
            norm_op_act_weightsss[-1][3][norm3_id] = 1
            norm_op_act_weightsss[-1][4][act4_id] = 1

            for k in range(j + 1):
                key = '{}->{}'.format(k, j + 1)
                idx = CONFIG['skip_module'].index(micro_arch[key])
                skip_weightss[pos][idx] = 1
                pos += 1

        skip_weightsss.append(skip_weightss)
        norm_op_act_weightssss.append(norm_op_act_weightsss)

    return skip_weightsss, norm_op_act_weightssss, depths, widths, ratios


def arch_generate_multipath(
    device,
    generate_micro=True,
    generate_macro=True,
    stage=4,
    multiplier=3,
    has_atten_mlp=CONFIG['normopact_module']['has_atten_mlp'],
    search_residual=False,
    pre_defind_arch=None,
    min_macro=False,
    max_macro=False,
    path_num=2,
):
    with torch.no_grad():
        arch_weights = []
        for _ in range(path_num):
            arch_weights.append(arch_generate(
                device,
                generate_micro,
                generate_macro,
                stage,
                multiplier,
                has_atten_mlp,
                search_residual,
                pre_defind_arch,
                min_macro,
                max_macro,
            ))
        base_weight = arch_weights[0]
        for i in range(1, path_num):
            for j in range(len(base_weight[0])):
                base_weight[0][j] = torch.sign(base_weight[0][j] + arch_weights[i][0][j])
            for j in range(len(base_weight[1])):
                for k in range(len(base_weight[1][0])):
                    for p in range(len(base_weight[1][0][0])):
                        base_weight[1][j][k][p] = (base_weight[1][j][k][p] * i + arch_weights[i][1][j][k][p]) / (i + 1)
        return base_weight


def del_vars(skip_weightsss, norm_op_act_weightssss, depths, widths, ratios):
    print(torch.cuda.memory_allocated())
    for skip_weightss in skip_weightsss:
        del skip_weightss
    for norm_op_act_weightsss in norm_op_act_weightssss:
        for norm_op_act_weightss in norm_op_act_weightsss:
            for norm_op_act_weights in norm_op_act_weightss:
                del norm_op_act_weights
    del depths
    del widths
    del ratios
    print(torch.cuda.memory_allocated())


if __name__ == "__main__":
    # torch.manual_seed(3)

    print(arch_generate_multipath(torch.device('cpu'), path_num=1)[2])
    a = 1
    exit()

    num = 1
    count = 0
    for i in range(num):
        device = torch.device('cpu')
        import time
        t = time.time()
        skip_weightsss, norm_op_act_weightssss, depths, widths, ratios = arch_generate(device, max_macro=True)
        t = time.time() - t
        print(skip_weightsss)
        print(norm_op_act_weightssss)
        print(depths)
        print(widths)
        print(ratios, '\n')
        print(is_good_arch([skip_weightsss, norm_op_act_weightssss, depths, widths, ratios]))
        arch_str = weight2_arch(skip_weightsss, norm_op_act_weightssss, depths, widths, ratios)
        print(arch_str, '\n')
        skip_weightsss, norm_op_act_weightssss, depths, widths, ratios = arch2_weight(arch_str, device)
        origin_arch_str = arch_str
        arch_str = weight2_arch(skip_weightsss, norm_op_act_weightssss, depths, widths, ratios)
        # print(skip_weightsss)
        # print(norm_op_act_weightssss)
        # print(depths)
        # print(widths)
        # print(ratios, '\n')
        print('Equal: {}'.format(str(origin_arch_str) == str(arch_str)))
        print(is_good_arch([skip_weightsss, norm_op_act_weightssss, depths, widths, ratios]))
        print(t * 1000, 'ms')

        if arch_str['stage-1']['micro']['1'] == 'skip-skip-skip-skip-skip':
            count += 1

    print(count / num)

    poolformer_s12_real_2 = {
        "stage-1": {
            "micro": {
                "1": "skip-GN-avgpool-skip-skip",
                "2": "skip-skip-skip-skip-skip",
                "3": "skip-skip-skip-skip-skip",
                "0->1": "skip",
                "0->2": "none",
                "1->2": "skip",
                "0->3": "none",
                "1->3": "none",
                "2->3": "none",
                "use_layer_scale": [True, True, False],
                "is_drop": [True, True, False],
            },
            "macro": {
                "depth": 2,
                "width": 64,
                "ratio": 4,
            }
        },
        "stage-2": {
            "micro": {
                "1": "skip-GN-avgpool-skip-skip",
                "2": "skip-skip-skip-skip-skip",
                "3": "skip-skip-skip-skip-skip",
                "0->1": "skip",
                "0->2": "none",
                "1->2": "skip",
                "0->3": "none",
                "1->3": "none",
                "2->3": "none",
                "use_layer_scale": [True, True, False],
                "is_drop": [True, True, False],
            },
            "macro": {
                "depth": 2,
                "width": 128,
                "ratio": 4,
            }
        },
        "stage-3": {
            "micro": {
                "1": "skip-GN-avgpool-skip-skip",
                "2": "skip-skip-skip-skip-skip",
                "3": "skip-skip-skip-skip-skip",
                "0->1": "skip",
                "0->2": "none",
                "1->2": "skip",
                "0->3": "none",
                "1->3": "none",
                "2->3": "none",
                "use_layer_scale": [True, True, False],
                "is_drop": [True, True, False],
            },
            "macro": {
                "depth": 6,
                "width": 320,
                "ratio": 4,
            }
        },
        "stage-4": {
            "micro": {
                "1": "skip-GN-avgpool-skip-skip",
                "2": "skip-skip-skip-skip-skip",
                "3": "skip-skip-skip-skip-skip",
                "0->1": "skip",
                "0->2": "none",
                "1->2": "skip",
                "0->3": "none",
                "1->3": "none",
                "2->3": "none",
                "use_layer_scale": [True, True, False],
                "is_drop": [True, True, False],
            },
            "macro": {
                "depth": 2,
                "width": 512,
                "ratio": 4,
            }
        },
    }
    skip_weightsss, norm_op_act_weightssss, depths, widths, ratios = arch_generate(device, generate_micro=True, generate_macro=False, pre_defind_arch=poolformer_s12_real_2)
    arch_str = weight2_arch(skip_weightsss, norm_op_act_weightssss, depths, widths, ratios)
    print(arch_str)
