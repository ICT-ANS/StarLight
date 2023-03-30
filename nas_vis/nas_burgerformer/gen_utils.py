'''
    This file defines functions used to generate network_def for evolutionary search.
    
    Network def looks like:
    [
        tensor([[1., 0.], # stage 1 skip
                [0., 1.],
                [1., 0.],
                [1., 0.],
                [0., 1.],
                [0., 1.]]), 
        tensor([[1., 0.], # stage 2 skip
                [0., 1.],
                [0., 1.],
                [0., 1.],
                [1., 0.],
                [0., 1.]]), 
        tensor([[1., 0.], # stage 3 skip
                [1., 0.],
                [1., 0.],
                [0., 1.],
                [1., 0.],
                [1., 0.]]),
        tensor([[1., 0.], # stage 4 skip
                [0., 1.],
                [1., 0.],
                [1., 0.],
                [1., 0.],
                [1., 0.]])
    ]
    [
        [ # stage 1 act-norm-mixer-norm-act
            [tensor([0., 0., 0., 1.]), tensor([0., 0., 0., 1.]), tensor([0., 1., 0.]), tensor([1., 0., 0., 0.]), tensor([0., 0., 1., 0.])], 
            [tensor([0., 0., 0., 1.]), tensor([1., 0., 0., 0.]), tensor([0., 0., 1.]), tensor([0., 0., 0., 1.]), tensor([0., 0., 0., 1.])], 
            [tensor([0., 0., 0., 1.]), tensor([0., 1., 0., 0.]), tensor([0., 1., 0.]), tensor([0., 0., 0., 1.]), tensor([0., 1., 0., 0.])]
        ], 
        [ # stage 2 act-norm-mixer-norm-act
            [tensor([0., 0., 0., 1.]), tensor([0., 0., 0., 1.]), tensor([1., 0., 0.]), tensor([0., 0., 1., 0.]), tensor([1., 0., 0., 0.])], 
            [tensor([0., 0., 0., 1.]), tensor([0., 0., 0., 1.]), tensor([1., 0., 0.]), tensor([1., 0., 0., 0.]), tensor([0., 1., 0., 0.])], 
            [tensor([0., 0., 0., 1.]), tensor([0., 1., 0., 0.]), tensor([0., 0., 1.]), tensor([0., 0., 0., 1.]), tensor([0., 0., 1., 0.])]], 
        [ # stage 3 act-norm-mixer-norm-act
            [tensor([0., 0., 0., 1.]), tensor([0., 0., 0., 1.]), tensor([0., 0., 1., 0., 0.]), tensor([0., 0., 1., 0.]), tensor([1., 0., 0., 0.])], 
            [tensor([0., 0., 0., 1.]), tensor([0., 1., 0., 0.]), tensor([0., 0., 0., 0., 1.]), tensor([0., 0., 0., 1.]), tensor([0., 0., 0., 1.])], 
            [tensor([0., 0., 1., 0.]), tensor([0., 1., 0., 0.]), tensor([0., 0., 0., 1., 0.]), tensor([0., 0., 0., 1.]), tensor([0., 0., 0., 1.])]], 
        [ # stage 4 act-norm-mixer-norm-act
            [tensor([0., 0., 1., 0.]), tensor([0., 0., 0., 1.]), tensor([0., 0., 0., 0., 1.]), tensor([0., 0., 1., 0.]), tensor([0., 0., 0., 1.])], 
            [tensor([0., 0., 0., 1.]), tensor([0., 0., 0., 1.]), tensor([0., 0., 1., 0., 0.]), tensor([0., 0., 1., 0.]), tensor([0., 0., 0., 1.])], 
            [tensor([1., 0., 0., 0.]), tensor([0., 1., 0., 0.]), tensor([0., 0., 0., 0., 1.]), tensor([0., 0., 0., 1.]), tensor([0., 0., 0., 1.])]
        ]
    ]
'''
import numpy as np
import copy
from nas_vis.nas_burgerformer.arch_generate import *
import arch
import math
from nas_vis.nas_burgerformer.cal_flops_params import cal_flops_parameters


def satisfy_target_res(flops, params, target, target_flops, target_params, thr):
    if target is None:
        return True
    elif target == 'flops':
        return flops >= thr * target_flops and flops <= target_flops
    elif target == 'params':
        return params >= thr * target_params and params <= target_params
    elif target == 'flops_params':
        return flops >= thr * target_flops and flops <= target_flops and params >= thr * target_params and params <= target_params
    else:
        raise NotImplementedError


def cal_dis_to_target_res(flops, params, target, target_flops, target_params):
    if target is None:
        return 0
    elif target == 'flops':
        return 1 - min(flops, target_flops) / max(flops, target_flops)
    elif target == 'params':
        return 1 - min(params, target_params) / max(params, target_params)
    elif target == 'flops_params':
        return 0.5 * (2 - min(flops, target_flops) / max(flops, target_flops) - min(params, target_params) / max(params, target_params))
    else:
        raise NotImplementedError


def mutate_func(parent_network_def, m_prob, device, only_micro, only_macro, pre_defined_arch='', target='flops', target_flops=1.8e9, target_params=12e6, thr=0.975):
    max_loop_num = 100
    count = 0
    min_dis = 1e16
    min_network_def = None
    while count < max_loop_num:
        network_def = copy.deepcopy(parent_network_def)
        skip_weightsss, norm_op_act_weightssss, depths, widths, ratios = arch_generate(device,
                                                                                       generate_micro=not only_macro,
                                                                                       generate_macro=not only_micro,
                                                                                       pre_defind_arch=None if pre_defined_arch == '' else eval('arch.{}'.format(pre_defined_arch)))
        tmp_network_def = [skip_weightsss, norm_op_act_weightssss, depths, widths, ratios]
        skip_weightsss, norm_op_act_weightssss, depths, widths, ratios = network_def
        tmp_skip_weightsss, tmp_norm_op_act_weightssss, tmp_depths, tmp_widths, tmp_ratios = tmp_network_def

        for skip_weightss, tmp_skip_weightss in zip(skip_weightsss, tmp_skip_weightsss):
            for edge in range(skip_weightss.size(0)):
                if np.random.uniform() <= m_prob:
                    skip_weightss[edge] = tmp_skip_weightss[edge]

        for norm_op_act_weightsss, tmp_norm_op_act_weightsss in zip(norm_op_act_weightssss, tmp_norm_op_act_weightssss):
            for norm_op_act_weightss, tmp_norm_op_act_weightss in zip(norm_op_act_weightsss, tmp_norm_op_act_weightsss):
                if np.random.uniform() <= m_prob:
                    for norm_op_act_weights, tmp_norm_op_act_weights in zip(norm_op_act_weightss, tmp_norm_op_act_weightss):
                        norm_op_act_weights.data = tmp_norm_op_act_weights.data

        for i in range(len(depths)):
            if np.random.uniform() <= m_prob:
                depths[i] = tmp_depths[i]

        for i in range(len(widths)):
            if np.random.uniform() <= m_prob:
                widths[i] = tmp_widths[i]

        for i in range(len(ratios)):
            if np.random.uniform() <= m_prob:
                ratios[i] = tmp_ratios[i]

        if is_good_arch(network_def):
            count += 1
        else:
            continue

        new_net_config = weight2_arch(network_def[0], network_def[1], network_def[2], network_def[3], network_def[4])
        flops, params = cal_flops_parameters(new_net_config)
        if satisfy_target_res(flops, params, target, target_flops, target_params, thr):
            min_network_def = network_def
            break
        else:
            dis = cal_dis_to_target_res(flops, params, target, target_flops, target_params)
            if dis < min_dis:
                min_dis = dis
                min_network_def = network_def
            if min_dis < 0.1:  # boost search
                break

    return min_network_def

    # if target == 'none':
    #     res = None
    # elif target == 'flops':
    #     res = flops
    #     target_res = target_flops
    # elif target == 'params':
    #     res = params
    #     target_res = target_params
    # else:
    #     raise NotImplementedError
    # if res is None:
    #     if is_good_arch(network_def):
    #         min_network_def = network_def
    #         break
    #     elif count >= max_loop_num:
    #         if min_network_def is None:
    #             min_network_def = network_def
    #         break
    #     else:
    #         count += 1
    # elif is_good_arch(network_def) and res >= thr * target_res and res <= target_res:
    #     min_network_def = network_def
    #     break
    # elif count >= max_loop_num:
    #     if min_network_def is None:
    #         min_network_def = network_def
    #     break
    # else:
    #     count += 1
    #     dis = math.fabs(target_res - res)
    #     if dis < min_dis:
    #         min_network_def = network_def
    #         min_dis = dis


def mutate_network_def(parent_network_def, m_prob, device, only_micro, only_macro, pre_defined_arch, target, target_flops, target_params, thr):
    '''
        Call mutate_func() to mutate network_def and check whether 
        the generated network_def has too low resource.
    '''
    m_network_def = mutate_func(parent_network_def, m_prob, device, only_micro, only_macro, pre_defined_arch, target, target_flops, target_params, thr)

    return m_network_def


def crossover_func(m_network_def, f_network_def, m_prob, only_micro, only_macro, pre_defined_arch, target='flops', target_flops=1.8e9, target_params=12e6, thr=0.975):
    max_loop_num = 100
    count = 0
    min_dis = 1e16
    min_network_def = None
    while count < max_loop_num:
        network_def = copy.deepcopy(m_network_def)
        tmp_network_def = copy.deepcopy(f_network_def)
        skip_weightsss, norm_op_act_weightssss, depths, widths, ratios = network_def
        tmp_skip_weightsss, tmp_norm_op_act_weightssss, tmp_depths, tmp_widths, tmp_ratios = tmp_network_def

        for skip_weightss, tmp_skip_weightss in zip(skip_weightsss, tmp_skip_weightsss):
            for edge in range(skip_weightss.size(0)):
                if np.random.uniform() <= m_prob:
                    skip_weightss[edge] = tmp_skip_weightss[edge]

        for norm_op_act_weightsss, tmp_norm_op_act_weightsss in zip(norm_op_act_weightssss, tmp_norm_op_act_weightssss):
            for norm_op_act_weightss, tmp_norm_op_act_weightss in zip(norm_op_act_weightsss, tmp_norm_op_act_weightsss):
                if np.random.uniform() <= m_prob:
                    for norm_op_act_weights, tmp_norm_op_act_weights in zip(norm_op_act_weightss, tmp_norm_op_act_weightss):
                        norm_op_act_weights.data = tmp_norm_op_act_weights.data

        for i in range(len(depths)):
            if np.random.uniform() <= m_prob:
                depths[i] = tmp_depths[i]

        for i in range(len(widths)):
            if np.random.uniform() <= m_prob:
                widths[i] = tmp_widths[i]

        for i in range(len(ratios)):
            if np.random.uniform() <= m_prob:
                ratios[i] = tmp_ratios[i]

        if is_good_arch(network_def):
            count += 1
        else:
            continue

        new_net_config = weight2_arch(network_def[0], network_def[1], network_def[2], network_def[3], network_def[4])
        flops, params = cal_flops_parameters(new_net_config)
        if satisfy_target_res(flops, params, target, target_flops, target_params, thr):
            min_network_def = network_def
            break
        else:
            dis = cal_dis_to_target_res(flops, params, target, target_flops, target_params)
            if dis < min_dis:
                min_dis = dis
                min_network_def = network_def
            if min_dis < 0.1:  # boost search
                break

    return min_network_def

    #     count += 1
    #     new_net_config = weight2_arch(network_def[0], network_def[1], network_def[2], network_def[3], network_def[4])
    #     flops, params = cal_flops_parameters(new_net_config)
    #     if target == 'none':
    #         res = None
    #     elif target == 'flops':
    #         res = flops
    #         target_res = target_flops
    #     elif target == 'params':
    #         res = params
    #         target_res = target_params
    #     else:
    #         raise NotImplementedError
    #     if res is None:
    #         if is_good_arch(network_def):
    #             min_network_def = network_def
    #             break
    #         elif count >= max_loop_num:
    #             if min_network_def is None:
    #                 min_network_def = network_def
    #             break
    #         else:
    #             count += 1
    #     elif is_good_arch(network_def) and res >= thr * target_res and res <= target_res:
    #         min_network_def = network_def
    #         break
    #     elif count >= max_loop_num:
    #         if min_network_def is None:
    #             min_network_def = network_def
    #         break
    #     else:
    #         count += 1
    #         dis = math.fabs(target_res - res)
    #         if dis < min_dis:
    #             min_network_def = network_def
    #             min_dis = dis

    # return min_network_def


def crossover_network_def(m_network_def, f_network_def, m_prob, device, only_micro, only_macro, pre_defined_arch, target, target_flops, target_params, thr):
    '''
        Call crossover_func() to generate network_def and check whether 
        the generated network_def has too low resource.
    '''
    c_network_def = crossover_func(m_network_def, f_network_def, m_prob, only_micro, only_macro, pre_defined_arch, target, target_flops, target_params, thr)

    return c_network_def


if __name__ == '__main__':
    pass
