from gen_utils import mutate_network_def, crossover_network_def, satisfy_target_res, cal_dis_to_target_res

import numpy as np
import warnings

import math
import arch
from nas_vis.nas_burgerformer.arch_generate import *
from nas_vis.nas_burgerformer.cal_flops_params import cal_flops_parameters

_CROSSOVER_SKIP_CHECKING_THRESHOLD = 100


class Individual():
    def __init__(self, skip_weightsss, norm_op_act_weightssss, depths, widths, ratios, score=-1):
        self.network_def = [skip_weightsss, norm_op_act_weightssss, depths, widths, ratios]
        self.score = score
        self.network_str = str(weight2_arch(skip_weightsss, norm_op_act_weightssss, depths, widths, ratios))

    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.network_str == other.network_str

    def __repr__(self):
        return '(network_def={}, score={})'.format(self.network_str, self.score)


# Reference: https://github.com/mit-han-lab/hardware-aware-transformers/blob/c8d6d71903854537d265129bea7c5d162c4ee210/fairseq/evolution.py#L183
class PopulationEvolver():
    def __init__(self, device):
        self.popu = []  # population
        self.history_popu = []  # save all previous networks to prevent evaluating the same networks
        self.device = device

    def random_sample(self, num_samples, only_micro, only_macro, pre_defined_arch='', target='flops', target_flops=1.8e9, target_params=12e6, thr=0.975):
        popu_idx = 0
        skip_checking_counter = 0  # prevent infinite loop when at later iterations crossover does not produce new samples
        skip_checking_threshold = _CROSSOVER_SKIP_CHECKING_THRESHOLD
        while popu_idx < num_samples:
            # if popu_idx > 0:
            #     print(min_dis)
            max_loop_num = 100
            count = 0
            min_dis = 1e16
            min_network_def = None
            while count < max_loop_num:
                skip_weightsss, norm_op_act_weightssss, depths, widths, ratios = arch_generate(self.device,
                                                                                               generate_micro=not only_macro,
                                                                                               generate_macro=not only_micro,
                                                                                               pre_defind_arch=None if pre_defined_arch == '' else eval('arch.{}'.format(pre_defined_arch)))
                network_def = (skip_weightsss, norm_op_act_weightssss, depths, widths, ratios)

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

                # new_net_config = weight2_arch(skip_weightsss, norm_op_act_weightssss, depths, widths, ratios)
                # flops, params = cal_flops_parameters(new_net_config)
                # if target == 'none':
                #     res = None
                # elif target == 'flops':
                #     res = flops
                #     target_res = target_flops
                # elif target == 'params':
                #     res = params
                #     target_res = target_params
                # elif target == 'flops_params':
                #     res = (flops, params)
                #     target_res = (target_flops, target_params)
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

            new_ind = Individual(min_network_def[0], min_network_def[1], min_network_def[2], min_network_def[3], min_network_def[4])
            if (new_ind not in self.popu and new_ind not in self.history_popu) or skip_checking_counter >= skip_checking_threshold:
                self.popu.append(new_ind)
                popu_idx = popu_idx + 1
                skip_checking_counter = 0
            else:
                skip_checking_counter = skip_checking_counter + 1
        return

    def update_history(self):
        for i in range(len(self.popu)):
            if self.popu[i] not in self.history_popu:
                self.history_popu.append(self.popu[i])
        # self.history_popu.extend(self.popu)
        self.popu = []
        return

    def sort_history(self):
        self.history_popu.sort(reverse=True)
        return

    def evolve_sample(self, parent_size, mutate_prob, mutate_size, only_micro, only_macro, pre_defined_arch='', crossover_size=None, target='flops', target_flops=1.8e9, target_params=12e6, thr=0.975):
        if self.popu:
            warnings.warn('[evolve_sample] popu is not empty.')
        if not self.history_popu:
            warnings.warn('[evolve_sample] history_popu is empty. Use update_history() before evolve_sample().')
            return
        if parent_size > len(self.history_popu):
            raise ValueError('Parent size is larger than history population size')

        self.sort_history()
        if crossover_size is None:
            crossover_size = mutate_size

        # mutation
        popu_idx = 0
        skip_checking_counter = 0  # prevent infinite loop when at later iterations crossover does not produce new samples
        skip_checking_threshold = _CROSSOVER_SKIP_CHECKING_THRESHOLD
        while popu_idx < mutate_size:
            parent_idx = np.random.randint(parent_size)
            parent_network_def = self.history_popu[parent_idx].network_def
            network_def = mutate_network_def(parent_network_def, mutate_prob, self.device, only_micro, only_macro, pre_defined_arch, target, target_flops, target_params, thr)
            new_ind = Individual(network_def[0], network_def[1], network_def[2], network_def[3], network_def[4])
            if new_ind not in self.popu and new_ind not in self.history_popu or skip_checking_counter >= skip_checking_threshold:
                self.popu.append(new_ind)
                popu_idx = popu_idx + 1
                skip_checking_counter = 0
            else:
                skip_checking_counter = skip_checking_counter + 1

        # crossover
        popu_idx = 0
        skip_checking_counter = 0  # prevent infinite loop when at later iterations crossover does not produce new samples
        skip_checking_threshold = _CROSSOVER_SKIP_CHECKING_THRESHOLD
        while popu_idx < crossover_size:
            parent_idx = np.random.choice(range(parent_size), size=2, replace=False)
            m_network_def = self.history_popu[parent_idx[0]].network_def
            f_network_def = self.history_popu[parent_idx[1]].network_def
            network_def = crossover_network_def(m_network_def, f_network_def, 0.5, self.device, only_micro, only_macro, pre_defined_arch, target, target_flops, target_params, thr)
            new_ind = Individual(network_def[0], network_def[1], network_def[2], network_def[3], network_def[4])
            if (new_ind not in self.popu and new_ind not in self.history_popu) or skip_checking_counter >= skip_checking_threshold:
                self.popu.append(new_ind)
                popu_idx = popu_idx + 1
                skip_checking_counter = 0
            else:
                skip_checking_counter = skip_checking_counter + 1

        return


if __name__ == '__main__':
    pass