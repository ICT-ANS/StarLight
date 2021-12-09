# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
import numpy as np
import torch
from .weight_masker import WeightMasker

__all__ = ['L1FilterPrunerMasker', 'L2FilterPrunerMasker', 'FPGMPrunerMasker',
           'TaylorFOWeightFilterPrunerMasker',]

logger = logging.getLogger('torch filter pruners')


class StructuredWeightMasker(WeightMasker):
    """
    A structured pruning masker base class that prunes convolutional layer filters.

    Parameters
    ----------
    model: nn.Module
        model to be pruned
    pruner: Pruner
        A Pruner instance used to prune the model
    preserve_round: int
        after pruning, preserve filters/channels round to `preserve_round`, for example:
        for a Conv2d layer, output channel is 32, sparsity is 0.2, if preserve_round is
        1 (no preserve round), then there will be int(32 * 0.2) = 6 filters pruned, and
        32 - 6 = 26 filters are preserved. If preserve_round is 4, preserved filters will
        be round up to 28 (which can be divided by 4) and only 4 filters are pruned.

    """

    def __init__(self, model, pruner, preserve_round=1, dependency_aware=False):
        self.model = model
        self.pruner = pruner
        self.preserve_round = preserve_round
        self.dependency_aware = dependency_aware

    def calc_mask(self, sparsity, wrapper, wrapper_idx=None, **depen_kwargs):
        """
        calculate the mask for `wrapper`.

        Parameters
        ----------
        sparsity: float/list of float
            The target sparsity of the wrapper. If we calculate the mask in
            the normal way, then sparsity is a float number. In contrast, if
            we calculate the mask in the dependency-aware way, sparsity is a
            list of float numbers, each float number corressponds to a sparsity
            of a layer.
        wrapper: PrunerModuleWrapper/list of PrunerModuleWrappers
            The wrapper of the target layer. If we calculate the mask in the normal
            way, then `wrapper` is an instance of PrunerModuleWrapper, else `wrapper`
            is a list of PrunerModuleWrapper.
        wrapper_idx: int/list of int
            The index of the wrapper.
        depen_kwargs: dict
            The kw_args for the dependency-aware mode.
        """
        if not self.dependency_aware:
            # calculate the mask in the normal way, each layer calculate its
            # own mask separately
            return self._normal_calc_mask(sparsity, wrapper, wrapper_idx)
        else:
            # if the dependency_aware switch is on, then calculate the mask
            # in the dependency-aware way
            return self._dependency_calc_mask(sparsity, wrapper, wrapper_idx, **depen_kwargs)

    def _get_current_state(self, sparsity, wrapper, wrapper_idx=None):
        """
        Some pruner may prune the layers in a iterative way. In each pruning iteration,
        we may get the current state of this wrapper/layer, and continue to prune this layer
        based on the current state. This function is to get the current pruning state of the
        target wrapper/layer.
        Parameters
        ----------
        sparsity: float
            pruning ratio,  preserved weight ratio is `1 - sparsity`
        wrapper: PrunerModuleWrapper
            layer wrapper of this layer
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        Returns
        -------
        base_mask: dict
            dict object that stores the mask of this wrapper in this iteration, if it is the
            first iteration, then we create a new mask with all ones. If there is already a
            mask in this wrapper, then we return the existing mask.
        weight: tensor
            the current weight of this layer
        num_prune: int
            how many filters we should prune
        """
        msg = 'module type {} is not supported!'.format(wrapper.type)
        assert wrapper.type == 'Conv2d', msg
        weight = wrapper.module.weight.data
        bias = None
        if hasattr(wrapper.module, 'bias') and wrapper.module.bias is not None:
            bias = wrapper.module.bias.data

        if wrapper.weight_mask is None:
            mask_weight = torch.ones(weight.size()).type_as(weight).detach()
        else:
            mask_weight = wrapper.weight_mask.clone()
        if bias is not None:
            if wrapper.bias_mask is None:
                mask_bias = torch.ones(bias.size()).type_as(bias).detach()
            else:
                mask_bias = wrapper.bias_mask.clone()
        else:
            mask_bias = None
        mask = {'weight_mask': mask_weight, 'bias_mask': mask_bias}

        num_total = weight.size(0)
        num_prune = int(num_total * sparsity)
        if self.preserve_round > 1:
            num_preserve = num_total - num_prune
            num_preserve = int(
                math.ceil(num_preserve * 1. / self.preserve_round) * self.preserve_round)
            if num_preserve > num_total:
                num_preserve = int(math.floor(
                    num_total * 1. / self.preserve_round) * self.preserve_round)
            num_prune = num_total - num_preserve
        # weight*mask_weight: apply base mask for iterative pruning
        return mask, weight * mask_weight, num_prune

    def _normal_calc_mask(self, sparsity, wrapper, wrapper_idx=None):
        """
        Calculate the mask of given layer.
        Parameters
        ----------
        sparsity: float
            pruning ratio,  preserved weight ratio is `1 - sparsity`
        wrapper: PrunerModuleWrapper
            layer wrapper of this layer
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        Returns
        -------
        dict
            dictionary for storing masks, keys of the dict:
            'weight_mask':  weight mask tensor
            'bias_mask': bias mask tensor (optional)
        """
        mask, weight, num_prune = self._get_current_state(
            sparsity, wrapper, wrapper_idx)
        num_total = weight.size(0)
        if num_total < 2 or num_prune < 1:
            return mask

        return self.get_mask(mask, weight, num_prune, wrapper, wrapper_idx)

    def _common_channel_to_prune(self, sparsities, wrappers, wrappers_idx, channel_dsets, groups):
        """
        Calculate the common channels should be pruned by all the layers in this group.
        This function is for filter pruning of Conv layers. if want to support the dependency-aware
        mode for others ops, you need to inherit this class and overwrite `_common_channel_to_prune`.

        Parameters
        ----------
        sparsities : list
            List of float that specify the sparsity for each conv layer.
        wrappers : list
            List of wrappers
        groups : list
            The number of the filter groups of each layer.
        wrappers_idx : list
            The indexes of the wrappers
        """
        # sparsity configs for each wrapper
        # sparsities = [_w.config['sparsity'] for _w in wrappers]
        # check the type of the input wrappers
        for _w in wrappers:
            msg = 'module type {} is not supported!'.format(_w.type)
            assert _w.type == 'Conv2d', msg
        # Among the dependent layers, the layer with smallest
        # sparsity determines the final benefit of the speedup
        # module. To better harvest the speed benefit, we need
        # to ensure that these dependent layers have at least
        # `min_sparsity` pruned channel are the same.
        if len(channel_dsets) == len(wrappers):
            # all the layers in the dependency sets are pruned
            min_sparsity = min(sparsities)
        else:
            # not all the layers in the dependency set
            # are pruned
            min_sparsity = 0
        # donnot prune the channels that we cannot harvest the speed from
        sparsities = [min_sparsity] * len(sparsities)
        # find the max number of the filter groups of the dependent
        # layers. The group constraint of this dependency set is decided
        # by the layer with the max groups.

        # should use the least common multiple for all the groups
        # the max_group is lower than the channel_count, because
        # the number of the filter is always divisible by the number of the group
        max_group = np.lcm.reduce(groups)
        channel_count = wrappers[0].module.weight.data.size(0)
        device = wrappers[0].module.weight.device
        channel_sum = torch.zeros(channel_count).to(device)
        for _w, _w_idx in zip(wrappers, wrappers_idx):
            # calculate the L1/L2 sum for all channels
            c_sum = self.get_channel_sum(_w, _w_idx)

            if c_sum is None:
                # if the channel sum cannot be calculated
                # now, return None
                return None
            channel_sum += c_sum

        # prune the same `min_sparsity` channels based on channel_sum
        # for all the layers in the channel sparsity
        target_pruned = int(channel_count * min_sparsity)
        # pruned_per_group may be zero, for example dw conv
        pruned_per_group = int(target_pruned / max_group)
        group_step = int(channel_count / max_group)

        channel_masks = []
        for gid in range(max_group):
            _start = gid * group_step
            _end = (gid + 1) * group_step
            if pruned_per_group > 0:
                threshold = torch.topk(
                    channel_sum[_start: _end], pruned_per_group, largest=False)[0].max()
                group_mask = torch.gt(channel_sum[_start:_end], threshold)
            else:
                group_mask = torch.ones(group_step).to(device)
            channel_masks.append(group_mask)
        channel_masks = torch.cat(channel_masks, dim=0)
        pruned_channel_index = (
            channel_masks == False).nonzero().squeeze(1).tolist()
        logger.info('Prune the %s channels for all dependent',
                    ','.join([str(x) for x in pruned_channel_index]))
        return channel_masks

    def _dependency_calc_mask(self, sparsities, wrappers, wrappers_idx, channel_dsets, groups):
        """
        Calculate the masks for the layers in the same dependency sets.
        Similar to the traditional original calc_mask, _dependency_calc_mask
        will prune the target layers based on the L1/L2 norm of the weights.
        However, StructuredWeightMasker prunes the filter completely based on the
        L1/L2 norm of each filter. In contrast, _dependency_calc_mask
        will try to satisfy the channel/group dependency(see nni.compression.torch.
        utils.shape_dependency for details). Specifically, _dependency_calc_mask
        will try to prune the same channels for the layers that have channel dependency.
        In addition, this mask calculator will also ensure that the number of filters
        pruned in each group is the same(meet the group dependency).

        Parameters
        ----------
        sparsities : list
            List of float that specify the sparsity for each conv layer.
        wrappers : list
            List of wrappers
        groups : list
            The number of the filter groups of each layer.
        wrappers_idx : list
            The indexes of the wrappers
        """
        channel_masks = self._common_channel_to_prune(
            sparsities, wrappers, wrappers_idx, channel_dsets, groups)
        # calculate the mask for each layer based on channel_masks, first
        # every layer will prune the same channels masked in channel_masks.
        # If the sparsity of a layers is larger than min_sparsity, then it
        # will continue prune sparsity - min_sparsity channels to meet the sparsity
        # config.
        masks = {}
        for _pos, _w in enumerate(wrappers):
            _w_idx = wrappers_idx[_pos]
            sparsity = sparsities[_pos]
            name = _w.name

            # _tmp_mask = self._normal_calc_mask(
            #     sparsity, _w, _w_idx, channel_masks)
            base_mask, current_weight, num_prune = self._get_current_state(
                sparsity, _w, _w_idx)
            num_total = current_weight.size(0)
            if num_total < 2 or num_prune < 1:
                masks[name] = base_mask
                continue
            _tmp_mask = self.get_mask(
                base_mask, current_weight, num_prune, _w, _w_idx, channel_masks)

            if _tmp_mask is None:
                # if the mask calculation fails
                return None
            masks[name] = _tmp_mask
        return masks

    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
        """
        Calculate the mask of given layer.

        Parameters
        ----------
        base_mask: dict
            The basic mask with the same shape of weight, all item in the basic mask is 1.
        weight: tensor
            the module weight to be pruned
        num_prune: int
            Num of filters to prune
        wrapper: PrunerModuleWrapper
            layer wrapper of this layer
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        channel_masks: Tensor
            If mask some channels for this layer in advance. In the dependency-aware
            mode, before calculating the masks for each layer, we will calculate a common
            mask for all the layers in the dependency set. For the pruners that doesnot
            support dependency-aware mode, they can just ignore this parameter.

        Returns
        -------
        dict
            dictionary for storing masks
        """
        raise NotImplementedError(
            '{} get_mask is not implemented'.format(self.__class__.__name__))

    def get_channel_sum(self, wrapper, wrapper_idx):
        """
        Calculate the importance weight for each channel. If want to support the
        dependency-aware mode for this one-shot pruner, this function must be
        implemented.
        Parameters
        ----------
        wrapper: PrunerModuleWrapper
            layer wrapper of this layer
        wrapper_idx: int
            index of this wrapper in pruner's all wrappers
        Returns
        -------
        tensor
            Tensor that indicates the importance of each channel
        """
        raise NotImplementedError(
            '{} get_channel_sum is not implemented'.format(self.__class__.__name__))


class L1FilterPrunerMasker(StructuredWeightMasker):
    """
    A structured pruning algorithm that prunes the filters of smallest magnitude
    weights sum in the convolution layers to achieve a preset level of network sparsity.
    Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet and Hans Peter Graf,
    "PRUNING FILTERS FOR EFFICIENT CONVNETS", 2017 ICLR
    https://arxiv.org/abs/1608.08710
    """

    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
        # get the l1-norm sum for each filter
        w_abs_structured = self.get_channel_sum(wrapper, wrapper_idx)
        if channel_masks is not None:
            # if we need to mask some channels in advance
            w_abs_structured = w_abs_structured * channel_masks
        threshold = torch.topk(w_abs_structured.view(-1),
                               num_prune, largest=False)[0].max()
        mask_weight = torch.gt(w_abs_structured, threshold)[
            :, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_abs_structured, threshold).type_as(
            weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}

    def get_channel_sum(self, wrapper, wrapper_idx):
        weight = wrapper.module.weight.data
        filters = weight.shape[0]
        w_abs = weight.abs()
        w_abs_structured = w_abs.view(filters, -1).sum(dim=1)
        return w_abs_structured


class L2FilterPrunerMasker(StructuredWeightMasker):
    """
    A structured pruning algorithm that prunes the filters with the
    smallest L2 norm of the weights.
    """

    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
        # get the l2-norm sum for each filter
        w_l2_norm = self.get_channel_sum(wrapper, wrapper_idx)
        if channel_masks is not None:
            # if we need to mask some channels in advance
            w_l2_norm = w_l2_norm * channel_masks
        threshold = torch.topk(
            w_l2_norm.view(-1), num_prune, largest=False)[0].max()
        mask_weight = torch.gt(w_l2_norm, threshold)[
            :, None, None, None].expand_as(weight).type_as(weight)
        mask_bias = torch.gt(w_l2_norm, threshold).type_as(
            weight).detach() if base_mask['bias_mask'] is not None else None

        return {'weight_mask': mask_weight.detach(), 'bias_mask': mask_bias}

    def get_channel_sum(self, wrapper, wrapper_idx):
        weight = wrapper.module.weight.data
        filters = weight.shape[0]
        w = weight.view(filters, -1)
        w_l2_norm = torch.sqrt((w ** 2).sum(dim=1))
        return w_l2_norm


class FPGMPrunerMasker(StructuredWeightMasker):
    """
    A filter pruner via geometric median.
    "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration",
    https://arxiv.org/pdf/1811.00250.pdf
    """

    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
        min_gm_idx = self._get_min_gm_kernel_idx(
            num_prune, wrapper, wrapper_idx, channel_masks)
        for idx in min_gm_idx:
            base_mask['weight_mask'][idx] = 0.
            if base_mask['bias_mask'] is not None:
                base_mask['bias_mask'][idx] = 0.
        return base_mask

    def _get_min_gm_kernel_idx(self, num_prune, wrapper, wrapper_idx, channel_masks):
        channel_dist = self.get_channel_sum(wrapper, wrapper_idx)
        if channel_masks is not None:
            channel_dist = channel_dist * channel_masks
        dist_list = [(channel_dist[i], i)
                     for i in range(channel_dist.size(0))]
        min_gm_kernels = sorted(dist_list, key=lambda x: x[0])[:num_prune]
        return [x[1] for x in min_gm_kernels]

    def _get_distance_sum(self, weight, out_idx):
        """
        Calculate the total distance between a specified filter (by out_idex and in_idx) and
        all other filters.
        Parameters
        ----------
        weight: Tensor
            convolutional filter weight
        out_idx: int
            output channel index of specified filter, this method calculates the total distance
            between this specified filter and all other filters.
        Returns
        -------
        float32
            The total distance
        """
        logger.debug('weight size: %s', weight.size())
        assert len(weight.size()) in [3, 4], 'unsupported weight shape'

        w = weight.view(weight.size(0), -1)
        anchor_w = w[out_idx].unsqueeze(0).expand(w.size(0), w.size(1))
        x = w - anchor_w
        x = (x * x).sum(-1)
        x = torch.sqrt(x)
        return x.sum()

    def get_channel_sum(self, wrapper, wrapper_idx):
        weight = wrapper.module.weight.data
        assert len(weight.size()) in [3, 4]
        dist_list = []
        for out_i in range(weight.size(0)):
            dist_sum = self._get_distance_sum(weight, out_i)
            dist_list.append(dist_sum)
        return torch.Tensor(dist_list).to(weight.device)


class TaylorFOWeightFilterPrunerMasker(StructuredWeightMasker):
    """
    A structured pruning algorithm that prunes the filters with the smallest
    importance approximations based on the first order taylor expansion on the weight.
    Molchanov, Pavlo and Mallya, Arun and Tyree, Stephen and Frosio, Iuri and Kautz, Jan,
    "Importance Estimation for Neural Network Pruning", CVPR 2019.
    http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf
    """

    def __init__(self, model, pruner, statistics_batch_num=1):
        super().__init__(model, pruner)
        self.statistics_batch_num = statistics_batch_num
        self.pruner.iterations = 0
        self.pruner.set_wrappers_attribute("contribution", None)
        self.pruner.patch_optimizer(self.calc_contributions)

    def get_mask(self, base_mask, weight, num_prune, wrapper, wrapper_idx, channel_masks=None):
        channel_contribution = self.get_channel_sum(wrapper, wrapper_idx)
        if channel_contribution is None:
            # iteration is not enough
            return None
        if channel_masks is not None:
            channel_contribution = channel_contribution * channel_masks
        prune_indices = torch.argsort(channel_contribution)[:num_prune]
        for idx in prune_indices:
            base_mask['weight_mask'][idx] = 0.
            if base_mask['bias_mask'] is not None:
                base_mask['bias_mask'][idx] = 0.
        return base_mask

    def calc_contributions(self):
        """
        Calculate the estimated importance of filters as a sum of individual contribution
        based on the first order taylor expansion.
        """
        if self.pruner.iterations >= self.statistics_batch_num:
            return

        for wrapper in self.pruner.get_modules_wrapper():
            filters = wrapper.module.weight.size(0)
            contribution = (
                wrapper.module.weight * wrapper.module.weight.grad).data.pow(2).view(filters, -1).sum(dim=1)
            if wrapper.contribution is None:
                wrapper.contribution = contribution
            else:
                wrapper.contribution += contribution

        self.pruner.iterations += 1

    def get_channel_sum(self, wrapper, wrapper_idx):
        if self.pruner.iterations < self.statistics_batch_num:
            return None
        if wrapper.contribution is None:
            return None
        return wrapper.contribution



