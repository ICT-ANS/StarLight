import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import *
from torch.autograd import Variable
from .genotypes import PRIMITIVES
from .genotypes import Genotype
import pickle
import os


def mask_gumbelsoftmax(vec, mask, tau, dim=-1):
    gumbels = -torch.empty_like(vec).exponential_().log()
    vec = (vec.log_softmax(dim=-1) + gumbels) / tau
    return F.softmax(vec.masked_fill(~mask, float('-inf')), dim=dim)


def mask_softmax(vec, mask, dim=-1):
    return F.softmax(vec.masked_fill(~mask, float('-inf')), dim=dim)


def cal_weight(alphas, betas, masks, type, norm=True, eps=1e-14, tau=None):
    if type == 'simple':
        if tau is not None:
            alphas = mask_gumbelsoftmax(alphas, masks, tau)
        else:
            alphas = mask_softmax(alphas, masks)
        betas = F.sigmoid(betas)
        weights = alphas * betas
        if norm:
            weights = weights / (weights.norm(1, dim=-1, keepdim=True) + eps)
        return weights
    elif type == 'all':
        if tau is not None:
            alphas = mask_gumbelsoftmax(alphas, masks, tau)
        else:
            alphas = mask_softmax(alphas, masks)
        betas = F.sigmoid(betas) * masks.float() + (1 - F.sigmoid(betas)) * (~masks).float()
        weights = alphas * betas.prod(dim=-1, keepdim=True)
        if norm:
            weights = weights / (weights.norm(1, dim=-1, keepdim=True) + eps)
        return weights


class MixedOp(nn.Module):
    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights, masks):
        return sum(w * op(x) for w, op, m in zip(weights, self._ops, masks) if m == 1)


class Cell(nn.Module):
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights, masks):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j], masks[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
    def __init__(self, C, num_classes, layers, criterion, confidence, dss_max_ops, saliency_type, norm=True, steps=4, multiplier=4, stem_multiplier=3, primitives=None):
        super(Network, self).__init__()
        assert saliency_type == 'simple' or saliency_type == 'all'
        assert dss_max_ops >= sum(1 for i in range(steps) for n in range(2 + i))
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._confidence = confidence
        self._dss_max_ops = dss_max_ops
        self._saliency_type = saliency_type
        self._norm = norm
        self._steps = steps
        self._multiplier = multiplier

        if primitives is not None:
            global PRIMITIVES
            PRIMITIVES = primitives

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()
        self._initialize_betas()
        self._initialize_others()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def set_tau(self, tau):
        self.tau = tau

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                masks = self._op_mask_reduce
                weights = cal_weight(self.alphas_reduce, self.betas_reduce, masks, self._saliency_type, self._norm, tau=self.tau)
            else:
                masks = self._op_mask_normal
                weights = cal_weight(self.alphas_normal, self.betas_normal, masks, self._saliency_type, self._norm, tau=self.tau)
            s0, s1 = s1, cell(s0, s1, weights, masks)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def set_space(self, indices, avail=True):
        assert avail is True or avail is False
        self._op_avail_normal[:, indices] = avail
        self._op_avail_reduce[:, indices] = avail

    def shrink_space(self, num):
        # set avail value False based on saliency
        def _shrink(saliency, avail):
            num_ops = len(PRIMITIVES)
            none_index = PRIMITIVES.index('none')

            op_nums_sum = avail.sum(1)

            saliency_v = saliency.view(-1)
            _, indices = torch.sort(saliency_v, 0, False)
            count = 0
            for i in range(saliency_v.size()[0]):
                if count >= num:
                    break
                row, col = indices[i] // num_ops, indices[i] % num_ops
                thr = 2 if avail[row][none_index] is True else 1
                if op_nums_sum[row] > thr and avail[row][col]:
                    avail[row][col].fill_(False)
                    count += 1

        with torch.no_grad():
            _shrink(self._op_saliency_normal, self._op_avail_normal)
            _shrink(self._op_saliency_reduce, self._op_avail_reduce)

    def generate_random_dss(self):
        def _generate(mask, avail):
            k = sum(1 for i in range(self._steps) for n in range(2 + i))
            num_ops = len(PRIMITIVES)

            none_index = PRIMITIVES.index('none')

            def whether_fill(row, col):
                if avail[row][col] == 1:
                    return True
                else:
                    return False

            def whether_count(row, col):
                if col == none_index:
                    mask[row][col].fill_(1)
                    return False
                elif mask[row][col] == 0:
                    return True
                else:
                    return False

            value = torch.rand(mask.size(), device=mask.device)

            mask.fill_(0)
            _, indices1 = torch.sort(value, 1, True)
            for i in range(k):  # select one in each row
                for j in range(num_ops):
                    if whether_fill(i, indices1[i][j]) and whether_count(i, indices1[i][j]):
                        mask[i][indices1[i][j]].fill_(1)
                        break

            value_v = value.view(-1)
            _, indices = torch.sort(value_v, 0, True)
            i, count = 0, 0
            while count < self._dss_max_ops - k and i < len(value_v):  # global select
                row, col = indices[i] // num_ops, indices[i] % num_ops
                if whether_fill(row, col):
                    if whether_count(row, col):
                        count += 1
                    mask[row][col].fill_(1)
                i += 1

        with torch.no_grad():
            _generate(self._op_mask_normal, self._op_avail_normal)
            _generate(self._op_mask_reduce, self._op_avail_reduce)

    def generate_dss(self):
        def _generate(ucb, mask, avail):
            k = sum(1 for i in range(self._steps) for n in range(2 + i))
            num_ops = len(PRIMITIVES)

            none_index = PRIMITIVES.index('none')

            def whether_fill(row, col):
                if avail[row][col] == 1:
                    return True
                else:
                    return False

            def whether_count(row, col):
                if col == none_index:
                    mask[row][col].fill_(1)
                    return False
                elif mask[row][col] == 0:
                    return True
                else:
                    return False

            mask.fill_(0)
            _, indices1 = torch.sort(ucb, 1, True)
            for i in range(k):  # select one in each row
                for j in range(num_ops):
                    if whether_fill(i, indices1[i][j]) and whether_count(i, indices1[i][j]):
                        mask[i][indices1[i][j]].fill_(1)
                        break

            ucb_v = ucb.view(-1)
            _, indices = torch.sort(ucb_v, 0, True)
            i, count = 0, 0
            while count < self._dss_max_ops - k and i < len(ucb_v):  # global select
                row, col = indices[i] // num_ops, indices[i] % num_ops
                if whether_fill(row, col):
                    if whether_count(row, col):
                        count += 1
                    mask[row][col].fill_(1)
                i += 1

        with torch.no_grad():
            _generate(self._ucb_normal, self._op_mask_normal, self._op_avail_normal)
            _generate(self._ucb_reduce, self._op_mask_reduce, self._op_avail_reduce)

    def update_dss(self, eps=1e-14):
        def _update(alphas, betas, mask, saliency, opt_steps, ucb):
            k = sum(1 for i in range(self._steps) for n in range(2 + i))
            num_ops = len(PRIMITIVES)

            weights = cal_weight(alphas, betas, mask, self._saliency_type, norm=False)

            for i in range(k):
                for j in range(num_ops):
                    if mask[i][j] == 1:
                        saliency[i][j].fill_((saliency[i][j] * opt_steps[i][j] + weights[i][j]) / (opt_steps[i][j] + 1))
                        opt_steps[i][j].add_(1)
            # ucb.zero_().add_(saliency / (saliency.norm(1) + eps) + self._confidence * torch.sqrt(self._total_opt_steps.log() / opt_steps))
            ucb.zero_().add_(F.sigmoid(betas) + self._confidence * torch.sqrt(self._total_opt_steps.log() / opt_steps))

        with torch.no_grad():
            _update(self.alphas_normal, self.betas_normal, self._op_mask_normal, self._op_saliency_normal, self._opt_steps_normal, self._ucb_normal)
            _update(self.alphas_reduce, self.betas_reduce, self._op_mask_reduce, self._op_saliency_reduce, self._opt_steps_reduce, self._ucb_reduce)
            self._total_opt_steps.add_(1)

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def _initialize_betas(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.betas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.betas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._space_parameters = [
            self.betas_normal,
            self.betas_reduce,
        ]

    def _initialize_others(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.register_buffer('_op_saliency_normal', torch.zeros(k, num_ops).cuda())
        self.register_buffer('_op_saliency_reduce', torch.zeros(k, num_ops).cuda())
        self._op_saliency = [
            self._op_saliency_normal,
            self._op_saliency_reduce,
        ]
        self.register_buffer('_opt_steps_normal', torch.ones(k, num_ops).cuda())
        self.register_buffer('_opt_steps_reduce', torch.ones(k, num_ops).cuda())
        self._opt_steps = [
            self._opt_steps_normal,
            self._opt_steps_reduce,
        ]
        self.register_buffer('_ucb_normal', torch.zeros(k, num_ops).cuda())
        self.register_buffer('_ucb_reduce', torch.zeros(k, num_ops).cuda())
        self._ucb = [
            self._ucb_normal,
            self._ucb_reduce,
        ]
        self.register_buffer('_op_mask_normal', torch.zeros([k, num_ops], dtype=torch.bool).cuda())
        self.register_buffer('_op_mask_reduce', torch.zeros([k, num_ops], dtype=torch.bool).cuda())
        self._op_mask = [
            self._op_mask_normal,
            self._op_mask_reduce,
        ]
        self.register_buffer('_op_avail_normal', torch.ones([k, num_ops], dtype=torch.bool).cuda())
        self.register_buffer('_op_avail_reduce', torch.ones([k, num_ops], dtype=torch.bool).cuda())
        self._op_avail = [
            self._op_avail_normal,
            self._op_avail_reduce,
        ]
        self.register_buffer('_total_opt_steps', torch.ones([]).cuda())

    def arch_parameters(self):
        return self._arch_parameters

    def space_parameters(self):
        return self._space_parameters

    def op_saliency(self):
        return self._op_saliency[0] * self._op_avail[0].float(), self._op_saliency[1] * self._op_avail[1].float()

    def save_info(self, path):
        dic = {}
        dic['alphas_normal'] = self.alphas_normal.detach().cpu().numpy()
        dic['alphas_reduce'] = self.alphas_reduce.detach().cpu().numpy()
        dic['betas_normal'] = self.betas_normal.detach().cpu().numpy()
        dic['betas_reduce'] = self.betas_reduce.detach().cpu().numpy()
        dic['_op_mask_normal'] = self._op_mask_normal.cpu().numpy()
        dic['_op_mask_reduce'] = self._op_mask_reduce.cpu().numpy()
        dic['_opt_steps_normal'] = self._opt_steps_normal.cpu().numpy()
        dic['_opt_steps_reduce'] = self._opt_steps_reduce.cpu().numpy()
        dic['_ucb_normal'] = self._ucb_normal.cpu().numpy()
        dic['_ucb_reduce'] = self._ucb_reduce.cpu().numpy()
        dic['_op_avail_normal'] = self._op_avail_normal.cpu().numpy()
        dic['_op_avail_reduce'] = self._op_avail_reduce.cpu().numpy()
        dic['_op_saliency_normal'] = self._op_saliency_normal.cpu().numpy()
        dic['_op_saliency_reduce'] = self._op_saliency_reduce.cpu().numpy()
        dic['_total_opt_steps'] = self._total_opt_steps.cpu().numpy()
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, '{}.pkl'.format(int(self._total_opt_steps)))
        pickle.dump(dic, open(filepath, 'wb'))

    def genotype(self):
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                start = end
                n += 1
            return gene

        gene_normal = _parse((self._op_saliency_normal.data * self._op_avail_normal.float().data).cpu().numpy())
        gene_reduce = _parse((self._op_saliency_reduce.data * self._op_avail_reduce.float().data).cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)
        genotype = Genotype(normal=gene_normal, normal_concat=concat, reduce=gene_reduce, reduce_concat=concat)
        return genotype
