# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from . import L1FilterPrunerMasker, L2FilterPrunerMasker, FPGMPrunerMasker, TaylorFOWeightFilterPrunerMasker

MASKER_DICT = {
    'l1': L1FilterPrunerMasker,
    'l2': L2FilterPrunerMasker,
    'fpgm': FPGMPrunerMasker,
    'taylorfo': TaylorFOWeightFilterPrunerMasker,
}
