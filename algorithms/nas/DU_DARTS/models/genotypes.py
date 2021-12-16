from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    # 'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# cifar10
du_darts_c10_s0 = Genotype(
    normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 2), ('dil_conv_3x3', 2), ('dil_conv_3x3', 4)], normal_concat=range(2, 6),
    reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
du_darts_c10_s1 = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_5x5', 1),
            ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
du_darts_c10_s2 = Genotype(
    normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_3x3', 2),
            ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('dil_conv_3x3', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

# cifar100
du_darts_c100_s0 = Genotype(
    normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0),
            ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('sep_conv_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
du_darts_c100_s1 = Genotype(
    normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0),
            ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=range(2, 6),
    reduce=[('skip_connect', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0),
            ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))
du_darts_c100_s2 = Genotype(
    normal=[('skip_connect', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 0),
            ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1),
            ('skip_connect', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 1)], reduce_concat=range(2, 6))



