from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = None

DDSAS_cifar10 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 2), ('skip_connect', 0), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3)],
                         normal_concat=range(2, 6),
                         reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('sep_conv_5x5', 2)],
                         reduce_concat=range(2, 6))

DDSAS_cifar100 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2)],
                          normal_concat=range(2, 6),
                          reduce=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2), ('skip_connect', 1), ('sep_conv_3x3', 2), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3)],
                          reduce_concat=range(2, 6))

DDSAS_imagenet = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('skip_connect', 2), ('sep_conv_5x5', 3)],
                          normal_concat=range(2, 6),
                          reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('skip_connect', 2), ('sep_conv_5x5', 3)],
                          reduce_concat=range(2, 6))


def init_space(name):
    if name is None:
        space_config = None
    else:
        space_config = eval('{}'.format(name))
    return space_config


space_config_static = {
    'PRIMITIVES': [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
    ],
    'type': 'static',
}

space_config_shrink = {
    'PRIMITIVES': [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
    ],
    'type': 'shrink',
    'stage': [
        (10, 14),  # 112 -> 98
        (20, 14),  # 98 -> 84
        (30, 14),  # 84 -> 70
        (40, 14),  # 70 -> 56
        (50, 14),  # 56 -> 42
    ],
}

space_config_expand = {
    'PRIMITIVES': [
        'none',
        'skip_connect',
        'sep_conv_3x3',
        'max_pool_3x3',
        'dil_conv_3x3',
        'avg_pool_3x3',
        'sep_conv_5x5',
        'dil_conv_5x5',
    ],
    'type': 'expand',
    'stage': [
        (0, 3),  # epoch, expand num
        (10, 4),
        (20, 5),
        (30, 6),
        (40, 7),
        (50, 8),
    ]
}

space_config_imagenet = {
    'PRIMITIVES': [
        'sep_conv_3x3',
        'dil_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_5x5',
        'none',
        'skip_connect',
        'max_pool_3x3',
        'avg_pool_3x3',
    ],
    'type': 'expand',
    'stage': [
        (0, 4),  # epoch, expand num
        (40, 8),
    ]
}