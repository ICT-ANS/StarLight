from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PARAMS = {'conv_3x1_1x3':864, 'conv_7x1_1x7':2016, 'sep_7x7': 1464, 'conv 3x3':1296, 'sep_5x5': 888, 'sep_3x3':504, 'dil_conv_5x5': 444, 'conv 1x1':144, 'dil_conv_3x3':252, 'skip_connect':0}
PRIMITIVES_NORMAL = [
    "skip_connect",
    "conv_3x1_1x3",
    "dil_conv_3x3",
    "conv 1x1",
    "conv 3x3",
    "sep_3x3",
    "sep_5x5",
    "sep_7x7"]

PRIMITIVES_REDUCE = [
    "skip_connect",
    "avg_pool_3x3",
    "max_pool_3x3",
    "max_pool_5x5",
    "max_pool_7x7"]

NASNet = Genotype(
  normal = [
    ('sep_5x5', 1),
    ('sep_3x3', 0),
    ('sep_5x5', 0),
    ('sep_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_5x5', 1),
    ('sep_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_3x3', 0),
    ('sep_5x5', 2),
    ('sep_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_7x7', 2),
    ('sep_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)

GDAS = Genotype(normal = [('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('avg_pool_3x3', 1), ('dil_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
