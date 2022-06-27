import random
from collections import namedtuple

Genotype = namedtuple('Genotype', 'cell cell_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

ad0_s = [[1, 3],
         [0, 0],
         [4, 4],
         [3, 4],
         [6, 4],
         [8, 6],
         [10, 4],
         [13, 5],
         [18, 5],
         [15, 4]]

ad1_s = [[1, 3],
         [0, 3],
         [3, 3],
         [4, 0],
         [5, 5],
         [6, 1],
         [10, 3],
         [12, 4],
         [19, 4],
         [17, 6]]


def generate_random_genotypes(n, step=5, num_op=8):
    genotype_list = []
    for _ in range(n):
        _genotype = []
        _index = 0
        for i in range(step):
            for j in range(2):
                _genotype.append([random.randint(_index, _index+i+1), random.randint(0, 7)])
            _index += i + 2
        genotype_list.append(_genotype)
    return genotype_list


if __name__ == '__main__':
    geno_list = generate_random_genotypes(5)

    for i in geno_list:
        print(i)
