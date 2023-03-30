import random
import numpy as np
import torch


def random_seed(seed=42, rank=0):
    torch.cuda.manual_seed(seed + rank)
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)
