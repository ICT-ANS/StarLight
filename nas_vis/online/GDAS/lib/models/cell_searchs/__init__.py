##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################
from .genotypes import Structure as CellStructure, architectures as CellArchitectures
from .search_model_gdas_nasnet import NASNetworkGDAS
nasnet_super_nets = {"GDAS": NASNetworkGDAS }