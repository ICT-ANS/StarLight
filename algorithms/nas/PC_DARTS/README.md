## Introduction

**PC-DARTS** is a memory-efficient differentiable architecture method based on **DARTS**. It mainly focuses on reducing the large memory cost of the super-net in one-shot NAS method, which means that it can also be combined with other one-shot NAS method e.g. **ENAS**. Different from previous methods that sampling operations, PC-DARTS samples channels of the constructed super-net. Interestingly, though we introduced randomness during the search process, the performance of the searched architecture is **better and more stable than DARTS!** For a detailed description of technical details and experimental results, please refer to the paper:

[Partial Channel Connections for Memory-Efficient Differentiable Architecture Search](https://openreview.net/forum?id=BJlS634tPr)

Yuhui Xu, Lingxi Xie, Xiaopeng Zhang, Xin Chen, Guo-Jun Qi, Qi Tian and Hongkai Xiong.


## Prepare dataset and output path
Specify the data_dir in the nas/PC_DARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/Cifar10
```
Specify the log_path in the nas/PC_DARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/logs/DARTS
```
Specify the ckpt_path in the nas/PC_DARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/models/DARTS
```

## Usage

### Architecture Search
To search an architecture with PC-DARTS on CIFAR-10, run:
```shell
cd StarLight/algorithms
bash nas/PC_DARTS/run/search.sh
```

### Architecture Retraining
To retrain an PC-DARTS searched architecture on CIFAR-10, run:
```shell
cd StarLight/algorithms
bash nas/PC_DARTS/run/train.sh
```


## Results
| Result \ Metric | Params/M | Top1 |
|:---------------:|:--------:|:----:|
|      Paper      |   3.6    | 2.57 |
|      Ours       |   2.8    | 2.66 |

Search a good arcitecture on ImageNet by using the search space of DARTS(**First Time!**).



## Acknowledgement
This code is re-organized based on the official release [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS).
Thanks to their great contributions.