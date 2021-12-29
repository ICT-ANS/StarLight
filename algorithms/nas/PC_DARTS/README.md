## Introduction

**PC-DARTS** is a memory-efficient differentiable architecture method based on **DARTS**. It mainly focuses on reducing the large memory cost of the super-net in one-shot NAS method, which means that it can also be combined with other one-shot NAS method e.g. **ENAS**. Different from previous methods that sampling operations, PC-DARTS samples channels of the constructed super-net. Interestingly, though we introduced randomness during the search process, the performance of the searched architecture is **better and more stable than DARTS!** For a detailed description of technical details and experimental results, please refer to the paper:

[Partial Channel Connections for Memory-Efficient Differentiable Architecture Search](https://openreview.net/forum?id=BJlS634tPr)

Yuhui Xu, Lingxi Xie, Xiaopeng Zhang, Xin Chen, Guo-Jun Qi, Qi Tian and Hongkai Xiong.

**This code is re-organized based on the official release [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS).**

## Results
### Results on CIFAR10
Method | Params(M) | Error(%)| Search-Cost
--- | --- | --- | ---
AmoebaNet-B|2.8|2.55|3150
DARTSV1 | 3.3 | 3.00 | 0.4
DARTSV2 | 3.3 | 2.76 | 1.0
SNAS    | 2.8 | 2.85 |1.5
PC-DARTS | 3.6 | **2.57** | **0.1**

Only **0.1 GPU-days** are used for a search on CIFAR-10!
### Results on ImageNet
Method | FLOPs |Top-1 Error(%)|Top-5 Error(%)| Search-Cost
--- | --- | --- | --- | ---
NASNet-A |564|26.0|8.4|1800
AmoebaNet-B|570|24.3|7.6|3150
PNAS     |588 |25.8 |8.1|225
DARTSV2 | 574 | 26.7 | 8.7 | 1.0
SNAS    | 522 | 27.3 | 9.3 |1.5
PC-DARTS | 597 | **24.2** | **7.3** | 3.8

Search a good arcitecture on ImageNet by using the search space of DARTS(**First Time!**).


## Usage
#### Search on CIFAR10

To run the code, you only need one Nvidia 1080ti(11G memory).
```
python main/train_search.py \\
```
#### Search on ImageNet

Data preparation: 10% and 2.5% images need to be random sampled prior from earch class of trainingset as train and val, respectively. The sampled data is saved into `./imagenet_search`.
Note that not to use torch.utils.data.sampler.SubsetRandomSampler for data sampling as imagenet is too large.
```
python main/train_search_imagenet.py \\
       --tmp_data_dir /path/to/your/sampled/data \\
       --save log_path \\
```
#### The evaluation process simply follows that of DARTS.

##### Here is the evaluation on CIFAR10:

```
python main/train.py \\
       --auxiliary \\
       --cutout \\
```

##### Here is the evaluation on ImageNet (mobile setting):
```
python main/train_imagenet.py \\
       --tmp_data_dir /path/to/your/data \\
       --save log_path \\
       --auxiliary \\
       --note note_of_this_run
```

## Reference

```Latex
@inproceedings{
xu2020pcdarts,
title={{\{}PC{\}}-{\{}DARTS{\}}: Partial Channel Connections for Memory-Efficient Architecture Search},
author={Yuhui Xu and Lingxi Xie and Xiaopeng Zhang and Xin Chen and Guo-Jun Qi and Qi Tian and Hongkai Xiong},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJlS634tPr}
}
