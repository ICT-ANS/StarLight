# [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760)
by Xin Chen, Lingxi Xie, Jun Wu and Qi Tian.



## Introduction

This repository contains the search and evaluation code for Progressive DARTS (PDARTS) .

It requires only **0.3 GPU-days** (7 hours on a single P100 card) to finish a search progress on CIFAR10 and CIFAR100 datasets,
much faster than DARTS, and achieves higher classification accuracy on both CIFAR and ImageNet datasets (mobole setting).


## Prepare dataset and output path
Specify the data_dir in the nas/PDARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/Cifar10
```
Specify the log_path in the nas/PDARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/logs/DARTS
```
Specify the ckpt_path in the nas/PDARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/models/DARTS
```


## Usage
### Architecture Search
To search an architecture with P-DARTS on CIFAR-10, run:
```shell
cd StarLight/algorithms
bash nas/P_DARTS/run/search.sh
```

### Architecture Retraining
To retrain an P-DARTS searched architecture on CIFAR-10, run:
```shell
cd StarLight/algorithms
bash nas/P_DARTS/run/train.sh
```


## Citation

```Latex
@inproceedings{chen2019progressive,
  title={Progressive differentiable architecture search: Bridging the depth gap between search and evaluation},
  author={Chen, Xin and Xie, Lingxi and Wu, Jun and Tian, Qi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1294--1303},
  year={2019}
}
```

## Acknowledgement
This code is re-organized based on the official release [PDARTS](https://github.com/chenxin061/pdarts).
Thanks to their great contributions.
