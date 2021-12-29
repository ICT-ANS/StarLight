# [Progressive Differentiable Architecture Search: Bridging the Depth Gap between Search and Evaluation](https://arxiv.org/abs/1904.12760)
by Xin Chen, Lingxi Xie, Jun Wu and Qi Tian.


**This code is re-organized based on the official release [PDARTS](https://github.com/chenxin061/pdarts).**

## Introduction

This repository contains the search and evaluation code for Progressive DARTS (PDARTS) .

It requires only **0.3 GPU-days** (7 hours on a single P100 card) to finish a search progress on CIFAR10 and CIFAR100 datasets,
much faster than DARTS, and achieves higher classification accuracy on both CIFAR and ImageNet datasets (mobole setting).

## Usage

To run this code, you need a GPU with at least **16GB memory**, and equip it with PyTorch 0.4 or above versions.

If you have a GPU with smaller memory, say 12GB, you need to use a smaller batch-size in the search stage.
In the current example, using 64 instead of 96 works -- it is a bit slower but does not impact accuracy much.


#### Run the following command to perform a search progress on CIFAR10.

```
python main/train_search.py \\
       --tmp_data_dir /path/to/your/data \\
       --save log_path \\
       --add_layers 6 \\
       --add_layers 12 \\
       --dropout_rate 0.1 \\
       --dropout_rate 0.4 \\
       --dropout_rate 0.7 \\
       --note note_of_this_run
Add --cifar100 if search on CIFAR100.
```

It needs ~7 hours on a single P100 GPU, or 12 hours on a single 1080-Ti GPU to finish everything.
Our test with a limitation of 12 memory on P100 GPU tooks about 9 hours to finish the search. 


#### The evaluation process simply follows that of DARTS.

##### Here is the evaluation on CIFAR10/100:

```
python main/train_cifar.py \\
       --tmp_data_dir /path/to/your/data \\
       --auxiliary \\
       --cutout \\
       --save log_path \\
       --note note_of_this_run
Add --cifar100 if evaluating on CIFAR100.
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
@inproceedings{chen2019progressive,
  title={Progressive differentiable architecture search: Bridging the depth gap between search and evaluation},
  author={Chen, Xin and Xie, Lingxi and Wu, Jun and Tian, Qi},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={1294--1303},
  year={2019}
}
```
