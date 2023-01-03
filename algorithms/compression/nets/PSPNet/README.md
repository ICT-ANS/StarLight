# PSPNet

## Introduction
PSPNet is an efficient semantic segmentation algorithm. 
This repo uses the network from [here](https://github.com/hszhao/semseg) and provides a compression example for PSPNet.
For a detailed description of technical details and experimental results, please refer to the paper: \
[Pyramid Scene Parsing Network](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html) \
Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia

## Prepare dataset
* Download and process the dataset according to the [official Cityscapes repo](https://github.com/mcordts/cityscapesScripts).
* Make sure the dataset and list of the samples are located in the following path: 
```shell
data_root=./dataset/cityscapes
train_list=./dataset/cityscapes/cityscapes_train_list.txt
test_list=./dataset/cityscapes/cityscapes_val_list.txt
```

## Prepare pre-trained model
* Download the pre-trained PSPNet50 checkpoint from the [Google Drive](https://drive.google.com/drive/folders/1A8JaqItMjz2XNzV6gNurW4WzUcYY5op_) provided by [the above repo](https://github.com/hszhao/semseg).
* Move the pre-trained checkpoint to the following path:
```shell
./pretrained/train_epoch_200.pth
```


## Usage
Model pruning. Set the sparsity in the following script and run this command:
```shell
bash scripts/auto_run_prune_pspnet.sh
```
Model quantization. Set the quantization mode in the following script and run this command:
```shell
bash scripts/auto_run_quan_pspnet.sh
```

## Results
|        Model        | FLOPs/G | Params/M |  mIoU/mAcc/allAcc  | Inference/s |
|:-------------------:|:-------:|:--------:|:-----:|:------------:|
|      Baseline       | 728.84  |  46.72   | 0.7730/0.8430/0.9597 |    0.060     |
|    Sparsity=0.5     | 182.68  |  11.70   | 0.7045/0.8015/0.9503 |    0.061    |
|   Baseline (INT8)   | 728.84  |  46.72   | 0.7717/0.8402/0.9594 |    0.028     |
| Sparsity=0.5 (INT8) | 182.68  |  11.70   | 0.7037/0.8006/0.9497 |    0.016     |


## Acknowledgement
This code is re-organized based on the official release [PSPNet](https://github.com/hszhao/semseg). Thanks to their great contributions.