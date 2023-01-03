# DeepLabV3

## Introduction
DeepLabV3 is an accurate semantic segmentation algorithm. 
This repo uses the network from [here](https://github.com/VainF/DeepLabV3Plus-Pytorch) and provides a compression example for DeepLabV3.
For a detailed description of technical details and experimental results, please refer to the paper: \
[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://openaccess.thecvf.com/content_ECCV_2018/html/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.html) \
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam

## Prepare dataset
* Download and process the dataset according to the [official Cityscapes repo](https://github.com/mcordts/cityscapesScripts).
* Make sure the dataset is located in the following path: 
```shell
data_root=./dataset/cityscapes
```

## Prepare pre-trained model
* Download the pre-trained DeepLabV3Plus-ResNet101 checkpoint from the [Google Drive](https://drive.google.com/file/d/1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb/view) provided by [the above repo](https://github.com/VainF/DeepLabV3Plus-Pytorch).
* Move the pre-trained checkpoint to the following path:
```shell
ckpt=./checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar
```
## Usage
Model pruning. Set the sparsity in the following script and run this command:
```shell
bash scripts/prune_deeplabv3plus.sh
```
Model quantization. Set the quantization mode in the following script and run this command:
```shell
bash scripts/quan_deeplabv3plus.sh
```

## Results
|        Model        | FLOPs/G | Params/M |   mIoU/mAcc/allAcc   | Inference/s |
|:-------------------:|:-------:|:--------:|:--------------------:|:-----------:|
|      Baseline       | 633.17  |  58.75   | 0.7621/0.8375/0.9587 |   0.0137    |
|    Sparsity=0.5     | 261.79  |  19.30   | 0.6843/0.7780/0.9416 |   0.0142    |
|   Baseline (INT8)   | 633.17  |  58.75   | 0.7564/0.8316/0.9579 |   0.2018    |
| Sparsity=0.5 (INT8) | 261.79  |  19.30   | 0.6808/0.7735/0.9409 |   0.1398    |


## Acknowledgement
This code is re-organized based on the release [DeepLabV3](https://github.com/VainF/DeepLabV3Plus-Pytorch). 
Thanks to their great contributions.