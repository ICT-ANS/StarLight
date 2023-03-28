# Usage

1. Creat `dataset/checkpoints/logs` folders in `data/`

```
cd data
ln -s YOUR_TINYIMAGENET200_PATH dataset/
mkdir checkpoint/ResNet50
cp /YOUR_PATH/resnet50.pth checkpoints/ResNet50/
mkdir logs/ResNet50
```

2. infer/pruning/quantization/pruning&quantization on **resnet50**
```
cd algorithms/compression

bash nets/ResNet/shell/infer.sh
bash nets/ResNet/shell/prune.sh
bash nets/ResNet/shell/quan.sh
bash nets/ResNet/shell/prune_quan.sh
```

3. **Replace resnet50 with resnet101** when training model on resnet101. You also have to change files in `algorithms/compression/nets/ResNet/shell`.
```
mkdir checkpoint/ResNet101
cp /YOUR_PATH/resnet101.pth checkpoints/ResNet101/
mkdir logs/ResNet101
```

# Method

## Pruning Methods
FPGM: [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/abs/1811.00250)

Taylor: [Importance Estimation for Neural Network Pruning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.pdf)

AGP: [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878)

## Quantization

TensorRT FP16/INT8/Mix: [TensorRT](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21664-toward-int8-inference-deploying-quantization-aware-trained-networks-using-tensorrt.pdf?t=eyJscyI6InJlZiIsImxzZCI6IlJFRi1yZXNvdXJjZXMubnZpZGlhLmNvbVwvZ3RjZC0yMDIwXC9HVEMyMDIwczIxNjY0In0)



# Result

|  | accuracy | FLOPs/M | parameters/M | Storage/MB | Latency/ms |
| :----: | :----: | :----: | :----: | :----: | :----: |
| ResNet50 | 60.42 | 334.05338 | 23.86471 | 92 | 23.72 |
| ResNet50 + FPGM | 58.3 | 244.04986 | 17.34622 | 67 | 13.82 |
| ResNet50 + AGP | 57.8 | 248.56562 | 17.64869 | 68 | 14.02 |
| ResNet50 + Taylor | 57.34 | 248.56562 | 17.64869 | 68 | 13.96 |
| ResNet50 + FP16 | 60.44 | 334.05338 | 23.86471 | 47 | 0.98 |
| ResNet50 + INT8 | 60.16 | 334.05338 | 23.86471 | 25 | 0.99 |
| ResNet50 + Mix | 60.22 | 334.05338 | 23.86471 | 36 | 0.89 |
| ResNet50 + FPGM + FP16 | 58.34 | 244.04986 | 17.34622 | 35 | 0.9 |





