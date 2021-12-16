# Usage

1. Creat dataset/checkpoints/logs folders in data/

```
cd data
ln -s YOUR_TINYIMAGENET200_PATH dataset/
mkdir checkpoint/ResNet50
cp /YOUR_PATH/resnet50.pth checkpoints/ResNet50/
mkdir logs/ResNet50
```

2. infer/pruning/quantization/pruning&quantization
```
bash shell/infer.sh
bash shell/prune.sh
bash shell/quan.sh
bash shell/prune_quan.sh
```
