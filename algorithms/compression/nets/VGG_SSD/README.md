# SSD-VGG
## Requirements
```
Python >= 3.6.0, PyTorch >= 1.6.0, torchvision >= 0.7.0
```


## Prepare dataset
Please refer to the [Repo](https://github.com/amdegroot/ssd.pytorch) for details.
Move the dataset to the path
```
'./StarLight/algorithms/data/VOC'
```


## Usage
Eval the original model
```
python eval.py --cfg ./configs/ssd_vgg_voc.yaml
```

Prune the model
```
python prune_finetune.py
```

Quant the model
```
python quant.py
```

Prune and Quant the model
```
python prune_quant.py
```


## Results
|      Model       | FLOPs/G | Params/M |  mAP   | infer_time/s |
| :--------------: | :-----: | :------: | :----: | :----------: |
|     baseline     |  31.40  |  26.28   | 0.778  |    0.02      |
|   sparsity0.2    |  21.29  |  18.24   | 0.776  |    0.004     |
|       int8       |  31.40  |  26.28   | 0.776  |    0.005     |
| sparsity0.2_int8 |  21.29  |  18.24   | 0.736  |    0.005     |



## Reference
```
@inproceedings{liu2016ssd,
  title={Ssd: Single shot multibox detector},
  author={Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C},
  booktitle={European conference on computer vision},
  pages={21--37},
  year={2016},
  organization={Springer}
}
```


## Acknowledgement
This code is re-organized based on [Repo](https://github.com/amdegroot/ssd.pytorch).
Thanks to their great contributions.
