# Model compression for SegDeepLab

## Usage

### Link your dataset
Make sure 'Mars_Seg_1119' is in the 'dastaset' directory:
```shell
ln -s /Your/path/to/dataset/ .
```

### Model Pruning
To perform model pruning, run:
```shell
python prune_mars.py
```

### Model Quantization
To perform model quantization, run:
```shell
python quan_mars.py
```
