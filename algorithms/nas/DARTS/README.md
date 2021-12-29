# Differentiable Architecture Search

**This code is re-organized based on the official release [DARTS](https://github.com/quark0/darts).** 

## Requirements
* Python 3.7.10
* PyTorch 1.7.1

## Usage
### Model Search
To search a DARTS model on CIFAR-10, run:
```shell
python main/train_search.py
```

### Architecture evaluation
To re-train the searched model, run:
```shell
python main/train.py --auxiliary --cutout --arch DARTS
```
