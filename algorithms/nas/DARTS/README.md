# DARTS: Differentiable Architecture Search (ICLR 2019)

## Requirements
* Python 3.7.10
* PyTorch 1.7.1

## Prepare dataset and output path
This repo uses CIFAR-10 dataset. Make sure the dataset is located in the following path:
```shell
./StarLight/data/cifar
```
Specify the log path in the code:
```shell
./StarLight/data/logs
```
Specify the checkpoint path in the code:
```shell
./StarLight/data/models
```

## Usage
### Architecture Search
To search an architecture with the DARTS method on CIFAR-10, run:
```shell
cd StarLight/algorithms
bash nas/DARTS/run/search.sh
```

### Architecture evaluation
To re-train the searched model, run:
```shell
cd StarLight/algorithms
bash nas/DARTS/run/train.sh
```

## Results
| Result \ Metric | Params/M |        Top1        |
|:---------------:|:--------:|:------------------:|
|      Paper      |   3.3    | 3.00 &plusmn; 0.14 |
|      Ours       |   2.5    |        2.99        |

## Acknowledgement
This code is re-organized based on the official release [DARTS](https://github.com/quark0/darts).
Great thanks to their contributions.
