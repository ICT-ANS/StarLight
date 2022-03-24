# DARTS: Differentiable Architecture Search (ICLR 2019)

## Requirements
* Python 3.7.10
* PyTorch 1.7.1

## Prepare dataset and output path
Specify the data_dir in the nas/DARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/Cifar10
```
Specify the log_path in the nas/DARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/logs/DARTS
```
Specify the ckpt_path in the nas/DARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/models/DARTS
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
