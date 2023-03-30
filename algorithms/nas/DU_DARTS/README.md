# DU-DARTS: Decreasing the Uncertainty of Differentiable Architecture Search (BMVC 2021)
![license](https://img.shields.io/badge/License-MIT-brightgreen)
![python](https://img.shields.io/badge/Python-3.7-blue)
![pytorch](https://img.shields.io/badge/PyTorch-1.7-orange)

![du-darts](figure/du-darts_arch_params.png)

## Requirements
* Python 3.7.10
* PyTorch 1.7.1

## Prepare dataset and output path
Specify the data_dir in the nas/DU_DARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/Cifar10
```
Specify the log_path in the nas/DU_DARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/logs/DARTS
```
Specify the ckpt_path in the nas/DU_DARTS/run/search.sh and nas/DARTS/run/train.sh:
```shell
./StarLight/data/models/DARTS
```

## Usage

### Architecture Search
To search a DU-DARTS model on CIFAR-10, run:
```shell
cd StarLight/algorithms
bash nas/DU_DARTS/run/search.sh
```

### Architecture Retraining
To retrain a DU-DARTS model on CIFAR-10, run:
```shell
cd StarLight/algorithms
bash nas/DU_DARTS/run/train.sh
```

## Results
| Results \ Metric |      Params/M      |        Top1        |
|:----------------:|:------------------:|:------------------:|
|      Paper       | 3.69 &plusmn; 0.06 | 2.38 &plusmn; 0.06 |



## Citation
Please cite our paper if you find anything helpful.
```
@inproceedings{lu2021dudarts,
        title={DU-DARTS: Decreasing the Uncertainty of Differentiable Architecture Search},
        author={Lu, Shun and Hu, Yu and Yang, Longxing and Sun, Zihao and Mei, Jilin and Zeng Yiming and Li, Xiaowei },
        booktitle={BMVC},
        year={2021}
}
```


## License
MIT License

## Acknowledgement
This code is re-organized based on the official release [DU-DARTS](https://github.com/ShunLu91/DU-DARTS).
Thanks to their great contributions.
