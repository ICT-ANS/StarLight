# Searching for a robust neural architecture in four gpu hours
## Requirements
```
Python >= 3.5.5, PyTorch >= 1.1.0, torchvision >= 0.3.0
```
## Dataset
Prepare dataset firstly, and the path are 
```
'./StarLight/algorithms/data/Cifar10'
```

## Neural Architecture Search
```
cd StarLight/algorithms
bash nas/GDAS/run/run_gdas_search.sh
```

## Neural Architecture Search
```
cd StarLight/algorithms
bash nas/GDAS/run/run_gdas_eval.sh
```

## Architecture Evaluation
```
cd StarLight/algorithms
bash nas/GDAS/run/run_gdas_test.sh
```

## Experimental Results
|          | Top1   | Params | FLOPs |
|   ----   | ----   | ----   | ----  |
| 论文结果  | 97.07% |  3.4M  |  --   |  
| 复现结果  | 97.1% |  4.3M  |  661M |

## Reference
```
@inproceedings{dong2019searching,
  title={Searching for a robust neural architecture in four gpu hours},
  author={Dong, Xuanyi and Yang, Yi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1761--1770},
  year={2019}
}
```

## Acknowledgement
The codes are based on the codes of GDAS (https://github.com/D-X-Y/AutoDL-Projects). We appreciate GDAS's codes and thank the authors of GDAS.
