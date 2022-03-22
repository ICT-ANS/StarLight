# SNAS: Stochastic Neural Architecture Search
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
bash nas/SNAS/run/run_snas_search.sh
```

## Architecture Evaluation
```
cd StarLight/algorithms
bash nas/SNAS/run/run_snas_eval.sh
```

## Architecture Test
```
cd StarLight/algorithms
bash nas/SNAS/run/run_snas_test.sh
```


## Experimental Results
|          | Top1   | Params | FLOPs |
|   ----   | ----   | ----   | ----  |
| 论文结果  | 97.0% |  2.9M  |  --   |  
| 复现结果  | 97.1% |  5.1M  |  824M |


## Reference
```
@article{xie2018snas,
  title={SNAS: stochastic neural architecture search},
  author={Xie, Sirui and Zheng, Hehui and Liu, Chunxiao and Lin, Liang},
  journal={arXiv preprint arXiv:1812.09926},
  year={2018}
}
```

## Acknowledgement
The codes are based on the codes of SNAS (https://github.com/SNAS-Series/SNAS-Series). We appreciate SNAS's codes and thank the authors of SNAS.