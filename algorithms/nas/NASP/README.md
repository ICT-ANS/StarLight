# Efficient Neural Architecture Search via Proximal Iterations
## Requirements
```
Python >= 3.5.5, PyTorch == 1.1.0, torchvision >= 0.3.0
```
## Dataset
Prepare dataset firstly, and the path are 
```
'./StarLight/algorithms/data/Cifar10'
```

## Neural Architecture Search
```
cd StarLight/algorithms
bash nas/NASP/run/run_nasp_search.sh
```

## Architecture Evaluation
```
cd StarLight/algorithms
bash nas/NASP/run/run_nasp_eval.sh
```

## Architecture Test
```
cd StarLight/algorithms
bash nas/NASP/run/run_nasp_test.sh
```


## Experimental Results
|          | Top1   | Params | FLOPs |
|   ----   | ----   | ----   | ----  |
| 论文结果  | 97.17% |  3.3M  |  --   |  
| 复现结果  | 97.33% |  3.7M  |  605M |


## Reference
```
@inproceedings{yao2020efficient,
  title={Efficient neural architecture search via proximal iterations},
  author={Yao, Quanming and Xu, Ju and Tu, Wei-Wei and Zhu, Zhanxing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={04},
  pages={6664--6671},
  year={2020}
}
```

## Acknowledgement
The codes are based on the codes of NASP (https://github.com/xujinfan/NASP-codes). We appreciate NASP's codes and thank the authors of NASP.
