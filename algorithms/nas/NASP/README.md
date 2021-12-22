# Efficient Neural Architecture Search via Proximal Iterations
## Requirements
```
Python >= 3.5.5, PyTorch == 0.3.1, torchvision == 0.2.0
```
## Dataset
Prepare dataset firstly, and the path are 
```
'./StarLight/algorithms/data/Cifar10'
```

## Neural Architecture Search
```
cd StarLight/algorithms
bash NASP/run/run_nasp.sh
```


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
The codes of this paper are based on the codes of DARTS (https://github.com/quark0/darts). We appreciate DARTS's codes and thank the authors of DARTS.
