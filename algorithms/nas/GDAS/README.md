# Searching for a robust neural architecture in four gpu hours
## Requirements
```
Python >= 3.5.5, PyTorch >= 0.3.1, torchvision == 0.2.0
```
## Dataset
Prepare dataset firstly, and the path are 
```
'./StarLight/algorithms/data/Cifar10'
```

## Neural Architecture Search
```
cd StarLight/algorithms
bash SNAS/run/GDAS_Search.sh
```


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

