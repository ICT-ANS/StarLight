# DDSAS: Dynamic and Diﬀerentiable Space-Architecture Search (ACML 2021)

## Requirements
* Python 3.6.8
* PyTorch 1.4.0


## Dataset
Prepare dataset firstly, and the path are 
```
'./StarLight/data/Cifar10'
```

## Neural Architecture Search
```
cd StarLight/algorithms
bash nas/DDSAS/run/run_ddsas_online.sh
```

## Result
|             | Top1  | Params/M |
|:-----------:| :----:| :----: |
| Results | 2.59 &plusmn; 0.17 | 3.5 |

## Citation
Please cite our paper if you find anything helpful.
```
@InProceedings{yang21,
      title = {DDSAS: Dynamic and Differentiable Space-Architecture Search},
      author = {Yang, Longxing and Hu, Yu and Lu, Shun and Sun, Zihao and Mei, Jilin and Zeng, Yiming and Shi, Zhiping and Han, Yinhe and Li, Xiaowei},
      booktitle={ACML},
      year={2021}
    }
```


## License
MIT License

## Acknowledgement
This code is heavily borrowed from [DARTS](https://github.com/quark0/darts) and [SGAS](https://github.com/lightaime/sgas). Great thanks to their contributions.
