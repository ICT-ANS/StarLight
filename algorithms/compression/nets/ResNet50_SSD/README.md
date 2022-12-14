## Introduction

SSD is a one-stage object detection algorithm. The algorithm utilizes the anchor mechanism to directly regression the category and location of the target. This code is based on [Repo](https://github.com/amdegroot/ssd.pytorch). For a detailed description of technical details and experimental results, please refer to the paper:

[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.


## Prepare dataset
Please refer to the [Repo](https://github.com/amdegroot/ssd.pytorch) for details.

## Usage

infer/pruning/quantization/pruning&quantization
```
bash shell/infer.sh
bash shell/prune.sh
bash shell/quan.sh
bash shell/prune_quan.sh
```


## Results
|      Model       | FLOPs/G | Params/M |  mAP   | infer_time/s |
| :--------------: | :-----: | :------: | :----: | :----------: |
|     baseline     |  9.926  |  38.058  | 0.7630 |    0.016     |
|   sparsity0.2    |  6.631  |  24.952  | 0.7534 |    0.008     |
|       fp16       |  9.926  |  38.058  | 0.7630 |    0.005     |
| sparsity0.2_fp16 |  6.631  |  24.952  | 0.7530 |    0.005     |



## Acknowledgement
This code is re-organized based on [Repo](https://github.com/amdegroot/ssd.pytorch).
Thanks to their great contributions.