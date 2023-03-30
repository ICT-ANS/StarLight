[//]: # (# StarLight)
<div align=center>
<img src="assets/StarLight.png" />
</div>


![lincense](https://img.shields.io/badge/license-MIT-brightgreen)
![Python](https://img.shields.io/badge/Python-%3E%3D3.6-important)
![Pytorch](https://img.shields.io/badge/PyTorch-%3E%3D1.0-important)
![release](https://img.shields.io/badge/release-v1.0-informational)
![docs](https://img.shields.io/badge/docs-updating-ff69b4)

StarLight helps in obtaining lightweight deep neural networks. StarLight consists primarily of three modules: network compression, neural architecture search, and visualization. The network compression module uses pruning and quantization techniques to convert a pre-trained network into a lightweight structure. The neural architecture search module designs efficient structures by utilizing differentiable architecture search methods. The visualization window can display all of the aforementioned processes, as well as visualizations of network intermediate features. We further provide a convenient tool [QuiverPyTorch](https://github.com/ICT-ANS/StarLight/tree/main/quiver_pytorch) to visualize the intermediate features of any networks.

<div align="center">
  <h3>
    <a href="https://ict-ans.github.io/StarLight.github.io/docs/Installation.html">
      Installation
    </a>
    <span> | </span>
    <a href="https://github.com/ICT-ANS/StarLight">
      Tutorials
    </a>
    <span> | </span>
    <a href="https://ict-ans.github.io/StarLight.github.io/">
      Documentation
    </a>
    <span> | </span>
    <a href="https://github.com/ICT-ANS/StarLight/tree/main/quiver_pytorch">
      QuiverPyTorch
    </a>
    <span> | </span>
    <a href="https://github.com/ICT-ANS/StarLight/issues">
      FAQs
    </a>
  </h3>
</div>

---

## Table of Contents
- [Highlighted Features](#highlighted-features)
- [Available Algorithms](#available-algorithms)
- [Demo](#demo)
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Basic usage](#basic-usage)
  - [Examples](#examples)
- [Guide for compressing your own networks](#guide-for-compressing-your-own-networks)
- [Visualize your own networks in StarLight](#visualize-your-own-networks-in-starlight)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [Contributing](#contributing)
  - [Reporting Bugs](#report-bugs)
  - [Commit New Networks](#commit-new-networks)
- [License](#license)

## Highlighted Features
* We present lightweight results of **6 popular networks**, including image classification, semantic segmentation, and object detection. 
* We have collected **over 50 bugs and solutions** during experiments in [Bug Summary](https://ict-ans.github.io/StarLight.github.io/docs/Bug%20Summary%20(EN).html), which can enable an efficient lightweight experience when dealing with your own networks. 
* With **just 1 yaml file**, you can easily visualize your own lightweight networks in StarLight. 
* In addition to 2D convolution pruning, we also provide **support for 3D convolution pruning**. Please refer to our [Documentation](https://ict-ans.github.io/StarLight.github.io/) for more details.
* To handle the unrecognized operations in ONNX models, we have collected **6 plugins for network quantization**, which will be available soon. 
* We provide a convenient tool to visualize the network intermediate features, namely [QuiverPyTorch](https://github.com/ICT-ANS/StarLight/tree/main/quiver_pytorch).


## Available Algorithms

* Available tasks

| Task Type             |      Pruning       |    Quantization     | Neural Architecture Search |
|-----------------------|:------------------:|:-------------------:|:--------------------------:|
| Image classification  | :white_check_mark: | :white_check_mark:  |     :white_check_mark:     | :white_check_mark: | :white_check_mark: |
| Semantic Segmentation | :white_check_mark: | :white_check_mark:  |                            | |
| Object Detection      | :white_check_mark: | :white_check_mark:  |                            | |

* Available algorithms

| Method                       |                                                                                                            Algorithms                                                                                                            |
|------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Pruning                      | [AGP](https://arxiv.org/abs/1710.01878), [FPGM](https://github.com/he-y/filter-pruning-geometric-median), [Taylor](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf), [L1](https://arxiv.org/abs/1608.08710), L2 |
| Quantization                 |                                                                  [PTQ](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#enable_int8_c)                                                                   |
| Neural Architecture Search  |  [DARTS](https://github.com/quark0/darts), [GDAS](https://github.com/D-X-Y/AutoDL-Projects/blob/main/docs/CVPR-2019-GDAS.md), [DU-DARTS](https://github.com/ShunLu91/DU-DARTS), [DDSAS](https://github.com/xingxing-123/DDSAS)   |


## Demo
* Pruning, quantization and feature visualization in StarLight. 
<img src="https://github.com/ICT-ANS/StarLight.github.io/blob/main/assets/StarLight_Compress.gif" alt="StarLight-Compress" width="100%"/>

* Neural architecture search and feature visualization in StarLight. 
<img src="https://github.com/ICT-ANS/StarLight.github.io/blob/main/assets/StarLight_DARTS.gif" alt="StarLight-DARTS" width="100%"/>

## Installation
* We summarized detailed steps for installation [here](https://ict-ans.github.io/StarLight.github.io/docs/Installation.html).

## Getting Started
### Network compression
* Coming soon.

### Neural Architecture
* Coming soon.

### Visualization in StarLight
* Coming soon.


## Guide for compressing your own networks
You can easily compress your own networks according to our [Compress Guide](https://ict-ans.github.io/StarLight.github.io/docs/Compress%20Guide.html).


## Visualize your own networks in StarLight
With **just 1 yaml file**, you can conveniently visualize your own lightweight networks in StarLight. Please refer to the [Visualization in StarLight](https://ict-ans.github.io/StarLight.github.io/docs/Visualization%20in%20StarLight.html) for more details.


## Acknowledgments
This work is supported in part by the National Key R&D Program of China under Grant No. 2018AAA0102701 and in part by the National Natural Science Foundation of China under Grant No. 62176250 and No. 62203424.
The following people have helped test the StarLight toolkit, read the document and provid valuable feedback: Pengze Wu, Haoyu Li, and Jiancong Zhou.
We would like to thank ChatGPT for polishing the presentation of the document.


## Citation
If you find that this project helps your research, you can cite StarLight as following:
```
@misc{StarLight,
  author    = {Shun Lu and Longxing Yang and Zihao Sun and Jilin Mei and Yu Hu,
  year      = {2023},
  address   = {Institute of Computing Technology, Chinese Academy of Sciences},
  title     = {StarLight: An Open-Source AutoML Toolkit for Lightweighting Deep Neural Networks},
  url       = {https://github.com/ICT-ANS/StarLight}
}
```

## Contributing
Thanks for your interest in [StarLight](https://github.com/ICT-ANS/StarLight) and for willing to contribute! We'd love to hear your feedback. 

### Report Bugs
* Please first try to check if an issue exists in our [Bug Summary](https://github.com/ICT-ANS/StarLight) or [Issues](https://github.com/ICT-ANS/StarLight/issues). 
* If not, please describe the bug in detail and we will give a timely reply. 

### Commit New Networks
* We are happy to integrate your network to our StarLight. Please provide your network with the results and hyper-parameters to us. And a detailed description would be better. Thank you!

## License
This project is under the MIT license - please see the [LICENSE](https://github.com/ICT-ANS/StarLight) for details