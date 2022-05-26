# StarLight

![lincense](https://img.shields.io/badge/license-MIT-brightgreen)
![Python](https://img.shields.io/badge/Python-%3E%3D3.6-important)
![Pytorch](https://img.shields.io/badge/PyTorch-%3E%3D1.0-important)
![release](https://img.shields.io/badge/release-v1.0-informational)
![docs](https://img.shields.io/badge/docs-updating-ff69b4)

StarLight is a general framework, which provides various algorithms for Neural Architecture Search (NAS), Model Pruning, and Model Quantization.

---

## Capabilities in a glance

<table>
  <tbody>
    <tr align="center" valign="bottom">
    <td>
      </td>
      <td>
        <b>Frameworks & Libraries</b>
        <img src="docs/img/bar.png"/>
      </td>
      <td>
        <b>Algorithms</b>
        <img src="docs/img/bar.png"/>
      </td>
    </tr>
    </tr>
    <tr valign="top">
    <td align="center" valign="middle">
    <b>Built-in</b>
      </td>
      <td>
      <ul><li><b>Supported Frameworks</b></li>
        <ul>
          <li>PyTorch</li>
          <li>Keras</li>
          <li>TensorFlow</li>
          <li>MXNet</li>
          <li>Caffe2</li>
          <a href="docs/en_US/SupportedFramework_Library.rst">More...</a><br/>
        </ul>
        </ul>
      </td>
      <td align="left" >
        <b>Neural Architecture Search</b>
        <ul>
          <b>Classification</b>
          <ul>
            <li><a href="docs/classification/DARTS.md">DARTS</a></li>
            <li><a href="docs/classification/P-DARTS.md">P-DARTS</a></li>
            <li><a href="docs/classification/PC-DARTS.md">PC-DARTS</a></li>
            <li><a href="docs/classification/R-DARTS.md">R-DARTS</a></li>
            <li><a href="docs/classification/S-DARTS.md">S-DARTS</a></li>
            <li><a href="docs/classification/SGAS.md">SGAS</a></li>
            <li><a href="docs/classification/SNAS.md">SNAS</a></li>
            <li><a href="docs/classification/GDAS.md">GDAS</a></li>
            <li><a href="docs/classification/NASP.md">NASP</a></li>
            </ul>
          <b>Semantic egmentation</b>
          <ul>
            <li><a href="docs/segmentation/Auto-DeepLab.md">Auto-DeepLab</a></li>
            <li><a href="docs/segmentation/FasterSeg.md">FasterSeg</a></li>
          </ul>
          <b>Object Detection</b>
            <ul>
              <li><a href="docs/en_US/Tuner/BuiltinTuner.rst#BOHB">xxx</a></li>
              <li><a href="docs/en_US/Tuner/BuiltinTuner.rst#TPE">xxx</a></li>
            </ul>
        </ul>
          <a href="docs/compression/compression.md">Model Compression</a>
          <ul>
            <b>Pruning</b>
            <ul>
              <li><a href="docs/en_US/Compression/Pruner.rst#agp-pruner">AGP Pruner</a></li>
              <li><a href="docs/en_US/Compression/Pruner.rst#slim-pruner">Slim Pruner</a></li>
              <li><a href="docs/en_US/Compression/Pruner.rst#fpgm-pruner">FPGM Pruner</a></li>
              <li><a href="docs/en_US/Compression/Pruner.rst#netadapt-pruner">NetAdapt Pruner</a></li>
              <li><a href="docs/en_US/Compression/Pruner.rst#simulatedannealing-pruner">SimulatedAnnealing Pruner</a></li>
              <li><a href="docs/en_US/Compression/Pruner.rst#admm-pruner">ADMM Pruner</a></li>
              <li><a href="docs/en_US/Compression/Pruner.rst#autocompress-pruner">AutoCompress Pruner</a></li>
            </ul>
            <b>Quantization</b>
            <ul>
              <li><a href="docs/en_US/Compression/Quantizer.rst#qat-quantizer">QAT Quantizer</a></li>
              <li><a href="docs/en_US/Compression/Quantizer.rst#dorefa-quantizer">DoReFa Quantizer</a></li>
            </ul>
          </ul>
          <a>Multi-Task</a>
            <ul>
              <li><a href="docs/multi-task/TrackRCNN.md">TrackRCNN</a></li>
              <li><a href="docs/multi-task/MTL.md">MTL</a></li>
              <li><a href="docs/multi-task/Prune.md">Prune</a></li>
            </ul>

## File Structure

- document - 文档信息
- html - 针对各自模块添加对应的html文件
- modelcompress - 模型压缩对应的代码文件夹
- nas - 模型搜索对应的代码文件夹
- qtui - qt对应的ui文件
- tools - 共用的一些小函数

## Environment

- 1. create Conda Envirment

```
conda create -n starlight python=3.6
conda activate starlight
```

- 2. install pytorch, cudnn 8.0.0

```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2
conda install --channel https://conda.anaconda.org/nvidia cudnn=8.0.0
```

- 3. install TensorRT

```
# Download TensorRT 7.1.3.4 from https://developer.nvidia.com/compute/machine-learning/tensorrt 

cd TensorRT-7.1.3.4/ 
pip install python/tensorrt-7.1.3.4-cp36-none-linux_x86_64.whl
pip install uff/uff-0.6.9-py2.py3-none-any.whl
pip install graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl
```

- 4. pyqt5==5.12, simply use pip to for installation:

```shell
    pip install pyqt5==5.12,
    pip install PyQtWebEngine==5.12
```

- 5. others

```shell
    pip install easydict, opencv-python, flask, flask_cors, gevent, imageio, pynvml, pyyaml, psutil, matplotlib, pycocotools, Cython, thop, tensorboard, schema, onnx, pycuda==2019.1.1, tqdm
```

## Usage

1. Example for model compression

```shell
    cd modelcompress
    python modelcomp.py
```

2. Use Qt Creator to edit UI file:

- Installation for Qt Creator, here we take Ubuntu as an example:

```shell
    sudo apt install qtcreator
```

- For other systems, you can download the suitable version of Qt Creator from this link: [http://download.qt.io/archive/qtcreator/4.4/4.4.1/](http://download.qt.io/archive/qtcreator/4.4/4.4.1/)
- After editing the UI file, you can get its corresponding Python code by running:

```shell
    # first arg is PY file to be converted and the second is the UI file
    pyuic5 -o modelcomp_ui.py modelcompress.ui 
```

[comment]: [comment]:
