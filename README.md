[//]: # (# StarLight)
![StarLight](assets/StarLight.png)

![lincense](https://img.shields.io/badge/license-MIT-brightgreen)
![Python](https://img.shields.io/badge/Python-%3E%3D3.6-important)
![Pytorch](https://img.shields.io/badge/PyTorch-%3E%3D1.0-important)
![release](https://img.shields.io/badge/release-v1.0-informational)
![docs](https://img.shields.io/badge/docs-updating-ff69b4)

StarLight helps in obtaining lightweight deep neural networks. StarLight consists primarily of three modules: network compression, neural architecture search, and visualization. The network compression module uses pruning and quantization techniques to convert a pre-trained network into a lightweight structure. The neural architecture search module designs efficient structures by utilizing differentiable architecture search methods. The visualization window can display all of the aforementioned processes, as well as visualizations of network intermediate features. We further provide a convenient tool [QuiverPyTorch](https://github.com/ICT-ANS/StarLight/tree/main/quiver_pytorch) to visualize the intermediate features of any networks.

<div align="center">
  <h3>
    <a href="https://github.com/ICT-ANS/StarLight">
      Tutorials
    </a>
    <span> | </span>
    <a href="https://github.com/ICT-ANS/StarLight">
      Documentation
    </a>
    <span> | </span>
    <a href="https://github.com/ICT-ANS/StarLight/tree/main/quiver_pytorch">
      QuiverPyTorch
    </a>
    <span> | </span>
    <a href="https://github.com/ICT-ANS/StarLight">
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
  - [Pruning](#pruning)
  - [Quantization](#quantization)
  - [Pruning and Quantization](#pruning-and-quantization)
  - [Integrate the network to StarLight](#integrate-the-network-to-starlight)
- [FAQs](#faqs)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
- [Contributing](#contributing)

## Highlighted Features
* 3D convolution pruning
* Plugins for network quantization
* Visualizations of network intermediate features, namely QuiverPyTorch.


## Available Algorithms

* Available tasks

| Task Type             |      Pruning       |    Quantization     | Neural Architecture Search |
|-----------------------|:------------------:|:-------------------:|:--------------------------:|
| Image classification  | :white_check_mark: | :white_check_mark:  |     :white_check_mark:     | :white_check_mark: | :white_check_mark: |
| Semantic Segmentation | :white_check_mark: | :white_check_mark:  |                            | |
| Object Detection      | :white_check_mark: | :white_check_mark:  |                            | |

* Available algorithms

| Method                       |          Algorithms          |
|------------------------------|:----------------------------:|
| Pruning                      |  AGP, FPGM, Taylor, L1, L2   |
| Quantization                 |       QAT, DoReFa, LSQ       |
| Neural Architecture Search  | DARTS, GDAS, DU-DARTS, DDSAS |


## Demo
* Wait for a GIF.


## Installation
- Create a conda envirment. You can use the Tsinghua source for the conda and pip to accelerate installation. 

```shell
  conda create -n starlight python=3.6
  conda activate starlight
```

- Install PyTorch and cuDNN.

```shell
  conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.2
  conda install --channel https://conda.anaconda.org/nvidia cudnn=8.0.0
```

- Install TensorRT.

```shell
  # Go to https://developer.nvidia.com/compute/machine-learning/tensorrt
  # Download TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz 
  tar -zxf TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
  cd TensorRT-7.1.3.4/ 
  pip install python/tensorrt-7.1.3.4-cp36-none-linux_x86_64.whl
  pip install uff/uff-0.6.9-py2.py3-none-any.whl
  pip install graphsurgeon/graphsurgeon-0.4.5-py2.py3-none-any.whl
  
  # Test if the TensorRT is installed successfully
  python
  import tensorrt # No error mean success
```
- If `libnvinfer.so.7` or `libcudnn.so.8` is missing when you import the tensorrt, simply specify there direction in the `~/.bashrc`:

```shell
# search their direction
find / -name libnvinfer.so.7
find / -name libcudnn.so.8
# for libnvinfer.so.7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/env/TensorRT-7.1.3.4/lib
# for libcudnn.so.8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/anaconda3/envs/starlight/lib
```

- If ImportError: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.27' not found, build GLIBC_2.27 manually:

```shell
# download glibc-2.27
wget http://ftp.gnu.org/gnu/glibc/glibc-2.27.tar.gz
tar -zxf glibc-2.27.tar.gz
cd glibc-2.27
mkdir build
cd build/
../configure --prefix=/opt/glibc-2.17 # <-- where you install glibc-2.27
# if error for gawk/bison, install them using: sudo apt-get install gawk/bison
make -j <number of CPU Cores>  # You can find your <number of CPU Cores> by using `nproc` command
make install
# patch your Python
patchelf --set-interpreter /opt/glibc-2.17/lib/ld-linux-x86-64.so.2 --set-rpath /opt/glibc-2.17/lib/ /root/anaconda3/envs/starlight/bin/python
```


- Install PYQT5 and PyQtWebEngine:

```shell
  pip install pyqt5==5.12
  pip install PyQtWebEngine==5.12
```

- Install other packages.

```shell
  pip install easydict opencv-python flask flask_cors gevent imageio pynvml pyyaml psutil matplotlib pycocotools Cython thop schema prettytable
  pip install onnx==1.11.0 pycuda==2019.1.1 tensorboard==2.9.1 tqdm
  pip install opencv-python pdf2image 
```


## Getting Started
### Basic usage
* Add description for pruning/quantization and using our software.

### Examples

## Guide for compressing your own networks
### Pruning
1. Load your pre-trained network
```
  # get YourPretarinedNetwork and load pre-trained weights for it
  model = YourPretarinedNetwork(args).to(device)
  model.load_state_dict(checkpoint['state_dict'])
```

2. Set `config_list` and choose a suitable `pruner`
```
  from lib.algorithms.pytorch.pruning import (TaylorFOWeightFilterPruner, FPGMPruner, AGPPruner)

  # choose a pruner: agp, taylor, or fpgm
  if args.pruner == 'agp':
      config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
      pruner = AGPPruner(
          model,
          config_list,
          optimizer,
          trainer,
          criterion,
          num_iterations=1,
          epochs_per_iteration=1,
          pruning_algorithm='taylorfo',
      )
  elif args.pruner == 'taylor':
      config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
      pruner = TaylorFOWeightFilterPruner(
          model,
          config_list,
          optimizer,
          trainer,
          criterion,
          sparsifying_training_batches=1,
      )
  elif args.pruner == 'fpgm':
      config_list = [{'sparsity': args.sparsity, 'op_types': ['Conv2d']}]
      pruner = FPGMPruner(
          model,
          config_list,
          optimizer,
          dummy_input=torch.rand(1, 3, 64, 64).to(device),
      )
  else:
      raise NotImplementedError
```
* `sparsity` specifies the pruning sparsity, ranging from 0.0 to 1.0. Larger sparsity corresponds to a more lightweight model.
* `op_types` specifies the type of pruned operation and can be either `Conv2d` or `Conv3d`, or both of them.
* `optimizer`, `trainer`, and `criterion` are the same as pre-training your network.

3. Use the pruner to generate the pruning mask
```
  # generate and export the pruning mask
  pruner.compress()
  pruner.export_model(
    os.path.join(args.save_dir, 'model_masked.pth'), 
    os.path.join(args.save_dir, 'mask.pth')
  )
```
* `model_masked.pth` includes the model weights and the generated pruning mask.
* `mask.pth` only includes the generated pruning mask.

4. Export your pruned model
```
  from lib.compression.pytorch import ModelSpeedup

  # initialize a new model instance and load pre-trained weights with the pruning mask
  model = YourPretarinedNetwork(args).to(device)
  model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model_masked.pth')))
  masks_file = os.path.join(args.save_dir, 'mask.pth')

  # use the speedup_model() of ModelSpeedup() to automatically export the pruned model
  m_speedup = ModelSpeedup(model, torch.rand(input_shape).to(device), masks_file, device)
  m_speedup.speedup_model()
```
* `input_shape` denotes the shape of your model inputs with `batchsize=1`. 
* This automatic export method is susceptible to errors when unrecognized structures are present in your model. To assist in resolving any bugs that may arise during the pruning process, we have compiled a summary of known issues in our [Bug Summary](https://github.com/ICT-ANS/StarLight).

5. Fine-tune your pruned model
* To fine-tune the pruned model, we recommend following your own pre-training process. 
* Since the pruned model has pre-trained weights and fewer parameters, we suggest using a smaller `learning_rate` during the fine-tuning process.

### Quantization
1. Load your pre-trained network
```
  # get YourPretarinedNetwork and load pre-trained weights for it
  model = YourPretarinedNetwork(args).to(device)
  model.load_state_dict(checkpoint['state_dict'])
```
2. Initialize the dataloader.
```
  import torchvision.datasets as datasets

  def get_data_loader(args):
      train_dir = os.path.join(args.data, 'train')
      train_dataset = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
      train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  
      val_dir = os.path.join(args.data, 'val')
      val_dataset = datasets.ImageFolder(val_dir, transform=transforms.ToTensor())
      val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
  
      n_train = len(train_dataset)
      indices = list(range(n_train))
      random.shuffle(indices)
      calib_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:args.calib_num])
      calib_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=calib_sampler)
  
      return train_loader, val_loader, calib_loader
  train_loader, val_loader, calib_loader = get_data_loader(args)
```
* `calib_loader` uses a subset from the training dataset to calibrate during subsequent quantization.

3. Specify `quan_mode` and output paths of onnx, trt, and cache.
```
  onnx_path = os.path.join(args.save_dir, '{}_{}.onnx'.format(args.model, args.quan_mode))
  trt_path = os.path.join(args.save_dir, '{}_{}.trt'.format(args.model, args.quan_mode))
  cache_path = os.path.join(args.save_dir, '{}_{}.cache'.format(args.model, args.quan_mode))

  if args.quan_mode == "int8":
      extra_layer_bit = 8
  elif args.quan_mode == "fp16":
      extra_layer_bit = 16
  elif args.quan_mode == "best":
      extra_layer_bit = -1
  else:
      extra_layer_bit = 32
```

4. Define the `engine` for inference.

```
  from lib.compression.pytorch.quantization_speedup import ModelSpeedupTensorRT

  engine = ModelSpeedupTensorRT(
      model,
      input_shape,
      config=None,
      calib_data_loader=calib_loader,
      batchsize=args.batch_size,
      onnx_path=onnx_path,
      calibration_cache=cache_path,
      extra_layer_bit=extra_layer_bit,
  )
  if not os.path.exists(trt_path):
      engine.compress()
      engine.export_quantized_model(trt_path)
  else:
      engine.load_quantized_model(trt_path)
```

5. Use the `engine` for inference.
```
  loss, top1, infer_time = validate(engine, val_loader, criterion)
```
* `engine` is similar to the `model` and can be inferred on either GPU or TensorRT. 
* While the `eval()` method is necessary for `model` inference, it is not required for `engine`.
* Inference with `engine` will return both the outputs and the inference time.

### Pruning and Quantization
* After completing the `Pruning` process outlined above, use the pruned model to undergo the `Quantization` process.

### Integrate your network to StarLight
* Below is a summary of main steps for integration. For more detailed information, please refer to the example provided on [Integrating the ResNet-TinyImageNet to StarLight](https://github.com/ICT-ANS/StarLight).

1. Put your dataset and models to the right folder. 
* Put your dataset in the folder of `data/compression/dataset/YOUR_DATASET`.
* Put your original model (without pruning or quantization) in the folder of `data/compression/dataset/inputs/YOUR_DATASET-YOUR_MODEL`. Note that this model should be named as `model.pth`, which includes the model structure and the pre-trained weights, and is acquired by [saving the entire model using PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

2. Configure your dataset and models.
* In `compression_vis/config/global.yaml`, according to our provided examples, fill in the following blocks: `dataset,model`, `support_combinations`, `origin_performance`, `figures`.
* In `compression_vis/config/hyperparameters_setting.yaml`, according to our provided examples, fill in the `default_setting` block with the required hyperparameters (`ft_lr`, `ft_bs`, `ft_epochs`, `prune_sparsity`, `gpus`).

3. Generate necessary information during your pruning or quantization.
* Before pruning or quantization, add the code below:
```
  # for pruning or quantization
  if args.write_yaml:
      flops, params, _ = count_flops_params(model, (1, 3, 64, 64), verbose=False)
      _, top1, _, infer_time, _ = validate(model, val_loader, criterion)
      storage = os.path.getsize(args.resume)
      with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
          yaml_data = {
              'Accuracy': {'baseline': round(top1, 2), 'method': None},
              'FLOPs': {'baseline': round(flops/1e6, 2), 'method': None},
              'Parameters': {'baseline': round(params/1e6, 2), 'method': None},
              'Infer_times': {'baseline': round(infer_time*1e3, 2), 'method': None},
              'Storage': {'baseline': round(storage/1e6, 2), 'method': None},
          }
          yaml.dump(yaml_data, f)
```

* After pruning (usually requires fine-tuning) or quantization, add the code below:
```
  # for pruning: 
  if epoch == args.finetune_epochs - 1:
      if args.write_yaml and not args.no_write_yaml_after_prune:
          storage = os.path.getsize(os.path.join(args.save_dir, 'model_speed_up_finetuned.pth'))
          with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
              yaml_data = {
                  'Accuracy': {'baseline': yaml_data['Accuracy']['baseline'], 'method': round(top1, 2)},
                  'FLOPs': {'baseline': yaml_data['FLOPs']['baseline'], 'method': round(flops/1e6, 2)},
                  'Parameters': {'baseline': yaml_data['Parameters']['baseline'], 'method': round(params/1e6, 2)},
                  'Infer_times': {'baseline': yaml_data['Infer_times']['baseline'], 'method': round(infer_time*1e3, 2)},
                  'Storage': {'baseline': yaml_data['Storage']['baseline'], 'method': round(storage/1e6, 2)},
                  'Output_file': os.path.join(args.save_dir, 'model_speed_up_finetuned.pth'),
              }
              yaml.dump(yaml_data, f)
  
  # for quantization:
  if args.write_yaml:
    storage = os.path.getsize(trt_path)
    with open(os.path.join(args.save_dir, 'logs.yaml'), 'w') as f:
        yaml_data = {
            'Accuracy': {'baseline': yaml_data['Accuracy']['baseline'], 'method': round(top1, 2)},
            'FLOPs': {'baseline': yaml_data['FLOPs']['baseline'], 'method': round(flops/1e6, 2)},
            'Parameters': {'baseline': yaml_data['Parameters']['baseline'], 'method': round(params/1e6, 2)},
            'Infer_times': {'baseline': yaml_data['Infer_times']['baseline'], 'method': round(infer_time*1e3, 2)},
            'Storage': {'baseline': yaml_data['Storage']['baseline'], 'method': round(storage/1e6, 2)},
            'Output_file': os.path.join(args.save_dir, '{}_{}.trt'.format(args.model, args.quan_mode)),
        }
        yaml.dump(yaml_data, f)

```


4. (Optional) Compress your network in StarLight using the online mode.
* Create a script `compress.sh` in the folder of `algorithms/compression/nets/YOUR_MODEL/shell`.
* Define the required hyper-parameters as below:
```
  dataset=$1
  model=$2
  prune_method=$3
  quan_method=$4
  ft_lr=$5
  ft_bs=$6
  ft_epochs=$7
  prune_sparisity=$8
  gpus=$9
  input_path=${10}
  output_path=${11}
  dataset_path=${12}
```
* Use the above hyper-parameters to start your pruning or quantization. Please refer to our provided examples in `algorithms/compression/nets/ResNet50/shell/compress.sh` to write your startup command.

5. Visualization of network features.
* Add 6 randomly selected pictures to the folder of `data/compression/quiver/YOUR_DATASET`.
* Specify the resolution of your inputs in the `img_size` block of `compression_vis/config/global.yaml`.
* Add your entired model namely `model.pth` to the folder of `data/compression/model_vis/YOUR_DATASET-YOUR_MODEL`.
* Add your entired pruned model to the folder of `data/compression/model_vis/YOUR_DATASET-YOUR_MODEL`, they should be named as `online-PRUNER.pth` or `offline-PRUNER.pth` for online and offline mode, respectively.


## FAQs


## Acknowledgments
* This work is funded by xxx xxx xxx.
* We owe a debt of gratitude to xxx, xxx and xxx for reading the manuscript and providing valuable feedback.
* We would like to thank xxx xxx for invaluable advice on the presentation of the document.


## Citation
If you find that this project helps your research, you can cite StarLight as following:
```
@misc{StarLight,
  author    = {xxx and xxx and xxx and xxx and xxx,
  year      = {2023},
  publisher = {StarLight},
  address   = {Institute of Computing Technology, Chinese Academy of Sciences},
  title     = {StarLight: xxxxxxxxxxxxxxxxxxxxxxx},
  url       = {https://github.com/ICT-ANS/StarLight}
}
```


## Contributing
Thanks for your interest in [StarLight](https://github.com/ICT-ANS/StarLight) and for willing to contribute! We'd love to hear your feedback. 

### Reporting Bugs
* Please first try to check if an issue exists in our [Bug Summary](https://github.com/ICT-ANS/StarLight) or `Issue`. 
* If not, please describe the bug in detail and we will give a timely reply. 

### Commiting New Models
* We are happy to integrate your network to our StarLight. Please provide your network with the results and hyper-parameters to us. And a detailed description would be better. Thank you!

## License
This project is under the MIT license - please see the [LICENSE](https://github.com/ICT-ANS/StarLight) for details