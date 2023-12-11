#!/usr/bin/env bash
cd ./utils_/
pwd

CUDA_PATH=/usr/local/cuda/

python build.py build_ext --inplace
# if you use anaconda3 maybe you need add this
# change code like https://github.com/rbgirshick/py-faster-rcnn/issues/706
python_version=$(python -c "import platform; v=platform.python_version_tuple(); print(v[0]+v[1])")
mv nms/cpu_nms.cpython-${python_version}-x86_64-linux-gnu.so nms/cpu_nms.so
mv nms/gpu_nms.cpython-${python_version}-x86_64-linux-gnu.so nms/gpu_nms.so
cd ..
