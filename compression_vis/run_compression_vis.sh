#!/bin/sh
source activate
conda deactivate
conda activate py37-torch1.5-trt7
cd /home/user/yanglongxing/StarLight
python compression_vis/compression.py