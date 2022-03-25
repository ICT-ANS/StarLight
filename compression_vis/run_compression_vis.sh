#!/bin/sh
source activate
conda deactivate
conda activate starlight
cd /home/xingxing/projects/StarLight
python compression_vis/compression.py