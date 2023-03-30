#!/bin/sh
source activate
conda deactivate
conda activate ui
cd /home/user/ANS/final_term/StarLight
python compression_vis/compression.py