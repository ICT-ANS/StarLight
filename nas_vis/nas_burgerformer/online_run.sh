#!/bin/bash

# input
workdir=$1
iteration=$2
flops=$3
params=$4
debug=$5

# for test
# workdir="/home/user/ANS/final_term/StarLight"
# iteration=20
# flops=1e9
# params=10e6
# debug=False

export PYTHONPATH=${workdir}:$PYTHONPATH

if [ $debug -eq 1 ];
then
    init_popu_size=5
    parent_size=2
    mutate_size=2
    b=25
else
    init_popu_size=500
    parent_size=75
    mutate_size=75
    b=100
fi


NUM_PROC=1
gpu="0"
resume="${workdir}/data/StarLight_Cache/nas.classification/BurgerFormer/checkpoint/supernet.pth"
data_dir="${workdir}/data/StarLight_Cache/nas.classification/data/ImageNet-100"
log_path="${workdir}/data/StarLight_Cache/nas.classification/BurgerFormer/logdir/search"

mkdir -p $log_path

CUDA_VISIBLE_DEVICES=$gpu \
python -m torch.distributed.launch --nproc_per_node=$NUM_PROC --master_port `expr 123456` \
    $workdir/nas_vis/nas_burgerformer/search_evo.py \
    --data-dir $data_dir \
    --model unifiedarch_s20 \
    -b $b \
    --resume $resume \
    --val-split sub-val \
    --num-classes 100 \
    --init-popu-size $init_popu_size \
    --parent-size $parent_size \
    --mutate-size $mutate_size \
    --target flops_params \
    --target_flops ${flops} \
    --target_params ${params} \
    --seed 0 \
    --output $log_path \
    > $log_path/../Online_BurgerFormer_ImageNet-100.log

