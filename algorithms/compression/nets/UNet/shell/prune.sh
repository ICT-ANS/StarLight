export PYTHONPATH=./Pytorch-UNet-master/:$PYTHONPATH

gpu=4
sparsity=0.5
save_dir=logs/prune/sparsity_${sparsity}

mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=$gpu nohup python prune.py \
    --sparsity=$sparsity \
    --save_dir=$save_dir \
    > logs/prune_${sparsity}.log 2>&1 &