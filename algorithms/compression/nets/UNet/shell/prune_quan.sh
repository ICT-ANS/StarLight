export PYTHONPATH=./Pytorch-UNet-master/:$PYTHONPATH

gpu=5
sparsity=0.2
quan_mode=fp16
save_dir=logs/prunequan/sparsity${sparsity}_${quan_mode}
prune_eval_path=logs/prune/sparsity_$sparsity

mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=$gpu nohup python quan.py \
    --quan_mode=$quan_mode \
    --save_dir=$save_dir \
    --prune_eval_path=$prune_eval_path \
    > logs/prunequan_sparsity${sparsity}_${quan_mode}.log 2>&1 &