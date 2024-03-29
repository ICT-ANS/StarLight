export PYTHONPATH=algorithms/compression/:$PYTHONPATH
gpu=0
model=resnet50
pruner=fpgm
sparsity=0.2
data_path=nets/ResNet/data/dataset/tiny-imagenet-200
model_path=nets/ResNet/data/checkpoints/ResNet50/resnet50.pth
log_path=nets/ResNet/data/logs/ResNet50/prune
save_dir=nets/ResNet/data/checkpoints/ResNet50/${pruner}_${sparsity}
mkdir -p $log_path
CUDA_VISIBLE_DEVICES=${gpu} \
    python nets/ResNet50/prune.py \
        --data=$data_path \
        --model=${model} \
        --resume=$model_path \
        --pruner=${pruner} \
        --baseline \
        --sparsity=${sparsity} \
        --save_dir=$save_dir \
        > ${log_path}/${pruner}_${sparsity}.log 2>&1