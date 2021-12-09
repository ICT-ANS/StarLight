export PYTHONPATH=algorithms/compression/:$PYTHONPATH
gpu=0
model=resnet50
pruner=fpgm
sparsity=0.2
quan=fp16
data_path=data/dataset/tiny-imagenet-200
model_path=data/checkpoints/ResNet50/${pruner}_${sparsity}
log_path=data/logs/ResNet50/prune_quan
save_dir=data/checkpoints/ResNet50/${pruner}_${sparsity}_${quan}
mkdir -p $log_path
CUDA_VISIBLE_DEVICES=${gpu} \
    python algorithms/compression/nets/ResNet50/quan.py \
        --model=${model} \
        --data=$data_path \
        --resume="" \
        --prune_eval_path=${model_path} \
        --quan_mode=${quan} \
        --baseline \
        --save_dir=$save_dir \
        > ${log_path}/${pruner}_${sparsity}_${quan}.log 2>&1