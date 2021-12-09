export PYTHONPATH=algorithms/compression/:$PYTHONPATH
gpu=0
model=resnet50
quan=fp16
data_path=data/dataset/tiny-imagenet-200
model_path=data/checkpoints/ResNet50/resnet50.pth
log_path=data/logs/ResNet50/quan
save_dir=data/checkpoints/ResNet50/${quan}
mkdir -p $log_path
CUDA_VISIBLE_DEVICES=${gpu} \
    python algorithms/compression/nets/ResNet50/quan.py \
        --model=${model} \
        --data=$data_path \
        --resume=${model_path} \
        --quan_mode=${quan} \
        --baseline \
        --save_dir=$save_dir \
        > ${log_path}/${quan}.log 2>&1