export PYTHONPATH=algorithms/compression/:$PYTHONPATH
gpu=0
model=resnet50
data_path=nets/ResNet/data/dataset/tiny-imagenet-200
model_path=nets/ResNet/data/checkpoints/ResNet50/resnet50.pth
log_path=nets/ResNet/data/logs/ResNet50/origin
save_dir=nets/ResNet/data/checkpoints/ResNet50/origin
mkdir -p $log_path
CUDA_VISIBLE_DEVICES=${gpu} \
    python nets/ResNet/infer.py \
        --model=${model} \
        --data=$data_path \
        --origin_model_path=$model_path \
        --save_dir=$save_dir \
        > ${log_path}/infer.log 2>&1