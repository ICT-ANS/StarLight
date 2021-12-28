export PYTHONPATH=algorithms/compression:$PYTHONPATH
export PYTHONPATH=algorithms/compression/nets/ResNet50_SSD/SSD_Pytorch:$PYTHONPATH
quan_mode=best
path=data/logs/ResNet50_SSD/quan/${quan_mode}
mkdir -p $path
CUDA_VISIBLE_DEVICES=0 \
    nohup python algorithms/compression/nets/ResNet50_SSD/quan.py \
        --cfg=algorithms/compression/nets/ResNet50_SSD/configs/quan.yaml \
        --weights=data/checkpoints/ResNet50_SSD/origin/origin.pth \
        --quan_mode=$quan_mode \
        --save_folder=$path \
        --baseline \
        > $path/log.txt