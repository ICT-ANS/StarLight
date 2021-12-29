export PYTHONPATH=algorithms/compression:$PYTHONPATH
export PYTHONPATH=algorithms/compression/nets/ResNet50_SSD/SSD_Pytorch:$PYTHONPATH
gpu=5
pruner=fpgm
sparsity=0.2
quan_mode=fp16
prune_eval_path=data/logs/ResNet50_SSD/prune/${pruner}_sparsity${sparsity}
path=data/logs/ResNet50_SSD/prune_quan/${pruner}${sparsity}_${quan_mode}
mkdir -p $path
CUDA_VISIBLE_DEVICES=$gpu \
    nohup python algorithms/compression/nets/ResNet50_SSD/quan.py \
        --cfg=algorithms/compression/nets/ResNet50_SSD/configs/prune_quan.yaml \
        --weights=data/checkpoints/ResNet50_SSD/origin/origin.pth \
        --pruner=$pruner \
        --prune_eval_path=$prune_eval_path \
        --quan_mode=$quan_mode \
        --save_folder=$path \
        --baseline \
        > $path/log.txt