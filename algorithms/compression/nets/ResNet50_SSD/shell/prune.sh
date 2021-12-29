export PYTHONPATH=algorithms/compression:$PYTHONPATH
export PYTHONPATH=algorithms/compression/nets/ResNet50_SSD/SSD_Pytorch:$PYTHONPATH
pruner=fpgm
sparsity=0.5
path=data/logs/ResNet50_SSD/prune/${pruner}_sparsity${sparsity}
mkdir -p $path
CUDA_VISIBLE_DEVICES=0 \
    nohup python algorithms/compression/nets/ResNet50_SSD/prune.py \
        --cfg=algorithms/compression/nets/ResNet50_SSD/configs/prune.yaml \
        --resume_net=data/checkpoints/ResNet50_SSD/origin/origin.pth \
        --pruner=$pruner \
        --save_folder=$path \
        --baseline \
        --sparsity=$sparsity \
        > $path/log.txt