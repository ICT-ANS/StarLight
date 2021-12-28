export PYTHONPATH=algorithms/compression:$PYTHONPATH
export PYTHONPATH=algorithms/compression/nets/ResNet50_SSD/SSD_Pytorch:$PYTHONPATH
gpu=0
echo "baseline"
cfg=algorithms/compression/nets/ResNet50_SSD/configs/origin.yaml
save_folder=data/logs/ResNet50_SSD/origin
mkdir -p $save_folder
CUDA_VISIBLE_DEVICES=$gpu python algorithms/compression/nets/ResNet50_SSD/infer.py \
    --weights=data/checkpoints/ResNet50_SSD/origin/origin.pth \
    --save_folder=${save_folder} --cfg=${cfg} \
    --mode=baseline \
    > ${save_folder}/output.log