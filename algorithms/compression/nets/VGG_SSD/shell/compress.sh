# gpus=0
# prune_method='fpgm'
# prune_sparisity=0.2
# ft_epochs=1
# input_path='/home/user/yanglongxing/StarLight_Sun/data/compression/inputs/VOC-VGGSSD'
# output_path=/home/user/yanglongxing/StarLight_Sun/data/compression/outputs/VOC-VGGSSD/online-fpgm-null
# mkdir -p ${output_path}
# CUDA_VISIBLE_DEVICES=${gpus} \
#     python algorithms/compression/nets/VGG_SSD/prune_finetune.py \
#         --weights=${input_path}/model.pth \
#         --pruner=${prune_method} \
#         --sparsity ${prune_sparisity} \
#         --save_dir=${output_path} \
#         --write_yaml \
#         --finetune_epochs=${ft_epochs} \



export PYTHONPATH=algorithms/compression/:$PYTHONPATH

dataset=$1
model=$2
prune_method=$3
quan_method=$4
ft_lr=$5
ft_bs=$6
ft_epochs=$7
prune_sparisity=$8
gpus=$9
input_path=${10}
output_path=${11}
dataset_path=${12}

if [ $prune_method != 'null' ] && [ $quan_method == 'null' ] # prune
then
    mkdir -p ${output_path}
    CUDA_VISIBLE_DEVICES=${gpus} \
        python algorithms/compression/nets/VGG_SSD/prune_finetune.py \
            --weights=${input_path}/model.pth \
            --pruner=${prune_method} \
            --sparsity=${prune_sparisity} \
            --save_dir=${output_path} \
            --write_yaml \
            --finetune_epochs=${ft_epochs} \

elif [ $prune_method == 'null' ] && [ $quan_method != 'null' ] # quant
then
    mkdir -p ${output_path}
    CUDA_VISIBLE_DEVICES=${gpus} \
        python algorithms/compression/nets/VGG_SSD/quant.py \
            --weights=${input_path}/model.pth \
            --quan_mode=${quan_method} \
            --save_dir=${output_path} \
            --write_yaml \

elif [ $prune_method != 'null' ] && [ $quan_method != 'null' ] # prune_quant
then
    mkdir -p ${output_path}
    CUDA_VISIBLE_DEVICES=${gpus} \
        python algorithms/compression/nets/VGG_SSD/prune_finetune.py \
            --weights=${input_path}/model.pth \
            --pruner=${prune_method} \
            --sparsity=${prune_sparisity} \
            --save_dir=${output_path} \
            --write_yaml \
            --no_write_yaml_after_prune \
            --finetune_epochs=${ft_epochs} \

    CUDA_VISIBLE_DEVICES=${gpus} \
        python algorithms/compression/nets/VGG_SSD/prune_quant.py \
            --weights=${input_path}/model.pth \
            --pruner=${prune_method} \
            --sparsity=${prune_sparisity} \
            --quan_mode=${quan_method} \
            --save_dir=${output_path} \
            --write_yaml \
            --finetune_epochs=${ft_epochs} \


fi