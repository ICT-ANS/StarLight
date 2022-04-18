export PYTHONPATH=algorithms/compression/:$PYTHONPATH
export PYTHONPATH=algorithms/compression/nets/ResNet50_SSD:$PYTHONPATH
export PYTHONPATH=algorithms/compression/nets/ResNet50_SSD/SSD_Pytorch:$PYTHONPATH

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
        python algorithms/compression/nets/ResNet50_SSD/prune.py \
            --data=${dataset_path} \
            --resume_net=${input_path}/model.pth \
            --pruner=${prune_method} \
            --sparsity=${prune_sparisity} \
            --save_folder=${output_path} \
            --write_yaml \
            --finetune_lr=${ft_lr} \
            --finetune_epochs=${ft_epochs} \
            --batch_size=${ft_bs}
elif [ $prune_method == 'null' ] && [ $quan_method != 'null' ] # quan
then
    mkdir -p ${output_path}
    CUDA_VISIBLE_DEVICES=${gpus} \
        python algorithms/compression/nets/ResNet50_SSD/quan.py \
            --data=${dataset_path} \
            --weight=${input_path}/model.pth \
            --quan_mode=${quan_method} \
            --save_folder=${output_path} \
            --write_yaml \
            --batch_size=${ft_bs}
elif [ $prune_method != 'null' ] && [ $quan_method != 'null' ] # prune and quan
then
    mkdir -p ${output_path}
    CUDA_VISIBLE_DEVICES=${gpus} \
        python algorithms/compression/nets/ResNet50_SSD/prune.py \
                --data=${dataset_path} \
                --resume_net=${input_path}/model.pth \
                --pruner=${prune_method} \
                --sparsity=${prune_sparisity} \
                --save_folder=${output_path} \
                --no_write_yaml_after_prune \
                --write_yaml \
                --finetune_lr=${ft_lr} \
                --finetune_epochs=${ft_epochs} \
                --batch_size=${ft_bs}
    CUDA_VISIBLE_DEVICES=${gpus} \
        python algorithms/compression/nets/ResNet50_SSD/quan.py \
                --data=${dataset_path} \
                --weight=${input_path}/model.pth \
                --quan_mode=${quan_method} \
                --save_folder=${output_path} \
                --prune_eval_path=${output_path} \
                --write_yaml \
                --batch_size=${ft_bs}
fi
