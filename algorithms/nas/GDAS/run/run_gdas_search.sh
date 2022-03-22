#!/bin/bash
# bash ./scripts-search/NASNet-space-search-by-GDAS.sh cifar10 1 -1
# echo script name: $0
# echo $# arguments
#if [ "$#" -ne 3 ] ;then
#  echo "Input illegal number of parameters " $#
#  echo "Need 3 parameters for dataset, track_running_stats, and seed"
#  exit 1
#fi
#if [ "$TORCH_HOME" = "" ]; then
#  echo "Must set TORCH_HOME envoriment variable for data dir saving"
#  exit 1
#else
#  echo "TORCH_HOME : $TORCH_HOME"
#fi



# if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
#   #data_path="$TORCH_HOME/cifar.python"
#   data_path="../data"
# else
#   data_path="$TORCH_HOME/cifar.python/ImageNet16"
# fi

excute_dir=$(pwd)
echo ${excute_dir}
script_name="${excute_dir}/nas/GDAS/main/GDAS_search.py"
data_path="${excute_dir}/data/Cifar10"
config_path="${excute_dir}/nas/GDAS/config/search-opts/GDAS-NASNet-CIFAR.config"
model_config="${excute_dir}/nas/GDAS/config/search-archs/GDAS-NASNet-CIFAR.config"
log_path="${excute_dir}/data/StarLight_Cache/nas.classification.darts/logdir/online_log/gdas_search_3.log"
save_dir=./nas_output/GDAS/search3
dataset=cifar10
track_running_stats=1
seed=3
space=darts

nohup python -u $script_name \
	--save_dir ${save_dir} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--config_path  ${config_path} \
	--model_config ${model_config} \
	--tau_max 10 --tau_min 0.1 --track_running_stats ${track_running_stats} \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--workers 4 --print_freq 200 --rand_seed ${seed} \
  > $log_path 2>&1 &

# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python $script_name \
# 	--save_dir ${save_dir} \
# 	--dataset ${dataset} --data_path ${data_path} \
# 	--search_space_name ${space} \
# 	--config_path  ${config_path} \
# 	--model_config ${model_config} \
# 	--tau_max 10 --tau_min 0.1 --track_running_stats ${track_running_stats} \
# 	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
# 	--workers 4 --print_freq 200 --rand_seed ${seed}
