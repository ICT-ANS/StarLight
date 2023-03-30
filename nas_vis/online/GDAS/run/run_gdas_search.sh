#!/bin/bash

WorkDir=$1
Debug=$2
script_name="${WorkDir}/nas_vis/online/GDAS/main/GDAS_search.py"
data_path="${WorkDir}/data/StarLight_Cache/nas.classification/data/cifar10"
config_path="${WorkDir}/nas_vis/online/GDAS/config/search-opts/GDAS-NASNet-CIFAR.config"
model_config="${WorkDir}/nas_vis/online/GDAS/config/search-archs/GDAS-NASNet-CIFAR.config"
log_path="${WorkDir}/data/StarLight_Cache/nas.classification/GDAS/logdir/Online_GDAS_CIFAR-10.log"
save_dir="${WorkDir}/data/StarLight_Cache/nas.classification/GDAS/logdir"
dataset=cifar10
track_running_stats=1
seed=3
space=darts

## Debug w/o log
#python $script_name \
#	--save_dir ${save_dir} \
#	--dataset ${dataset} --data_path ${data_path} \
#	--search_space_name ${space} \
#	--config_path  ${config_path} \
#	--model_config ${model_config} \
#	--tau_max 10 --tau_min 0.1 --track_running_stats ${track_running_stats} \
#	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
#	--workers 4 --print_freq 200 --rand_seed ${seed} --debug ${Debug}

nohup python -u $script_name \
	--save_dir ${save_dir} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} \
	--config_path  ${config_path} \
	--model_config ${model_config} \
	--tau_max 10 --tau_min 0.1 --track_running_stats ${track_running_stats} \
	--arch_learning_rate 0.0003 --arch_weight_decay 0.001 \
	--workers 4 --print_freq 200 --rand_seed ${seed} --debug ${Debug} \
  > $log_path 2>&1 &
#
#tail -f $log_path

