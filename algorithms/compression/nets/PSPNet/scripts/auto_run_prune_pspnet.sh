#
##quan_mode=int8
##CUDA_VISIBLE_DEVICES=6 nohup python -u quan.py --quan_mode ${quan_mode} > logdir/log_quan_${quan_mode} 2>&1 &
#
#quan_mode=fp16
#CUDA_VISIBLE_DEVICES=7 nohup python -u quan.py --quan_mode ${quan_mode} > logdir/log_quan_${quan_mode} 2>&1 &
#
#tail -f logdir/log_quan_${quan_mode}


## debug prune pspnet
#log_name=debug_pspnet
#CUDA_VISIBLE_DEVICES=7 nohup python -u prune_pspnet.py > logdir/${log_name} 2>&1 &

## debug quan deeplab_v3
#quan_mode=int8
#log_name=debug_quan_segdeeplab_${quan_mode}
#CUDA_VISIBLE_DEVICES=7 nohup python -u quan_v1_only_backone.py --quan_mode ${quan_mode} > logdir/${log_name} 2>&1 &

#tail -f logdir/${log_name}


Pruner=fpgm
GPU_List=(4 5 6 7)
#GPU_List=(0 1 2 3)
Sparsity_List=(0.2 0.4 0.5 0.7)
#PruneLR_List=(1e-4 1e-4 8e-5 8e-5 4e-5 4e-5 1e-5 1e-5)
#FinetuneLR_List=(2e-4 1e-4 6e-5 2e-5 1e-5 8e-6 5e-6 2e-6)

for((g=0; g<${#GPU_List[*]}; g++)); do
  GPU=${GPU_List[g]}
#  PruneLR=${PruneLR_List[g]}
  PruneLR=0.01
#  FinetuneLR=${FinetuneLR_List[g]}
  FinetuneLR=0.01
  Sparsity=${Sparsity_List[g]}
#  Sparsity=0.5
  log_name=prune_pspnet_${Pruner}_p${PruneLR}_f${FinetuneLR}_s${Sparsity}
  CUDA_VISIBLE_DEVICES=${GPU} nohup python -u prune_pspnet.py --pruner ${Pruner} --sparsity ${Sparsity} \
  --prune_lr ${PruneLR} --finetune_lr ${FinetuneLR} \
    > logdir/${log_name} 2>&1 &
  echo "GPU:${GPU} Log_Name:${log_name}"
done
tail -f logdir/${log_name}
