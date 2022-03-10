Pruner=agp
#GPU_List=(0 1 2 3 4 5 6 7)
GPU_List=(5 6 7)
Sparsity_List=(0.3 0.5 0.8)
#PruneLR_List=(1e-4 1e-4 8e-5 8e-5 4e-5 4e-5 1e-5 1e-5)
#FinetuneLR_List=(2e-4 1e-4 6e-5 2e-5 1e-5 8e-6 5e-6 2e-6)

for((g=0; g<${#GPU_List[*]}; g++)); do
  GPU=${GPU_List[g]}
#  PruneLR=${PruneLR_List[g]}
  PruneLR=2e-5
#  FinetuneLR=${FinetuneLR_List[g]}
  FinetuneLR=2e-4
  Sparsity=${Sparsity_List[g]}
#  Sparsity=0.5
  log_name=prune_seg_deeplab_efficientnetb3_${Pruner}_p${PruneLR}_f${FinetuneLR}_${Sparsity}
  CUDA_VISIBLE_DEVICES=${GPU} nohup python -u prune_mars.py --pruner ${Pruner} --sparsity ${Sparsity} \
  --prune_lr ${PruneLR} --finetune_lr ${FinetuneLR} \
    > logdir/${log_name} 2>&1 &
  echo "GPU:${GPU} Log_Name:${log_name}"
done
tail -f logdir/${log_name}
