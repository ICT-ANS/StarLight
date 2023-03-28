
GPU_ID=7
Sparsity=0.5
FinetuneLR=1e-2
LogName=logdir/log_prune_deeplabv3plus_s${Sparsity}_ft${FinetuneLR}

nohup python -u prune_deeplabv3plus.py --gpu_id ${GPU_ID} --sparsity ${Sparsity} \
  --finetune_lr ${FinetuneLR} --output_stride 16 \
  --ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar \
  > ${LogName} 2>&1 &

sleep 2s

echo "GPU:${GPU} Log_Name:${LogName}"
tail -f ${LogName}

