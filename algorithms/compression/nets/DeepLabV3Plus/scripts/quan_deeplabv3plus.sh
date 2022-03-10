
GPU=6
QuanMode=fp16
LogName=logdir/log_quan_deeplabv3plus_${QuanMode}

CUDA_VISIBLE_DEVICES=${GPU} nohup python -u quan_deeplabv3plus.py --quan_mode ${QuanMode} \
  > ${LogName} 2>&1 &

sleep 2s

echo "GPU:${GPU} Log_Name:${LogName}"
tail -f ${LogName}

