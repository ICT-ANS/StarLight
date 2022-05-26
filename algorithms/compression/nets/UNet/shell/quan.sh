export PYTHONPATH=./Pytorch-UNet-master/:$PYTHONPATH

gpu=3
quan_mode=fp16
save_dir=logs/quan/quan_$quan_mode

mkdir -p $save_dir

CUDA_VISIBLE_DEVICES=$gpu nohup python quan.py \
    --quan_mode=$quan_mode \
    --save_dir=$save_dir \
    > logs/quan_$quan_mode.log 2>&1 &