python main.py --model deeplabv3plus_resnet101 --gpu_id 0 \
--crop_val --lr 0.01 --crop_size 513 --batch_size 16 --output_stride 16 \
--ckpt checkpoints/best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar \
--test_only

# /home/lushun/dataset/nasbench201/hub/checkpoints/
