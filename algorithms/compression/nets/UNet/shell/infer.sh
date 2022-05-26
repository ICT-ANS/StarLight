export PYTHONPATH=./Pytorch-UNet-master/:$PYTHONPATH

mkdir -p logs/
gpu=3

# # origin
# CUDA_VISIBLE_DEVICES=$gpu python infer.py > logs/infer_origin.log 2>&1
# # prune
# CUDA_VISIBLE_DEVICES=$gpu python infer.py --prune_eval_path logs/prune/sparsity_0.2 > logs/infer_prune.log 2>&1
# quan
CUDA_VISIBLE_DEVICES=$gpu python infer.py --quan_path logs/quan/quan_fp16 --quan_mode fp16 > logs/infer_quan.log 2>&1
# prune_quan
CUDA_VISIBLE_DEVICES=$gpu python infer.py --quan_path logs/prunequan/sparsity0.2_fp16 --quan_mode fp16 > logs/infer_prunequan.log 2>&1