excute_dir=$(pwd)
script_name="${excute_dir}/nas/DU_DARTS/main/train_search.py"
data_dir="${excute_dir}/data/Cifar10"
ckpt_path="${excute_dir}/data/models/DU_DARTS"
log_path="${excute_dir}/data/logs/DU_DARTS"

nohup python  -u  $script_name --data $data_dir --save $ckpt_path \
   > $log_path 2>&1 &