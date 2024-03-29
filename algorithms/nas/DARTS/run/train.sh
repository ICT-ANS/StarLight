excute_dir=$(dirname $(pwd))
script_name="${excute_dir}/nas/DARTS/main/train.py"
data_dir="${excute_dir}/data/Cifar10"
ckpt_path="${excute_dir}/data/models/DARTS"
log_path="${excute_dir}/data/logs/DARTS"

nohup python  -u  $script_name --data $data_dir --save $ckpt_path \
   > $log_path 2>&1 &