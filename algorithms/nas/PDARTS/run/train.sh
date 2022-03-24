excute_dir=$(dirname $(pwd))
script_name="${excute_dir}/nas/PDARTS/main/train.py"
data_dir="${excute_dir}/data/Cifar10"
ckpt_path="${excute_dir}/data/models/PDARTS"
log_path="${excute_dir}/data/logs/PDARTS"

nohup python  -u  $script_name --data $data_dir --ckpt $ckpt_path \
   > $log_path 2>&1 &