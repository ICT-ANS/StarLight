excute_dir=$(pwd)

script_name="${excute_dir}/nas/GDAS/main/train.py"
data_dir="${excute_dir}/data/Cifar10"
log_path="${excute_dir}/data/StarLight_Cache/nas.classification.darts/logdir/online_log/gdas_eval_3.log"
nohup python  -u  $script_name --data $data_dir \
  > $log_path 2>&1 &
# python $script_name --data $data_dir