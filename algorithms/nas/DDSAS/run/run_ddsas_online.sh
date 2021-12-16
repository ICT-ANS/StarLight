excute_dir=$(pwd)
if [ "$#" -eq 2 ] ;then
  echo "Debug=$1, excute_dir=${excute_dir}"
  debug=$1
else
  echo "Debug=True, excute_dir=${excute_dir}"
  debug=True
fi

script_name="${excute_dir}/nas/DDSAS/main/train_search.py"
data_dir="${excute_dir}/data/StarLight_Cache/nas.classification.ddsas"
log_path="${excute_dir}/data/StarLight_Cache/nas.classification.ddsas/logdir/online_log/ddsas_online.log"
nohup python  -u  $script_name --data $data_dir --debug $debug \
    --dataset cifar10 \
    --arch DDSAS_cifar10_search \
    --saliency_type simple \
    --dss_max_ops 28 \
    --dss_freq 30 \
    --seed 0 \
    > $log_path 2>&1 &

