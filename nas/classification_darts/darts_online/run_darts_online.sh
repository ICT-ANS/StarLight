#excute_dir=$(dirname $(dirname $(pwd)))
excute_dir=$(pwd)
if [ "$#" -eq 2 ] ;then
  echo "Debug=$1, excute_dir=${excute_dir}"
  debug=$1
else
  echo "Debug=True, excute_dir=${excute_dir}"
  debug=True
fi

script_name="${excute_dir}/nas/classification_darts/darts_online/train_search.py"
data_dir="${excute_dir}/data/StarLight_Cache/nas.classification.darts"
log_path="${excute_dir}/data/StarLight_Cache/nas.classification.darts/logdir/online_log/darts_online.log"
nohup python  -u  $script_name --data $data_dir --debug $debug \
  > $log_path 2>&1 &
#python  -u  $script_name --data $data_dir --debug $debug \
#  > $log_path 2>&1
#tail -f $log_path