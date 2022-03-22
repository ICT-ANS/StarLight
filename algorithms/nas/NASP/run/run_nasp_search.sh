#excute_dir=$(dirname $(dirname $(pwd)))
excute_dir=$(pwd)
# if [ "$#" -eq 2 ] ;then
#   echo "Debug=$1, excute_dir=${excute_dir}"
#   debug=$1
# else
#   echo "Debug=True, excute_dir=${excute_dir}"
#   debug=True
# fi


script_name="${excute_dir}/nas/NASP/main/train_search.py"
data_dir="${excute_dir}/data/Cifar10"
log_path="${excute_dir}/data/StarLight_Cache/nas.classification.darts/logdir/online_log/nasp_search_3.log"
nohup python  -u  $script_name --data $data_dir \
  > $log_path 2>&1 &
# python $script_name --data $data_dir
