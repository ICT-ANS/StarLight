WorkDir=$1
Debug=$2

ScriptName="${WorkDir}/nas_vis/online/DARTS/train_search.py"
DataDir="${WorkDir}/data/StarLight_Cache/nas.classification"
LogPath="${WorkDir}/data/StarLight_Cache/nas.classification/DARTS/logdir/Online_DARTS_CIFAR-10.log"

nohup python -u ${ScriptName} --data ${DataDir} --debug ${Debug} \
  > ${LogPath} 2>&1 &
#python  -u  $script_name --data $data_dir --debug $debug \
#  > $log_path 2>&1
#tail -f ${LogPath}
