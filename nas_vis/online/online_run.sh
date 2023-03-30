Method=$1
WorkDir=$2
Debug=$3

if [ ${Method} == 'DARTS' ] ;then
    echo "Method=${Method}, WorkDir=${WorkDir}, Debug=${Debug}"
    ScriptName="${WorkDir}/nas_vis/online/DARTS/train_search.py"
    DataDir="${WorkDir}/data/StarLight_Cache/nas.classification"
    LogPath="${WorkDir}/data/StarLight_Cache/nas.classification/DARTS/logdir/Online_DARTS_CIFAR-10.log"

    nohup python -u ${ScriptName} --data ${DataDir} --debug ${Debug} \
    > ${LogPath} 2>&1 &
    
elif [ ${Method} == 'GDAS' ] ;then
    echo "Method=${Method}, WorkDir=${WorkDir}, Debug=${Debug}"
    bash "${WorkDir}/nas_vis/online/GDAS/run/run_gdas_search.sh" ${WorkDir} ${Debug}
else
    echo "[Error] Method=${Method}, WorkDir=${WorkDir}, Debug=${Debug}"
fi