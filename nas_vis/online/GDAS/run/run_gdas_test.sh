excute_dir=$(pwd)
script_name="${excute_dir}/nas/GDAS/main/test.py"
data_dir="${excute_dir}/data/Cifar10"
log_path="${excute_dir}/data/StarLight_Cache/nas.classification.darts/logdir/online_log/gdas_eval_3.log"
model_path="${excute_dir}/nas_output/GDAS/eval1/best_weights.pt"
python $script_name --data $data_dir --model_path $model_path
