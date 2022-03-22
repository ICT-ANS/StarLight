excute_dir=$(pwd)
script_name="${excute_dir}/nas/SNAS/main/test_edge_all.py"
data_dir="${excute_dir}/data/Cifar10"
log_path="${excute_dir}/data/StarLight_Cache/nas.classification.darts/logdir/online_log/snas_search_0.log"
model_path="${excute_dir}/nas_output/SNAS/eval1/best_weights.pt"
python $script_name --data $data_dir --model_path $model_path
