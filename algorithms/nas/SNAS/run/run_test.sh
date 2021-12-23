path=$(dirname $(dirname $(dirname $(pwd))))
echo ${path}
excute_dir=$(pwd)
echo ${excute_dir}
script_name="${excute_dir}/nas/SNAS/main/train_search.py"
python $script_name
