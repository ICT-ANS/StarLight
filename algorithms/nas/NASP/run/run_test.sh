path=$(dirname $(dirname $(dirname $(pwd))))
echo ${path}
excute_dir=$(pwd)
echo ${excute_dir}
script_name="${excute_dir}/nas/NASP/main/train_search.py"
python $script_name
