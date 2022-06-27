dataset=$1
method=$2
export MKL_THREADING_LAYER=GNU
python darts_online_debug/debug_online.py --dataset ${dataset} --method ${method}
