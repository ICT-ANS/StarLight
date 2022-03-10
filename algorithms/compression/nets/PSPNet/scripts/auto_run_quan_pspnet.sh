###########################
# Quantization for PSPNet
###########################

#SparsityList=(0.3 0.5 0.7)
#SparsityList=(0.5)
QuanMode=(fp16 int8)
IDX=0
#ModelName=pspnet_sparse
ModelName=pspnet

#for((s=0; s<${#SparsityList[*]}; s++)); do
for((q=0; q<${#QuanMode[*]}; q++)); do
  GPU=$((${IDX} % 8))
  let IDX+=1
#  Sparsity=${SparsityList[s]}
  Sparsity=0.0
  Mode=${QuanMode[q]}
  LogName=quan_${ModelName}_s${Sparsity}_${Mode}
  CUDA_VISIBLE_DEVICES=${GPU} nohup python -u quan_pspnet.py --model ${ModelName} --sparsity ${Sparsity} \
   --quan_mode ${Mode}  > logdir/${LogName} 2>&1 &
  echo "GPU:${GPU} LogName:${LogName}"
done
#done
tail -f logdir/${LogName}
