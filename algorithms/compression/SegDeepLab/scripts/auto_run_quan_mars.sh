###########################
# Quantization for SegDeepLab
###########################

SparsityList=(0.3 0.5 0.8)
QuanMode=(fp16 int8)
IDX=0
ModelName=seg_deeplab_efficientnetb3_sparse

for((s=0; s<${#SparsityList[*]}; s++)); do
for((q=0; q<${#QuanMode[*]}; q++)); do
  GPU=$((${IDX} % 2))
  let GPU+=6
  let IDX+=1
  Sparsity=${SparsityList[s]}
  Mode=${QuanMode[q]}
  LogName=quan_${ModelName}_s${Sparsity}_${Mode}
  CUDA_VISIBLE_DEVICES=${GPU} nohup python -u quan_mars.py --model ${ModelName} --sparsity ${Sparsity} \
   --quan_mode ${Mode} --gpu_id ${GPU}  > logdir/${LogName} 2>&1 &
  echo "GPU:${GPU} LogName:${LogName}"
  if [ $GPU = 7 ] ; then
          echo "sleep 30s"
          sleep 30s
  fi

done
done
tail -f logdir/${LogName}
