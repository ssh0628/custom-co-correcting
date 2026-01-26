echo "========================================================"
echo " PetSkin Final"
echo " Start Time: `date`"
echo "========================================================"

#echo -e "\n\e[1;42m 2. [L400 / A0.8 / FR 0.05] Conservative + Gradual 30 + Warm Up 3\e[0m"
#python Co-Correcting.py --dir experiment/petskin/exp_400_0.8_0.05_warmup_3 \
# --dataset petskin --dataRoot /root/project/dataset/dataset \
# --noise_type clean --noise 0.0 --optim SGD \
# --num-gradual 30 \
# --forget-rate 0.05 \
# --lambda1 400 --alpha 0.8 \
# --warmup 3 \
# --stage1 70 --stage2 200 \
# --epochs 320 \
# --data_device 1 --workers 16 --batch-size 128 \
# --lr 0.02 --lr2 0.02

echo -e "\n\e[1;45m [Special] Load Best Model & Start Correction Immediately \e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_correction_start \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --resume "experiment/petskin//exp0_true_baseline/model_best.pth.tar" \
 --warmup 0 \
 --stage1 179 \
 --stage2 280 \
 --epochs 380 \
 --forget-rate 0.05 --num-gradual 10 \
 --lambda1 400 --alpha 0.8 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.0002 --lr2 0.0002

echo "========================================================"
echo " All Experiments Completed!"
echo " End Time: `date`"
echo "========================================================"