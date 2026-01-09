echo "========================================================"
echo " PetSkin Final Challenge: Breaking the 70% Wall"
echo " Strategy: Forget Rate 0.0 (All Data) + Co-Correction"
echo " Start Time: `date`"
echo "========================================================"

#!/bin/bash
echo -e "\n\e[1;46m [Exp 12 - New Start] Hybrid SOTA (0 to 500 Epochs)\e[0m"

python Co-Correcting.py --dir experiment/petskin/exp12_new_500 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 0 \
 --forget-rate 0.0 \
 --lambda1 400 --alpha 0.8 \
 --stage1 70 --stage2 300 \
 --epochs 500 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

echo -e "\n\e[1;42m [FR 0.1 - New Start] 10% Filtered (0 to 500 Epochs)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_fr0.1_new_500 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.1 \
 --lambda1 400 --alpha 0.8 \
 --stage1 70 --stage2 300 \
 --epochs 500 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02


echo -e "\n\e[1;43m [FR 0.05 - New Start] 5% Filtered (0 to 500 Epochs)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_fr0.05_new_500 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --stage1 70 --stage2 300 \
 --epochs 500 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02


echo -e "\n\e[1;44m [Alpha 0.4 - New Start] The Balanced One (0 to 500 Epochs)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_alpha0.4_fr0.1_new_500 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.1 \
 --lambda1 400 --alpha 0.4 \
 --stage1 70 --stage2 300 \
 --epochs 500 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

echo -e "\n\e[1;44m [Alpha 0.4 - New Start] The Balanced One (0 to 500 Epochs)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_alpha0.4_fr0.05_new_500 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.4 \
 --stage1 70 --stage2 300 \
 --epochs 500 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02
echo "========================================================"
echo " All Experiments Completed!"
echo " End Time: `date`"
echo "========================================================"