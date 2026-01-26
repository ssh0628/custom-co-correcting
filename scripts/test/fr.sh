#!/bin/bash

echo "========================================================"
echo " PetSkin Final Challenge: The Grid Search"
echo " Strategy: Gradual 30 + Full Combination Sweep"
echo " Total Experiments: 8"
echo " Start Time: `date`"
echo "========================================================"

# ------------------------------------------------------------------
# [Group 1] Lambda 400 Series (The Standard)
# ------------------------------------------------------------------

echo -e "\n\e[1;42m 1. [L400 / A0.8 / FR 0.1] Standard + Gradual 30\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_fr0.1_L400_A0.8 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.1 \
 --lambda1 400 --alpha 0.8 \
 --warmup 0 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

echo -e "\n\e[1;42m 2. [L400 / A0.8 / FR 0.05] Conservative + Gradual 30\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_fr0.05_L400_A0.8 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --warmup 0 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

# ------------------------------------------------------------------
# [Group 2] Lambda 800 Series (Strong Correction)
# ------------------------------------------------------------------

echo -e "\n\e[1;45m 5. [L800 / A0.8 / FR 0.1] High Lambda + Standard\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_fr0.1_L800_A0.8 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.1 \
 --lambda1 800 --alpha 0.8 \
 --warmup 0 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

echo -e "\n\e[1;45m 6. [L800 / A0.8 / FR 0.05] High Lambda + Conservative\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_fr0.05_L800_A0.8 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.05 \
 --lambda1 800 --alpha 0.8 \
 --warmup 0 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

# ------------------------------------------------------------------
# [Group 3] 짜투리
# ------------------------------------------------------------------

echo -e "\n\e[1;44m 3. [L400 / A0.4 / FR 0.1] Balanced Alpha + Gradual 30\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_fr0.1_L400_A0.4 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.1 \
 --lambda1 400 --alpha 0.4 \
 --warmup 0 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

echo -e "\n\e[1;44m 4. [L400 / A0.4 / FR 0.05] Balanced Alpha + Conservative\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_fr0.05_L400_A0.4 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.4 \
 --warmup 0 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

echo -e "\n\e[1;41m 7. [L800 / A0.4 / FR 0.1] High Lambda + Balanced Alpha\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_fr0.1_L800_A0.4 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.1 \
 --lambda1 800 --alpha 0.4 \
 --warmup 0 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

echo -e "\n\e[1;41m 8. [L800 / A0.4 / FR 0.05] High Lambda + Balanced + Conservative\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_fr0.05_L800_A0.4 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.05 \
 --lambda1 800 --alpha 0.4 \
 --warmup 0 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

echo -e "\n\e[1;42m 2. [L400 / A0.8 / FR 0.05] Conservative + Gradual 30\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp_fr0.05_L400_A0.8_10 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --warmup 10 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

python Co-Correcting.py --dir experiment/petskin/exp0_true_baseline \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 0 --forget-rate 0.0 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --warm-up 320 --stage1 320 --stage2 320 --epochs 320
 
echo "========================================================"
echo " All 8 Experiments Completed!"
echo " End Time: `date`"
echo "========================================================"