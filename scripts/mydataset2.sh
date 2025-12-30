#!/bin/bash

echo "========================================================"
echo " PetSkin Clean Dataset Hyperparameter Tuning"
echo " Start Time: `date`"
echo "========================================================"

# ------------------------------------------------------------------
# 1. [Baseline] 기준점 잡기
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 1] Baseline Standard Setting\e[0m"
python Co-Correcting.py --dir experiment/petskin/clean_baseline \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 32 \
 --lr 0.01 --lr2 0.01 --forget-rate 0.1 --lambda1 1000 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 1: `date`"


# ------------------------------------------------------------------
# 2. [Learning Rate] 고속 학습 (High LR)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 2] High Learning Rate (lr=0.02)\e[0m"
python Co-Correcting.py --dir experiment/petskin/clean_high_lr \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 32 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.1 --lambda1 1000 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 2: `date`"


# ------------------------------------------------------------------
# 3. [Learning Rate] 정밀 학습 (Low LR)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 3] Low Learning Rate (lr=0.005)\e[0m"
python Co-Correcting.py --dir experiment/petskin/clean_low_lr \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 32 \
 --lr 0.005 --lr2 0.005 --forget-rate 0.1 --lambda1 1000 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 3: `date`"


# ------------------------------------------------------------------
# 4. [Epoch] 장기 학습 (Long Epochs)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 4] Long Training (Epochs=300)\e[0m"
python Co-Correcting.py --dir experiment/petskin/clean_long_epoch \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 32 \
 --lr 0.01 --lr2 0.01 --forget-rate 0.1 --lambda1 1000 \
 --stage1 70 --stage2 200 --epochs 300
echo "Finished Exp 4: `date`"


# ------------------------------------------------------------------
# 5. [Forget Rate] 데이터 신뢰 모드 (Low Forget Rate)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 5] Low Forget Rate (fr=0.05)\e[0m"
python Co-Correcting.py --dir experiment/petskin/clean_low_fr \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 32 \
 --lr 0.01 --lr2 0.01 --forget-rate 0.05 --lambda1 1000 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 5: `date`"


# ------------------------------------------------------------------
# 6. [Forget Rate] 노이즈 회피 모드 (High Forget Rate)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 6] High Forget Rate (fr=0.2)\e[0m"
python Co-Correcting.py --dir experiment/petskin/clean_high_fr \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 32 \
 --lr 0.01 --lr2 0.01 --forget-rate 0.2 --lambda1 1000 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 6: `date`"


# ------------------------------------------------------------------
# 7. [Lambda] 강력한 라벨 수정 (Strong Correction)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 7] Strong Label Correction (lambda=2500)\e[0m"
python Co-Correcting.py --dir experiment/petskin/clean_strong_cor \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 32 \
 --lr 0.01 --lr2 0.01 --forget-rate 0.1 --lambda1 2500 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 7: `date`"


# ------------------------------------------------------------------
# 8. [Lambda] 부드러운 라벨 수정 (Weak Correction)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 8] Weak Label Correction (lambda=500)\e[0m"
python Co-Correcting.py --dir experiment/petskin/clean_weak_cor \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 32 \
 --lr 0.01 --lr2 0.01 --forget-rate 0.1 --lambda1 500 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 8: `date`"

echo "========================================================"
echo " All Hyperparameter Tuning Experiments Finished!"
echo " End Time: `date`"
echo "========================================================"