#!/bin/bash

echo "========================================================"
echo " PetSkin Final Challenge: Breaking the 70% Wall"
echo " Strategy: Forget Rate 0.0 (All Data) + Co-Correction"
echo " Start Time: `date`"
echo "========================================================"

# 공통 설정 변수 (수정하기 편하게 분리)
DATASET="petskin"
DATAROOT="/root/project/dataset/dataset"
WORKERS=16
BATCH_SIZE=128
EPOCHS=320               # 충분한 학습 시간 보장
LR=0.02

# ------------------------------------------------------------------
# [Exp 12] New Hybrid SOTA (가장 유력한 우승 후보)
# 전략: ResNet의 물량공세(FR=0) + 검증된 교정값(A=0.8, L=400)
# 기대: 노이즈까지 다 먹은 ResNet(70.5%)보다, 고쳐서 먹은 모델이 더 높아야 함.
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 12] New Hybrid SOTA (FR=0.0, A=0.8, L=400)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp12_new_hybrid_sota \
 --dataset ${DATASET} --dataRoot ${DATAROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 0 \
 --forget-rate 0.0 \
 --lambda1 400 --alpha 0.8 \
 --stage1 70 --stage2 200 --epochs ${EPOCHS} \
 --data_device 1 --workers ${WORKERS} --batch-size ${BATCH_SIZE} \
 --lr ${LR} --lr2 ${LR}
echo "Finished Exp 12: `date`"


# ------------------------------------------------------------------
# [Exp 13] Aggressive Correction (노이즈가 많을 땐 강하게!)
# 전략: FR=0이라 노이즈가 많이 들어오니, Lambda를 2배(800)로 높여서 빨리 고친다.
# 비교: Exp 12(L=400) vs Exp 13(L=800)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 13] Aggressive Correction (FR=0.0, A=0.8, L=800)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp13_aggressive_correction \
 --dataset ${DATASET} --dataRoot ${DATAROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 0 \
 --forget-rate 0.0 \
 --lambda1 800 --alpha 0.8 \
 --stage1 70 --stage2 200 --epochs ${EPOCHS} \
 --data_device 1 --workers ${WORKERS} --batch-size ${BATCH_SIZE} \
 --lr ${LR} --lr2 ${LR}
echo "Finished Exp 13: `date`"


# ------------------------------------------------------------------
# [Exp 14] Super Conservative (철벽 방어 - 원본 수호)
# 전략: 원본 라벨 신뢰도를 90%(0.9)로 높여서, 함부로 라벨을 못 고치게 막음.
# 이유: Baseline(원본 100% 신뢰)이 잘했으니, 우리도 원본을 최대한 존중하되 '진짜 이상한 것'만 살짝 고치자.
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 14] Super Conservative (FR=0.0, A=0.9, L=400)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp14_super_conservative \
 --dataset ${DATASET} --dataRoot ${DATAROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 0 \
 --forget-rate 0.0 \
 --lambda1 400 --alpha 0.9 \
 --stage1 70 --stage2 200 --epochs ${EPOCHS} \
 --data_device 1 --workers ${WORKERS} --batch-size ${BATCH_SIZE} \
 --lr ${LR} --lr2 ${LR}
echo "Finished Exp 14: `date`"

echo "========================================================"
echo " All Experiments Completed!"
echo " End Time: `date`"
echo "========================================================"