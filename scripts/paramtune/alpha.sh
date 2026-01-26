#!/bin/bash

# ==========================================
BEST_GRADUAL=30
BEST_FR=0.05
BEST_LAMBDA=400
# ==========================================

# --- [고정 파라미터] ---
DATA_ROOT="/root/project/dataset/dataset"
STAGE1=70
STAGE2=200
WARMUP=15
SEED=0

# Alpha 0.8
python Co-Correcting.py --dir experiment/petskin/tune/step4_alpha0.8 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --warmup ${WARMUP} \
 --num-gradual ${BEST_GRADUAL} \
 --forget-rate ${BEST_FR} \
 --lambda1 ${BEST_LAMBDA} \
 --alpha 0.8 \
 --random-seed ${SEED}

# Alpha 0.6
python Co-Correcting.py --dir experiment/petskin/tune/step4_alpha0.6 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --warmup ${WARMUP} \
 --num-gradual ${BEST_GRADUAL} \
 --forget-rate ${BEST_FR} \
 --lambda1 ${BEST_LAMBDA} \
 --alpha 0.6 \
 --random-seed ${SEED}

# Alpha 0.4
python Co-Correcting.py --dir experiment/petskin/tune/step4_alpha0.4 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --warmup ${WARMUP} \
 --num-gradual ${BEST_GRADUAL} \
 --forget-rate ${BEST_FR} \
 --lambda1 ${BEST_LAMBDA} \
 --alpha 0.4 \
 --random-seed ${SEED}

# Alpha 0.2
python Co-Correcting.py --dir experiment/petskin/tune/step4_alpha0.2 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --warmup ${WARMUP} \
 --num-gradual ${BEST_GRADUAL} \
 --forget-rate ${BEST_FR} \
 --lambda1 ${BEST_LAMBDA} \
 --alpha 0.2 \
 --random-seed ${SEED}