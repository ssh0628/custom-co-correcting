#!/bin/bash

# ==========================================
BEST_GRADUAL=30
BEST_FR=0.05
# ==========================================

# --- [고정 파라미터] ---
DATA_ROOT="/root/project/dataset/dataset"
STAGE1=70
STAGE2=200
WARMUP=15
ALPHA=0.5
SEED=0

# Lambda 200
python Co-Correcting.py --dir experiment/petskin/tune/step3_lam200 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --warmup ${WARMUP} \
 --num-gradual ${BEST_GRADUAL} \
 --forget-rate ${BEST_FR} \
 --lambda1 200 \
 --alpha ${ALPHA} \
 --random-seed ${SEED}

# Lambda 100
python Co-Correcting.py --dir experiment/petskin/tune/step3_lam100 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --warmup ${WARMUP} \
 --num-gradual ${BEST_GRADUAL} \
 --forget-rate ${BEST_FR} \
 --lambda1 100 \
 --alpha ${ALPHA} \
 --random-seed ${SEED}

# Lambda 600
python Co-Correcting.py --dir experiment/petskin/tune/step3_lam600 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --warmup ${WARMUP} \
 --num-gradual ${BEST_GRADUAL} \
 --forget-rate ${BEST_FR} \
 --lambda1 600 \
 --alpha ${ALPHA} \
 --random-seed ${SEED}