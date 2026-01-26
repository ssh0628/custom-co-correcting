#!/bin/bash
# ==========================================
BEST_GRADUAL=30
# ==========================================

# --- [고정 파라미터] ---
DATA_ROOT="/root/project/dataset/dataset"
STAGE1=70
STAGE2=200
WARMUP=15
LAMBDA=400 
ALPHA=0.5
SEED=0

# Forget Rate 0.05
python Co-Correcting.py --dir experiment/petskin/tune/step2_fr0.05 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --warmup ${WARMUP} \
 --num-gradual ${BEST_GRADUAL} \
 --forget-rate 0.05 \
 --lambda1 ${LAMBDA} --alpha ${ALPHA} \
 --random-seed ${SEED}

# Forget Rate 0.1
python Co-Correcting.py --dir experiment/petskin/tune/step2_fr0.1 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --warmup ${WARMUP} \
 --num-gradual ${BEST_GRADUAL} \
 --forget-rate 0.1 \
 --lambda1 ${LAMBDA} --alpha ${ALPHA} \
 --random-seed ${SEED}

# Forget Rate 0.15
python Co-Correcting.py --dir experiment/petskin/tune/step2_fr0.15 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --warmup ${WARMUP} \
 --num-gradual ${BEST_GRADUAL} \
 --forget-rate 0.15 \
 --lambda1 ${LAMBDA} --alpha ${ALPHA} \
 --random-seed ${SEED}