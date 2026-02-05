#!/bin/bash
set -e

DATA_ROOT="/root/project/dataset/cache_npy"
WEIGHT_DECAY=0.1
DROP_PATH_RATE=0.2
SEED=0

# Common
BEST_GRADUAL=10
WARMUP=5
EPOCH=200

echo "========================================================"
echo "Running ALL: ConvNeXt checks | Seed=${SEED}"
echo "Base Dir: experiment/convnext/"
echo "========================================================"


# =========================================================
# 1) CE ONLY
# =========================================================
ALPHA=0.0
BEST_FR=0.0
LAMBDA=0

STAGE1=101
STAGE2=101

EXP_NAME="convnext_check_1_CE"
echo "========================================================"
echo "[1/3] Running: ConvNeXt | CE ONLY | Seed=${SEED}"
echo "Save Directory: experiment/convnext/${EXP_NAME}"
echo "========================================================"

python custom-co-correcting.py --dir experiment/convnext/${EXP_NAME} \
    --backbone convnext \
    --dataset petskin \
    --dataRoot ${DATA_ROOT} \
    --noise_type clean --noise 0.0 \
    --optim AdamW \
    --scheduler Cosine \
    --weight-decay ${WEIGHT_DECAY} \
    --epochs ${EPOCH} \
    --batch-size 256 \
    --workers 16 \
    --lr 0.001 \
    --lr2 0.00003 \
    --drop-path-rate ${DROP_PATH_RATE} \
    --stage1 ${STAGE1} \
    --stage2 ${STAGE2} \
    --warmup ${WARMUP} \
    --pretrained 1 \
    --num-gradual ${BEST_GRADUAL} \
    --forget-rate ${BEST_FR} \
    --forget-type coteaching \
    --alpha ${ALPHA} \
    --lambda1 ${LAMBDA} \
    --random-seed ${SEED}

echo "Finished: ${EXP_NAME}"
echo ""


# =========================================================
# 2) AGREEMENT ONLY (closest possible under your code)
# =========================================================
ALPHA=0.0
BEST_FR=0.03
LAMBDA=0
BETA=0.0

STAGE1=101
STAGE2=101

EXP_NAME="convnext_check_2_AGREEMENT"
echo "========================================================"
echo "[2/3] Running: ConvNeXt | AGREEMENT ONLY | Seed=${SEED}"
echo "Save Directory: experiment/convnext/${EXP_NAME}"
echo "========================================================"

python custom-co-correcting.py --dir experiment/convnext/${EXP_NAME} \
    --backbone convnext \
    --dataset petskin \
    --dataRoot ${DATA_ROOT} \
    --noise_type clean --noise 0.0 \
    --optim AdamW \
    --scheduler Cosine \
    --weight-decay ${WEIGHT_DECAY} \
    --epochs ${EPOCH} \
    --batch-size 256 \
    --workers 16 \
    --lr 0.001 \
    --lr2 0.00003 \
    --drop-path-rate ${DROP_PATH_RATE} \
    --stage1 ${STAGE1} \
    --stage2 ${STAGE2} \
    --warmup ${WARMUP} \
    --pretrained 1 \
    --num-gradual ${BEST_GRADUAL} \
    --forget-rate ${BEST_FR} \
    --forget-type coteaching_plus \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --lambda1 ${LAMBDA} \
    --random-seed ${SEED}

echo "Finished: ${EXP_NAME}"
echo ""


# =========================================================
# 3) LABEL CORRECTION
# =========================================================
ALPHA=0.8
BEST_FR=0.03
LAMBDA=100
BETA=0.1

EPOCH=200
STAGE1=70
STAGE2=150

EXP_NAME="convnext_check_3_LABEL_CORRECTION"
echo "========================================================"
echo "[3/3] Running: ConvNeXt | LABEL CORRECTION ONLY | Seed=${SEED}"
echo "Save Directory: experiment/convnext/${EXP_NAME}"
echo "========================================================"

python custom-co-correcting.py --dir experiment/convnext/${EXP_NAME} \
    --backbone convnext \
    --dataset petskin \
    --dataRoot ${DATA_ROOT} \
    --noise_type clean --noise 0.0 \
    --optim AdamW \
    --scheduler Cosine \
    --weight-decay ${WEIGHT_DECAY} \
    --epochs ${EPOCH} \
    --batch-size 256 \
    --workers 16 \
    --lr 0.001 \
    --lr2 0.00003 \
    --drop-path-rate ${DROP_PATH_RATE} \
    --stage1 ${STAGE1} \
    --stage2 ${STAGE2} \
    --warmup ${WARMUP} \
    --pretrained 1 \
    --num-gradual ${BEST_GRADUAL} \
    --forget-rate ${BEST_FR} \
    --forget-type coteaching_plus \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --lambda1 ${LAMBDA} \
    --random-seed ${SEED}

echo "Finished: ${EXP_NAME}"
echo ""


echo "========================================================"
echo "ALL DONE | Seed=${SEED}"
echo "Saved under: experiment/convnext/"
echo "========================================================"