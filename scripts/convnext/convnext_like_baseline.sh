#!/bin/bash

# best co-correcting hyperparameters
ALPHA=0.3
BEST_FR=0.05
LAMBDA=400
# ======================================

# should be changed
BEST_GRADUAL=10
WARMUP=10
STAGE1=30
STAGE2=70
EPOCH=150
# ======================================


DATA_ROOT="/root/project/dataset/dataset"
WEIGHT_DECAY=0.01 # baseline
DROP_PATH_RATE=0.1
SEED=42

EXP_NAME="convnext_like_baseline"
echo "========================================================"
echo "Running: ConvNeXt | Seed=${SEED} | Optim=AdamW | Scheduler=Cosine"
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
    --num-gradual ${BEST_GRADUAL} \
    --forget-rate ${BEST_FR} \
    --alpha ${ALPHA} \
    --lambda1 ${LAMBDA} \
    --random-seed ${SEED}


echo "========================================================"
echo "Finished: ConvNeXt | Seed=${SEED} | Optim=AdamW | Scheduler=Cosine"
echo "Saved Directory: experiment/convnext/${EXP_NAME}"
echo "========================================================"