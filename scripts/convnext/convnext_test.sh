#!/bin/bash

DATA_ROOT="/root/project/dataset/dataset"
BEST_GRADUAL=5
BEST_FR=0.05
ALPHA=0.3
WEIGHT_DECAY=0.1
WARMUP=5
STAGE1=15
STAGE2=60
EPOCH=100

for s in 0 1 2
do
    EXP_NAME="pca256_wd1e2_seed${s}"
    
    echo "========================================================"
    echo "Running: ConvNeXt | Seed=${s} | Optim=AdamW | Scheduler=Cosine"
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
        --batch-size 128 \
        --workers 16 \
        --lr 0.00005 \
        --lr2 0.00001 \
        --stage1 ${STAGE1} \
        --stage2 ${STAGE2} \
        --warmup ${WARMUP} \
        --num-gradual ${BEST_GRADUAL} \
        --forget-rate ${BEST_FR} \
        --alpha ${ALPHA} \
        --lambda1 400 \
        --random-seed ${s}
done
