#!/bin/bash
DATA_ROOT="/root/project/dataset/dataset"
STAGE1=70
STAGE2=200
WARMUP=15
FR=0.05
LAMBDA=400
ALPHA=0.5
SEED=0

# Gradual 20
python Co-Correcting.py --dir experiment/petskin/tune/step1_grad20 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --warmup ${WARMUP} \
 --num-gradual 20 \
 --random-seed ${SEED} \
 --forget-rate ${FR} --lambda1 ${LAMBDA} --alpha ${ALPHA}

# Gradual 10
python Co-Correcting.py --dir experiment/petskin/tune/step1_grad10 \
 --dataset petskin --dataRoot ${DATA_ROOT} \
 --noise_type clean --noise 0.0 --optim SGD \
 --stage1 ${STAGE1} --stage2 ${STAGE2} \
 --epochs 320 --batch-size 128 --workers 16 --lr 0.02 --lr2 0.02 \
 --warmup ${WARMUP} \
 --num-gradual 10 \
 --random-seed ${SEED} \
 --forget-rate ${FR} --lambda1 ${LAMBDA} --alpha ${ALPHA}