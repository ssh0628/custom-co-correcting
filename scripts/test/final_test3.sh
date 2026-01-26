echo "========================================================"
echo " PetSkin Grand Final: The Ultimate Ensemble "
echo " Track A: Optimize Champion (Alpha 0.8) "
echo " Track B: Explore Potential (Alpha 0.7 ~ 0.2) "
echo " Start Time: `date`"
echo "========================================================"

# [Track A - 1순위] 챔피언의 스피드업 (Warm15 + Grad10 + Alpha 0.8)
# 이유: 현재 1등(Warm15+Grad30+a0.8)에서 속도만 올림. 
# 논리: "안정적인 0.8에 Grad10의 폭발력을 더하면 74%를 뚫지 않을까?"
python Co-Correcting.py --dir experiment/petskin/exp_warm15_grad10_a08 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 10 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --warmup 15 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

# [Track A - 2순위] 챔피언의 완숙미 (Warm20 + Grad30 + Alpha 0.8)
# 이유: 0.8이 1등을 했으니, 웜업을 조금 더 줘서(20) 더 완벽하게 준비시킨 뒤 출발.
# 논리: "보수적인 0.8에는 보수적인 웜업(20)이 맞을 수도 있다."
python Co-Correcting.py --dir experiment/petskin/exp_warm20_grad30_a08 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --warmup 20 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

# [Track B - 3순위] 미싱 링크 (Warm15 + Grad10 + Alpha 0.7)
# 이유: 0.8(1등)과 0.6(안정적)의 사이. 여기가 진짜 숨겨진 골든 존일 가능성 높음.
python Co-Correcting.py --dir experiment/petskin/exp_warm15_grad10_a07 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 10 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.7 \
 --warmup 15 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

# [Track B - 4순위] 검증된 밸런스 (Warm15 + Grad10 + Alpha 0.6)
# 이유: Grad30에서 0.6이 0.5를 이겼음. Grad10에서도 0.6이 0.5(74.01%)를 이길 가능성 체크.
python Co-Correcting.py --dir experiment/petskin/exp_warm15_grad10_a06 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 10 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.6 \
 --warmup 15 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

# [Track B - 5순위] 공격형 에이스 (Warm15 + Grad10 + Alpha 0.4)
# 이유: 현재 1등(a0.5)에서 Alpha만 한 단계 낮춤. 가장 안전하고 확실한 SOTA 갱신 후보.
python Co-Correcting.py --dir experiment/petskin/exp_warm15_grad10_a04 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 10 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.4 \
 --warmup 15 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

# [Track B - 6순위] 유연한 도전자 (Warm10 + Grad10 + Alpha 0.4)
# 이유: Warmup 10의 유연함 + Grad 10의 스피드 + Alpha 0.4의 합리성.
python Co-Correcting.py --dir experiment/petskin/exp_warm10_grad10_a04 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 10 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.4 \
 --warmup 10 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

# [Track B - 7순위] 정밀 타격 (Warm15 + Grad10 + Alpha 0.3)
# 이유: 0.4와 0.2 사이를 찌르는 스나이퍼 전략.
python Co-Correcting.py --dir experiment/petskin/exp_warm15_grad10_a03 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 10 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.3 \
 --warmup 15 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

# [Track B - 8순위] 혁명가 (Warm15 + Grad10 + Alpha 0.2)
# 이유: 과적합을 박살 내고 76% 이상을 노리는 승부수.
python Co-Correcting.py --dir experiment/petskin/exp_warm15_grad10_a02 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 10 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.2 \
 --warmup 15 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

# [Track A - 9순위] 다크호스 (Warm10 + Grad10 + Alpha 0.8)
# 이유: 0.8 시리즈의 마지막 퍼즐. 웜업을 줄이고 속도를 높임.
python Co-Correcting.py --dir experiment/petskin/exp_warm10_grad10_a08 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 10 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --warmup 10 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

echo "========================================================"
echo " All Experiments Completed! Good Luck! "
echo " End Time: `date`"
echo "========================================================"