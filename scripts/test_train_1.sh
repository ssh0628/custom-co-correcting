echo "========================================================"
echo " PetSkin Final Experiments: Paper Recommended Range (50~400)"
echo " Strategy: From Baseline to Optimized Co-Correcting"
echo " Start Time: `date`"
echo "========================================================"

# Exp 1: "ResNet의 본실력"을 확인 (가장 중요)
# Exp 2, 3, 4: 논문이 말하는 정석 범위(400, 200, 50)를 모두 테스트.
# Exp 5, 6, 9: "YOLO가 잘했으니 원본 라벨을 믿자(Alpha up)"는 가설 검증.

# ------------------------------------------------------------------
# [Exp 1] Baseline: Pure ResNet-50 (Standard)
# 설명: 알고리즘을 끄고 순수하게 학습 (Stage1,2를 200으로 설정)
# ------------------------------------------------------------------

echo -e "\n\e[1;46m[Exp 1] Baseline: Pure ResNet (No Correction)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp1_baseline \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.1 \
 --stage1 200 --stage2 200 --epochs 200
echo "Finished Exp 1: `date`"


# ------------------------------------------------------------------
# [Exp 2] Paper Max (논문 최대 권장값)
# 설명: Lambda 400 (노이즈가 많다고 가정할 때 쓰는 최대치)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 2] Paper Max (L=400, A=0.4)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp2_paper_max \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.1 \
 --lambda1 400 --alpha 0.4 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 2: `date`"


# ------------------------------------------------------------------
# [Exp 3] Paper Medium (논문 중간값)
# 설명: Lambda 200 (적당한 노이즈 대응)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 3] Paper Medium (L=200, A=0.4)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp3_paper_mid \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.1 \
 --lambda1 200 --alpha 0.4 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 3: `date`"


# ------------------------------------------------------------------
# [Exp 4] Paper Low (논문 최소값 - Clean 가정)
# 설명: Lambda 50 (데이터가 깨끗할 때 추천)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 4] Paper Low (L=50, A=0.4)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp4_paper_low \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.1 \
 --lambda1 50 --alpha 0.4 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 4: `date`"


echo "========================================================"
echo " All 10 Experiments Completed! (Paper Compliance Mode)"
echo " End Time: `date`"
echo "========================================================"