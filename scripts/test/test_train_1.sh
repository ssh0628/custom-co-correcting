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

# ------------------------------------------------------------------
# [Exp 4-1] Paper Max Over (논문 최대 권장값 이상)
# 설명: Lambda 600 (노이즈가 많다고 가정할 때 쓰는 최대치)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 2] Paper Max (L=600, A=0.4)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp2_1_paper_max_over \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.1 \
 --lambda1 600 --alpha 0.4 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 2: `date`"

# ------------------------------------------------------------------
# [Exp 5] Alpha UP (A=0.8)
# 목적: "데이터가 깨끗하니 원본 라벨을 80% 믿어보자"
# 비교: Exp 2(A=0.4) vs Exp 11(A=0.8)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 11] Alpha UP (L=400, A=0.8, FR=0.1)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp5_alpha_up \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.1 \
 --lambda1 400 --alpha 0.8 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 11: `date`"

# ------------------------------------------------------------------
# [Exp 6] Alpha DOWN (A=0.1)
# 목적: "혹시 모르니 모델에게 수정의 자유를 더 줘보자"
# 비교: Exp 2(A=0.4) vs Exp 12(A=0.1)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 12] Alpha DOWN (L=400, A=0.1, FR=0.1)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp6_alpha_down \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.1 \
 --lambda1 400 --alpha 0.1 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 12: `date`"

# ------------------------------------------------------------------
# [Exp 7] Forget Rate DOWN (FR=0.05)
# 목적: "데이터가 깨끗하니 5%만 버리고 다 살려보자"
# 비교: Exp 2(FR=0.1) vs Exp 13(FR=0.05)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 13] Noise Rate DOWN (L=400, A=0.4, FR=0.05)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp7_fr_down \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.4 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 13: `date`"

# ------------------------------------------------------------------
# [Exp 8] Forget Rate UP (FR=0.2)
# 목적: "혹시 20% 정도는 버려야 성능이 오를까? (검증용)"
# 비교: Exp 2(FR=0.1) vs Exp 14(FR=0.2)
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 14] Noise Rate UP (L=400, A=0.4, FR=0.2)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp8_fr_up \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.2 \
 --lambda1 400 --alpha 0.4 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 14: `date`"

# ------------------------------------------------------------------
# [Exp 9] The Hybrid Best (조합)
# 목적: "Alpha 높이고(Clean), FR 낮추고(Clean)" -> 이론상 최강 조합
# 설정: L=400, A=0.8, FR=0.05
# ------------------------------------------------------------------
echo -e "\n\e[1;46m[Exp 15] Hybrid Best (L=400, A=0.8, FR=0.05)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp9_hybrid_best \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --stage1 50 --stage2 150 --epochs 200
echo "Finished Exp 15: `date`"

echo -e "\n\e[1;46m[Exp 0] True Baseline: Pure ResNet \e[0m"
python Co-Correcting.py --dir experiment/petskin/exp1_true_baseline \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 0 --forget-rate 0.0 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --stage1 200 --stage2 200 --epochs 200

echo -e "\n\e[1;46m[Exp 5-1] Alpha UP (L=400, A=0.8, FR=0.1, epoch=320)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp5-1_alpha_up \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.1 \
 --lambda1 400 --alpha 0.8 \
 --stage1 70 --stage2 200 --epochs 320
echo "Finished Exp 5-1: `date`"


echo -e "\n\e[1;46m[Exp 10] Hybrid Best (L=400, A=0.8, FR=0.05, epoch=320)\e[0m"
python Co-Correcting.py --dir experiment/petskin/exp10_hybrid_best_epoch_up \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --stage1 70 --stage2 200 --epochs 320
echo "Finished Exp 10: `date`"

echo -e "\n\e[1;46m[Exp 11] Exp10 + ASAM + ANL \e[0m"
python Co-Correcting.py --dir experiment/petskin/exp11_best_anl_asam \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim ASAM --num-gradual 10 --exponent 1 \
 --forget-type coteaching_plus --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 --cost_type anl \
 --stage1 70 --stage2 200 --epochs 320
 echo "Finished Exp 11: `date`"

 

echo "========================================================"
echo " All 10 Experiments Completed! (Paper Compliance Mode)"
echo " End Time: `date`"
echo "========================================================"