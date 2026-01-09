echo "========================================================"
echo " PetSkin Optimization: Lambda(400), Alpha(0.8), Forget_Rate(0.05)"
echo " Start Time: `date`"
echo "========================================================"

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
echo " Optimization Experiments Completed!"
echo " End Time: `date`"
echo "========================================================"