echo "========================================================"
echo " PetSkin Final Challenge "
echo " Start Time: `date`"
echo "========================================================"

python Co-Correcting.py --dir experiment/petskin/exp_warm5_ek30 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --warmup 5 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

python Co-Correcting.py --dir experiment/petskin/exp_warm5_ek10 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 10 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --warmup 5 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

python Co-Correcting.py --dir experiment/petskin/exp_warm10_ek50 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 50 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --warmup 10 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

python Co-Correcting.py --dir experiment/petskin/exp_warm15_ek30 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.8 \
 --warmup 15 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

python Co-Correcting.py --dir experiment/petskin/exp_warm10_a09 \
 --dataset petskin --dataRoot /root/project/dataset/dataset \
 --noise_type clean --noise 0.0 --optim SGD \
 --num-gradual 30 \
 --forget-rate 0.05 \
 --lambda1 400 --alpha 0.9 \
 --warmup 10 \
 --stage1 70 --stage2 200 \
 --epochs 320 \
 --data_device 1 --workers 16 --batch-size 128 \
 --lr 0.02 --lr2 0.02

echo "========================================================"
echo " All Experiments Completed!"
echo " End Time: `date`"
echo "========================================================"