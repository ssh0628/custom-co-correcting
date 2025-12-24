# --------------------------------------------------------
# 1. Clean Data Experiment (Noise 0.0)
# --------------------------------------------------------
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [Test 1/6] Training Clean Data (Noise 0.0)"
Write-Host " Setting: Epochs 1 (Quick Test)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/test/pcam/clean" `
    --dataset pcam `
    --noise_type clean `
    --noise 0.0 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.05 `
    --lambda1 50 `
    --lr 1e-3 `
    --lr2 1e-4 `
    --epochs 1

# --------------------------------------------------------
# 2. Noise 0.05 Experiment
# --------------------------------------------------------
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " [Test 2/6] Training Noise 0.05"
Write-Host " Setting: Epochs 1 (Quick Test)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/test/pcam/0.05" `
    --dataset pcam `
    --noise_type sn `
    --noise 0.05 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.05 `
    --lambda1 75 `
    --lr 1e-3 `
    --lr2 1e-4 `
    --epochs 1

# --------------------------------------------------------
# 3. Noise 0.1 Experiment
# --------------------------------------------------------
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " [Test 3/6] Training Noise 0.1"
Write-Host " Setting: Epochs 1 (Quick Test)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/test/pcam/0.1" `
    --dataset pcam `
    --noise_type sn `
    --noise 0.1 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.1 `
    --lambda1 100 `
    --lr 1e-3 `
    --lr2 1e-4 `
    --epochs 1

# --------------------------------------------------------
# 4. Noise 0.2 Experiment
# --------------------------------------------------------
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " [Test 4/6] Training Noise 0.2"
Write-Host " Setting: Epochs 1 (Quick Test)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/test/pcam/0.2" `
    --dataset pcam `
    --noise_type sn `
    --noise 0.2 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.2 `
    --lambda1 150 `
    --lr 1e-3 `
    --lr2 1e-4 `
    --epochs 1

# --------------------------------------------------------
# 5. Noise 0.3 Experiment
# --------------------------------------------------------
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " [Test 5/6] Training Noise 0.3"
Write-Host " Setting: Epochs 1 (Quick Test)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/test/pcam/0.3" `
    --dataset pcam `
    --noise_type sn `
    --noise 0.3 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.3 `
    --lambda1 175 `
    --lr 1e-3 `
    --lr2 1e-4 `
    --epochs 1

# --------------------------------------------------------
# 6. Noise 0.4 Experiment
# --------------------------------------------------------
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " [Test 6/6] Training Noise 0.4"
Write-Host " Setting: Epochs 1 (Quick Test)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/test/pcam/0.4" `
    --dataset pcam `
    --noise_type sn `
    --noise 0.4 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.4 `
    --lambda1 200 `
    --lr 1e-3 `
    --lr2 1e-4 `
    --epochs 1

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " All Tests Finished Successfully!"
Write-Host "==========================================================" -ForegroundColor Green

if ($LASTEXITCODE -ne 0) { Write-Host "중간에 오류가 발생했습니다." -ForegroundColor Red }