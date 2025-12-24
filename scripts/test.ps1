# --------------------------------------------------------
# 1. Clean Data Experiment (Noise 0.0)
# --------------------------------------------------------
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " [Experiment 1/6] Training Clean Data (Noise 0.0)"
Write-Host " Time: $(Get-Date)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/paper/pcam/full/clean" `
    --dataset pcam `
    --noise_type clean `
    --noise 0.0 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.05 `
    --lambda1 50 `
    --lr 1e-3 `
    --lr2 1e-4

# --------------------------------------------------------
# 2. Noise 0.05 Experiment
# --------------------------------------------------------
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " [Experiment 2/6] Training Noise 0.05"
Write-Host " Time: $(Get-Date)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/paper/pcam/full/0.05" `
    --dataset pcam `
    --noise_type sn `
    --noise 0.05 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.05 `
    --lambda1 75 `
    --lr 1e-3 `
    --lr2 1e-4

# --------------------------------------------------------
# 3. Noise 0.1 Experiment
# --------------------------------------------------------
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " [Experiment 3/6] Training Noise 0.1"
Write-Host " Time: $(Get-Date)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/paper/pcam/full/0.1" `
    --dataset pcam `
    --noise_type sn `
    --noise 0.1 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.1 `
    --lambda1 100 `
    --lr 1e-3 `
    --lr2 1e-4

# --------------------------------------------------------
# 4. Noise 0.2 Experiment
# --------------------------------------------------------
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " [Experiment 4/6] Training Noise 0.2"
Write-Host " Time: $(Get-Date)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/paper/pcam/full/0.2" `
    --dataset pcam `
    --noise_type sn `
    --noise 0.2 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.2 `
    --lambda1 150 `
    --lr 1e-3 `
    --lr2 1e-4

# --------------------------------------------------------
# 5. Noise 0.3 Experiment
# --------------------------------------------------------
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " [Experiment 5/6] Training Noise 0.3"
Write-Host " Time: $(Get-Date)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/paper/pcam/full/0.3" `
    --dataset pcam `
    --noise_type sn `
    --noise 0.3 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.3 `
    --lambda1 175 `
    --lr 1e-3 `
    --lr2 1e-4

# --------------------------------------------------------
# 6. Noise 0.4 Experiment
# --------------------------------------------------------
Write-Host "`n==========================================================" -ForegroundColor Cyan
Write-Host " [Experiment 6/6] Training Noise 0.4"
Write-Host " Time: $(Get-Date)"
Write-Host "==========================================================" -ForegroundColor Cyan

python Co-Correcting.py `
    --dir "experiment/paper/pcam/full/0.4" `
    --dataset pcam `
    --noise_type sn `
    --noise 0.4 `
    --optim "SGD" `
    --mix-grad 1 `
    --discard 0 `
    --forget-rate 0.4 `
    --lambda1 200 `
    --lr 1e-3 `
    --lr2 1e-4

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " All Training Finished!"
Write-Host " End Time: $(Get-Date)"
Write-Host "==========================================================" -ForegroundColor Green