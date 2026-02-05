import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from timm import create_model
from timm.data import resolve_data_config
from PIL import ImageFile
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# [ALIGNMENT] Add project root to path to import dataset.PetSkin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.PetSkin import PetSkin

# =========================
# 0. PIL / CUDA Settings
# =========================
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# 1. Settings
# =========================
DATA_DIR = "/root/project/dataset/dataset"
SAVE_DIR = "/root/project/convnext/convnextv2"
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_PATH = os.path.join(SAVE_DIR, "train_log.csv")

NUM_CLASSES = 6
BATCH_SIZE = 256
NUM_EPOCHS = 200
IMG_SIZE = 224

# IMPORTANT: Used 0 to match cache_npy filenames (e.g., seed0)
RANDOM_SEED = 0

pretrained = True
model_name = "convnextv2_tiny.fcmae_ft_in22k_in1k"
drop_path_rate = 0.1

print(f"[Info] Expect cache files like: petskin_train_seed{RANDOM_SEED}_size{IMG_SIZE}x{IMG_SIZE}_len*.npy")

# =========================
# 2. Model
# =========================
model = create_model(
    model_name,
    pretrained=pretrained,
    num_classes=NUM_CLASSES,
    drop_path_rate=drop_path_rate
).to(device)

# Backbone freeze (Initial)
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

data_config = resolve_data_config({}, model=model)

# =========================
# 3. Transform
# =========================
# [ALIGNMENT] IMPORTANT:
# PetSkin.img_loader() already does bbox-crop + resize to (IMG_SIZE, IMG_SIZE) using Image.NEAREST
# and saves/loads from cache_npy. So DO NOT resize again here.
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.2, 0.75, 0.25, 0.04),
    transforms.ToTensor(),
    transforms.Normalize(data_config['mean'], data_config['std']),
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(data_config['mean'], data_config['std']),
])

# =========================
# 4. Dataset & Split
# =========================
print(f"[Info] Loading PetSkin Dataset with Seed {RANDOM_SEED}")
train_dataset = PetSkin(
    root=DATA_DIR,
    train=0,  # train
    transform=train_transform,
    noise_type='clean',
    noise_rate=0.0,
    device=1,  # cache mode
    split_ratios=(0.7, 0.15, 0.15),
    random_seed=RANDOM_SEED
)

test_dataset = PetSkin(
    root=DATA_DIR,
    train=1,  # test
    transform=eval_transform,
    noise_type='clean',
    noise_rate=0.0,
    device=1,  # cache mode
    split_ratios=(0.7, 0.15, 0.15),
    random_seed=RANDOM_SEED
)

val_dataset = PetSkin(
    root=DATA_DIR,
    train=2,  # val
    transform=eval_transform,
    noise_type='clean',
    noise_rate=0.0,
    device=1,  # cache mode
    split_ratios=(0.7, 0.15, 0.15),
    random_seed=RANDOM_SEED
)

# =========================
# 5. DataLoader
# =========================
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=16, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

# =========================
# 6. Optimizer / Scheduler
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=0.05
)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# =========================
# Initialize CSV Log
# =========================
with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

# =========================
# 7. Train Loop
# =========================
best_val = 0.0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")

    # Backbone unfreeze (same as baseline)
    if epoch == 15:
        print(">>> Unfreezing backbone")
        for p in model.parameters():
            p.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS-epoch, eta_min=1e-6)

    # ---- Train ----
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y, idx in tqdm(train_loader, desc="Train"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        correct += (out.argmax(1) == y).sum().item()
        total += bs

    train_loss = loss_sum / total
    train_acc = correct / total

    # Step at end of epoch (baseline/standard pattern)
    scheduler.step()

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for x, y, idx in tqdm(val_loader, desc="Val"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = criterion(out, y)
                pred = out.argmax(1)

            bs = y.size(0)
            val_loss_sum += loss.item() * bs
            correct += (pred == y).sum().item()
            total += bs

    val_acc = correct / total
    val_loss = val_loss_sum / total
    lr = optimizer.param_groups[0]["lr"]

    print(f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} LR={lr:.2e}")

    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([epoch+1, lr, train_loss, train_acc, val_loss, val_acc])

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 30:
            print(f"Early Stopping triggered at Epoch {epoch+1}")
            break

# =========================
# 8. Test
# =========================
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth"), map_location="cpu"))
model.to(device)
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for x, y, idx in tqdm(test_loader, desc="Test"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"Test Accuracy: {correct / total:.4f}")