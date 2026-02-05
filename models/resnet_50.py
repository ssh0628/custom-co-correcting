import os
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm import create_model
from timm.data import resolve_data_config
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# =========================
# 0. PIL / CUDA / AMP Settings
# =========================
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# AMP config
amp_device = "cuda" if device.type == "cuda" else "cpu"
use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
print(f"AMP Config: device={amp_device}, dtype={amp_dtype}")

use_scaler = (device.type == "cuda" and amp_dtype == torch.float16)
scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
print(f"GradScaler enabled: {use_scaler}")

# =========================
# 1. Settings
# =========================
NPY_DIR = "/root/project/dataset/cache_npy"
SAVE_DIR = "/root/project/resnet_50/resnet_50_relabeled"
os.makedirs(SAVE_DIR, exist_ok=True)
LOG_PATH = os.path.join(SAVE_DIR, "train_log.csv")

BATCH_SIZE = 256
NUM_EPOCHS = 100
IMG_SIZE = 224
RANDOM_SEED = 0
PATIENCE = 20

# Freeze/Unfreeze schedule
FREEZE_EPOCHS = 5

WEIGHT_DECAY = 0.01
LR_FREEZE = 1e-3   # head-only
LR_UNFREEZE = 1e-4 # full finetune

PRETRAINED = True
MODEL_NAME = "resnet50.a1_in1k"
DROP_PATH_RATE = 0.0  # ResNet에는 보통 0.0 권장

WORKERS = 16

# =========================
# Seed
# =========================
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

print(f"[Info] Reading Data from: {NPY_DIR}")

# =========================
# 1.1 Class Check
# =========================
cls_json_path = os.path.join(NPY_DIR, "classes.json")
if not os.path.exists(cls_json_path):
    raise FileNotFoundError(f"[ERR] classes.json not found in {NPY_DIR}")

with open(cls_json_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
classes = meta.get("classes", [])
if not classes:
    raise ValueError("[ERR] classes.json has empty classes")
NUM_CLASSES = len(classes)
print(f"[OK] NUM_CLASSES={NUM_CLASSES}, classes={classes}")

# =========================
# 2. Dataset (NPY paths + labels)
# =========================
class NPYPathDataset(Dataset):
    def __init__(self, npy_dir, split, transform=None, fallback_size=224):
        super().__init__()
        self.transform = transform
        self.npy_dir = Path(npy_dir)
        self.split = split
        self.fallback_size = int(fallback_size)

        paths_file = self.npy_dir / f"{split}_paths.npy"
        labels_file = self.npy_dir / f"{split}_labels.npy"
        if not paths_file.exists() or not labels_file.exists():
            raise RuntimeError(f"[ERR] Missing NPY files for split '{split}' in {npy_dir}")

        # allow_pickle=False 권장
        self.paths = np.load(paths_file, allow_pickle=False)
        self.labels = np.load(labels_file, allow_pickle=False).astype(np.int64)

        if len(self.paths) != len(self.labels):
            raise ValueError(f"[ERR] paths/labels length mismatch: {len(self.paths)} vs {len(self.labels)}")

        print(f"[{split.upper()}] Loaded {len(self.paths)} samples.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        path = p.item() if isinstance(p, np.generic) else str(p)
        label = int(self.labels[idx])

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.fallback_size, self.fallback_size))

        if self.transform:
            img = self.transform(img)

        return img, label

# =========================
# 3. Model
# =========================
model = create_model(
    MODEL_NAME,
    pretrained=PRETRAINED,
    num_classes=NUM_CLASSES,
    drop_path_rate=DROP_PATH_RATE
).to(device)

data_config = resolve_data_config({}, model=model)

# ---- freeze backbone safely (ResNet: fc/classifier/head 등)
HEAD_KEYS = ("fc", "classifier", "head")

def freeze_backbone(m: torch.nn.Module):
    for n, p in m.named_parameters():
        p.requires_grad = any(k in n for k in HEAD_KEYS)

def unfreeze_all(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = True

freeze_backbone(model)
print("[Info] Backbone frozen (head-only training).")

# =========================
# 4. Transform (ConvNeXt 기준과 동일하게)
# =========================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.2, 0.75, 0.25, 0.04),
    transforms.ToTensor(),
    transforms.Normalize(data_config["mean"], data_config["std"]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(data_config["mean"], data_config["std"]),
])

# =========================
# 5. DataLoaders
# =========================
train_dataset = NPYPathDataset(NPY_DIR, "train", transform=train_transform, fallback_size=IMG_SIZE)
val_dataset   = NPYPathDataset(NPY_DIR, "val",   transform=eval_transform,  fallback_size=IMG_SIZE)
test_dataset  = NPYPathDataset(NPY_DIR, "test",  transform=eval_transform,  fallback_size=IMG_SIZE)

pin = (device.type == "cuda")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=WORKERS, pin_memory=pin)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=WORKERS, pin_memory=pin)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=WORKERS, pin_memory=pin)

# =========================
# 6. Optimizer / Scheduler
# =========================
criterion = nn.CrossEntropyLoss()

def make_optimizer(lr: float):
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )

optimizer = make_optimizer(LR_FREEZE)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# =========================
# 7. Train Loop
# =========================
with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"])

best_val = 0.0
patience_counter = 0

best_path = os.path.join(SAVE_DIR, "best_model.pth")

for epoch in range(NUM_EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")

    # Unfreeze
    if epoch == FREEZE_EPOCHS:
        print(f">>> Unfreezing all layers at epoch={epoch+1}")
        unfreeze_all(model)
        optimizer = optim.AdamW(model.parameters(), lr=LR_UNFREEZE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, NUM_EPOCHS - epoch), eta_min=1e-6)

    # ---- Train ----
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in tqdm(train_loader, desc="Train"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            out = model(x)
            loss = criterion(out, y)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        loss_sum += loss.item() * bs
        correct += (out.argmax(1) == y).sum().item()
        total += bs

    train_loss = loss_sum / total
    train_acc = correct / total

    scheduler.step()

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Val"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
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

    print(f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} | ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} | LR={lr:.2e}")

    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([epoch+1, lr, train_loss, train_acc, val_loss, val_acc])

    # Save best
    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), best_path)
        patience_counter = 0
        print(f"[BEST] New best val_acc={best_val:.4f} -> saved {best_path}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"[EARLY STOP] at epoch={epoch+1} (patience={PATIENCE})")
            break

# =========================
# 8. Test
# =========================
print(f"\n[Test] Loading best model: {best_path}")
state = torch.load(best_path, map_location="cpu")
model.load_state_dict(state, strict=True)
model.to(device)
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Test"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f"Test Accuracy: {correct / total:.4f}")
print(f"[SAVED] train_log.csv: {LOG_PATH}")
print(f"[SAVED] best_model.pth: {best_path}")