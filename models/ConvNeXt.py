import os
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from timm import create_model
from timm.data import resolve_data_config
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# =========================
# 0. PIL / CUDA 설정
# =========================
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# =========================
# 1. 설정
# =========================
DATA_DIR = "/root/project/dataset/dataset"
SAVE_DIR = "results_convnextv2_fcmae_finetune2"
CACHE_DIR = os.path.join(DATA_DIR, "cache_npy")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

LOG_PATH = os.path.join(SAVE_DIR, "train_log.csv")

NUM_CLASSES = 6
BATCH_SIZE = 256
NUM_EPOCHS = 200
IMG_SIZE = 224
RANDOM_SEED = 42

pretrained = True
model_name = "convnextv2_tiny.fcmae_ft_in22k_in1k"
drop_path_rate = 0.2

class_names = [f"A{i}" for i in range(1, NUM_CLASSES+1)]
class_to_idx = {c: i for i, c in enumerate(class_names)}

# =========================
# 2. Dataset (NPY 캐시)
# =========================
class PetSkinNPYDataset(Dataset):
    def __init__(self, root):
        self.resize_dim = (IMG_SIZE, IMG_SIZE)
        self.img_paths, self.labels = [], []

        for cls in class_names:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir): continue
            for r, _, files in os.walk(cls_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.img_paths.append(os.path.join(r, f))
                        self.labels.append(class_to_idx[cls])

        self.img_paths = np.array(self.img_paths)
        self.labels = np.array(self.labels)

        cache_name = f"petskin_all_{IMG_SIZE}x{IMG_SIZE}_len{len(self.img_paths)}.npy"
        self.cache_path = os.path.join(CACHE_DIR, cache_name)

        if os.path.exists(self.cache_path):
            self.images = np.load(self.cache_path, mmap_mode='r')
        else:
            imgs = []
            for p in tqdm(self.img_paths, desc="Caching Images"):
                img = Image.open(p).convert("RGB").resize(self.resize_dim)
                imgs.append(np.asarray(img, dtype=np.uint8))
            np.save(self.cache_path, np.stack(imgs))
            self.images = np.load(self.cache_path, mmap_mode='r')

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx):
        return Image.fromarray(self.images[idx]), torch.tensor(self.labels[idx])

# =========================
# 3. Model
# =========================
model = create_model(
    model_name,
    pretrained=pretrained,
    num_classes=NUM_CLASSES,
    drop_path_rate=drop_path_rate
).to(device)

# Backbone freeze (초기)
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

data_config = resolve_data_config({}, model=model)

# =========================
# 4. Transform
# =========================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ColorJitter(0.2,0.75,0.25,0.04),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(data_config['mean'], data_config['std']),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(data_config['mean'], data_config['std']),
])

class Wrap(Dataset):
    def __init__(self, ds, tfm): self.ds, self.tfm = ds, tfm
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x, y = self.ds[i]
        return self.tfm(x), y

# =========================
# 5. Split
# =========================
base_dataset = PetSkinNPYDataset(DATA_DIR)
indices = np.arange(len(base_dataset))
labels = base_dataset.labels

sss1 = StratifiedShuffleSplit(1, test_size=0.3, random_state=RANDOM_SEED)
train_idx, temp_idx = next(sss1.split(indices, labels))

sss2 = StratifiedShuffleSplit(1, test_size=1/3, random_state=RANDOM_SEED)
val_idx, test_idx = next(sss2.split(temp_idx, labels[temp_idx]))

train_loader = DataLoader(Wrap(Subset(base_dataset, train_idx), train_transform),
                          BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(Wrap(Subset(base_dataset, temp_idx[val_idx]), eval_transform),
                        BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
test_loader = DataLoader(Wrap(Subset(base_dataset, temp_idx[test_idx]), eval_transform),
                         BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)

# =========================
# 6. Optimizer / Scheduler
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=1e-3, weight_decay=0.05)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
scaler = torch.cuda.amp.GradScaler()

# =========================
# CSV 로그 초기화
# =========================
with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_acc"])

# =========================
# 7. Train Loop
# =========================
best_val = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{NUM_EPOCHS} =====")

    # Backbone unfreeze
    if epoch == 15:
        print(">>> Unfreezing backbone")
        for p in model.parameters():
            p.requires_grad = True
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.05)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS-epoch, eta_min=1e-6)

    # ---- Train ----
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in tqdm(train_loader, desc="Train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Val"):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_acc = correct / total
    lr = optimizer.param_groups[0]["lr"]

    print(f"TrainAcc={train_acc:.4f} ValAcc={val_acc:.4f} LR={lr:.2e}")

    with open(LOG_PATH, "a", newline="") as f:
        csv.writer(f).writerow([epoch+1, lr, train_loss, train_acc, val_acc])

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))

# =========================
# 8. Test
# =========================
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best_model.pth")))
model.eval()
correct, total = 0, 0

for x, y in tqdm(test_loader, desc="Test"):
    x, y = x.to(device), y.to(device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = model(x).argmax(1)
    correct += (pred == y).sum().item()
    total += y.size(0)

print(f"Test Accuracy: {correct / total:.4f}")