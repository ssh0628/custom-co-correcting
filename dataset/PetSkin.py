import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFile
import os
from os.path import join
from pathlib import Path

from dataset.utils import noisify

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PetSkin(torch.utils.data.Dataset):
    '''
    Pet Skin Lesion Dataset
    Refactored to load from pre-generated .npy files (paths & labels)
    produced by to_npy.py.
    '''

    def __init__(self,
                 root,
                 train=0, # 0: train, 1: test, 2: val
                 transform=None,
                 target_transform=None,
                 noise_type='clean',
                 noise_rate=0.00,
                 device=1,
                 random_seed=0,  # Added random_seed
                 **kwargs 
                 ):
        
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        self.device = device
        self.noise_type = noise_type
        
        # Map train index to split name
        if self.train == 0:
            self.split = "train"
        elif self.train == 1:
            self.split = "test"
        elif self.train == 2:
            self.split = "val"
        else:
            raise ValueError(f"Invalid train argument: {train}. Must be 0, 1, or 2.")

        # Load Paths and Labels from NPY files
        # Expecting structure: root/{split}_paths.npy
        paths_file = self.root / f"{self.split}_paths.npy"
        labels_file = self.root / f"{self.split}_labels.npy"
        
        if not paths_file.exists() or not labels_file.exists():
             # Fallback check: maybe inside "cache_npy" subdir if root is dataset root
             fallback_root = self.root / "cache_npy"
             paths_file_fb = fallback_root / f"{self.split}_paths.npy"
             labels_file_fb = fallback_root / f"{self.split}_labels.npy"
             
             if paths_file_fb.exists() and labels_file_fb.exists():
                 self.root = fallback_root
                 paths_file = paths_file_fb
                 labels_file = labels_file_fb
             else:
                 raise RuntimeError(f"[ERR] Missing NPY files for split '{self.split}'.\n"
                                    f"Checked: {paths_file} AND {paths_file_fb}\n"
                                    f"Please run 'to_npy.py' first.")

        print(f"[{self.split.upper()}] Loading from {str(paths_file)}...")
        self.data = np.load(paths_file, allow_pickle=True)
        self.labels = np.load(labels_file, allow_pickle=True).astype(np.int64)
        print(f"[{self.split.upper()}] Loaded {len(self.data)} samples.")

        # Noisy labels generation
        self.labels = np.asarray(self.labels)
        
        if len(self.labels) > 0:
            NB_CLASSES = int(self.labels.max()) + 1
        else:
            NB_CLASSES = 8
        
        if noise_type == 'clean':
            self.noise_or_not = np.ones([len(self.labels)], dtype=bool)
            self.noisy_labels = self.labels
        else:
            # Generate noisy labels using specific random_seed
            self.noisy_labels, self.actual_noise_rate = noisify(dataset="petskin",
                                                                  nb_classes=NB_CLASSES,
                                                                  train_labels=np.expand_dims(self.labels, 1),
                                                                  noise_type=noise_type,
                                                                  noise_rate=noise_rate,
                                                                  random_state=random_seed)
            self.noisy_labels = self.noisy_labels.squeeze()
            self.noise_or_not = self.noisy_labels == self.labels

    def __getitem__(self, index):
        img_path = str(self.data[index])
        target = self.noisy_labels[index]
        
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                
                if self.transform is not None:
                    img = self.transform(img)
                
                if self.target_transform is not None:
                    target = self.target_transform(target)
                    
                return img, target, index
                
        except Exception as e:
            print(f"[ERR] Failed to load {img_path}: {e}")
            # Robust fallback: minimize crashing
            img = Image.new('RGB', (224, 224))
            if self.transform is not None:
                img = self.transform(img)
            return img, target, index

    def __len__(self):
        return len(self.data)

def get_image_size_from_transform(transform):
    if not transform: return None
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            if isinstance(t, transforms.Resize):
                return t.size[0] if isinstance(t.size, tuple) else t.size
    return None
