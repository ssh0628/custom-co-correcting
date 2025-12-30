import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
import os
import os.path
from os.path import join
import glob
import json
from sklearn.model_selection import train_test_split
from multiprocessing.pool import ThreadPool
import time

from dataset.utils import noisify

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class PetSkin(torch.utils.data.Dataset):
    '''
    Pet Skin Lesion Dataset (A1-A6)
    Optimized with Multi-threading, NPY Caching, and Robust Error Handling.
    '''

    def __init__(self,
                 root,
                 train=0, # 0: train, 1: test, 2: val
                 transform=None,
                 target_transform=None,
                 noise_type='clean',
                 noise_rate=0.00,
                 device=1,
                 split_ratios=(0.7, 0.15, 0.15),
                 random_seed=0
                 ):
        
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        self.device = device
        self.noise_type = noise_type
        self.labelOrder = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
        self.class_map = {label: i for i, label in enumerate(self.labelOrder)}

        # Gather all data
        self.all_data = []
        self.all_labels = []
        
        # Check if root has Datasets folder or is the Datasets folder
        search_root = root
        if os.path.isdir(join(root, 'Datasets')):
            search_root = join(root, 'Datasets')

        print(f"Loading PetSkin data from {search_root}")

        for label_name in self.labelOrder:
            class_dir = join(search_root, label_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory {class_dir} not found.")
                continue
            
            # Recursive search for jpg files
            for root_dir, dirs, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith('.jpg') or file.lower().endswith('.jpeg'):
                        self.all_data.append(join(root_dir, file))
                        self.all_labels.append(self.class_map[label_name])

        self.all_data = np.array(self.all_data)
        self.all_labels = np.array(self.all_labels)

        # Splitting with Stratification
        num_samples = len(self.all_data)
        indices = np.arange(num_samples)
        
        # 1. Split Train vs (Val + Test)
        train_indices, temp_indices, _, temp_labels = train_test_split(
            indices, self.all_labels,
            train_size=split_ratios[0],
            stratify=self.all_labels,
            random_state=random_seed
        )

        # 2. Split Val vs Test from the remaining data
        relative_val_size = split_ratios[1] / (split_ratios[1] + split_ratios[2])
        
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=relative_val_size,
            stratify=temp_labels,
            random_state=random_seed
        )

        if self.train == 0:
            self.data = self.all_data[train_indices]
            self.labels = self.all_labels[train_indices]
            subset_name = "train"
        elif self.train == 1:
            self.data = self.all_data[test_indices]
            self.labels = self.all_labels[test_indices]
            subset_name = "test"
        elif self.train == 2:
            self.data = self.all_data[val_indices]
            self.labels = self.all_labels[val_indices]
            subset_name = "val"
        
        if image_size := get_image_size_from_transform(transform):
             self.resize_dim = (image_size, image_size)
        else:
             self.resize_dim = (224, 224) 

        self.loaded_data = []
        
        # --- Optimized Loading Logic ---
        if self.device == 1:
            # Prepare Cache Directory
            cache_dir = join(self.root, 'cache_npy')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Cache Filename: includes subset, seed, size, and dataset length to detect changes
            cache_name = f"petskin_{subset_name}_seed{random_seed}_size{self.resize_dim[0]}x{self.resize_dim[1]}_len{len(self.data)}.npy"
            cache_path = join(cache_dir, cache_name)

            if os.path.exists(cache_path):
                print(f"[{subset_name}] Found cached data at {cache_path}. Loading...")
                st = time.time()
                self.loaded_data = np.load(cache_path)
                print(f"[{subset_name}] Loaded {len(self.loaded_data)} images from cache in {time.time()-st:.2f}s")
            else:
                # Reduced threads to 16 for stability
                print(f"[{subset_name}] Cache not found. Loading {len(self.data)} images into RAM using 16 threads...")
                st = time.time()
                
                # Multi-threading loading
                pool = ThreadPool(16)
                # Use map to preserve order matching self.data
                results = pool.map(self.img_loader, self.data)
                pool.close()
                pool.join()
                
                self.loaded_data = np.array(results)
                print(f"[{subset_name}] Loaded and formatted data in {time.time()-st:.2f}s")
                
                print(f"[{subset_name}] Saving cache to {cache_path}...")
                np.save(cache_path, self.loaded_data)
                print(f"[{subset_name}] Cache saved.")
        
        # Noisy labels generation
        self.labels = np.asarray(self.labels)
        
        if noise_type == 'clean':
            self.noise_or_not = np.ones([len(self.labels)], dtype=bool)
            self.noisy_labels = self.labels
        else:
            self.noisy_labels, self.actual_noise_rate = noisify(dataset="petskin",
                                                                  nb_classes=6,
                                                                  train_labels=np.expand_dims(self.labels, 1),
                                                                  noise_type=noise_type,
                                                                  noise_rate=noise_rate,
                                                                  random_state=random_seed)
            self.noisy_labels = self.noisy_labels.squeeze()
            self.noise_or_not = self.noisy_labels == self.labels

    def img_loader(self, img_path):
        """
        Robust image loader with error handling and fallback to black image.
        """
        try:
            # Check for JSON file with bounding box info first to know how to crop if needed
            json_path = os.path.splitext(img_path)[0] + '.json'
            
            # Open Image with error handling
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                
                # Bounding Box Logic
                if os.path.exists(json_path):
                    try:
                        # USE utf-8 encoding for Windows compatibility
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Navigate: labelingInfo -> box -> location -> x, y, width, height
                            if 'labelingInfo' in data:
                                for info_item in data['labelingInfo']:
                                    if 'box' in info_item:
                                        box_info = info_item
                                        if 'location' in box_info['box'] and len(box_info['box']['location']) > 0:
                                            loc = box_info['box']['location'][0]
                                            try:
                                                x = int(loc.get('x'))
                                                y = int(loc.get('y'))
                                                w = int(loc.get('width'))
                                                h = int(loc.get('height'))
                                                
                                                # Validate dimensions
                                                if w > 0 and h > 0:
                                                    crop_box = (x, y, x + w, y + h)
                                                    # Ensure crop box is within image bounds (optional safety, PIL handles it gracefully usually)
                                                    img = img.crop(crop_box)
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                    except Exception:
                        # If JSON parsing fails, just use the original image
                        pass

                return np.asarray(img.resize(self.resize_dim, Image.NEAREST)).astype(np.uint8)
                
        except (OSError, UnidentifiedImageError, Exception) as e:
            # Critical Error Handler: Return black image
            print(f"Error loading image {img_path}: {e}. Returning black image.")
            return np.zeros((self.resize_dim[1], self.resize_dim[0], 3), dtype=np.uint8)

    def __getitem__(self, index):
        if self.device == 1 and len(self.loaded_data) > 0:
            img = self.loaded_data[index]
        else:
            img = self.img_loader(self.data[index])
            
        target = self.noisy_labels[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

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
