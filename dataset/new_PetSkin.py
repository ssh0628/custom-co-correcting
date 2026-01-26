import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError, ImageOps
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
    Optimized with Context-Aware Crop & Resize logic.
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
            cache_name = f"petskin_new_{subset_name}_seed{random_seed}_size{self.resize_dim[0]}x{self.resize_dim[1]}_len{len(self.data)}.npy"
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

    def pad_reflection(self, img, pad_l, pad_t, pad_r, pad_b):
        """
        Applies reflection padding using PIL.
        Strategy: Pad Horizontally first, then Vertically to handle corners naturally.
        """
        w, h = img.size
        
        # 1. Horizontal Padding
        if pad_l > 0 or pad_r > 0:
            new_w = w + pad_l + pad_r
            h_img = Image.new(img.mode, (new_w, h))
            h_img.paste(img, (pad_l, 0))
            
            if pad_l > 0:
                # Mirror left strip
                left_strip = img.crop((0, 0, min(w, pad_l), h))
                left_pad = ImageOps.mirror(left_strip)
                if left_pad.width < pad_l:
                    left_pad = left_pad.resize((pad_l, h), Image.NEAREST) # Stretch fallback if tiny image
                h_img.paste(left_pad, (0, 0))
                
            if pad_r > 0:
                # Mirror right strip
                right_strip = img.crop((w - min(w, pad_r), 0, w, h))
                right_pad = ImageOps.mirror(right_strip)
                if right_pad.width < pad_r:
                    right_pad = right_pad.resize((pad_r, h), Image.NEAREST)
                h_img.paste(right_pad, (pad_l + w, 0))
            
            img = h_img
            w = new_w # Update width for vertical step

        # 2. Vertical Padding
        if pad_t > 0 or pad_b > 0:
            new_h = h + pad_t + pad_b
            v_img = Image.new(img.mode, (w, new_h))
            v_img.paste(img, (0, pad_t))
            
            if pad_t > 0:
                # Flip top strip
                top_strip = img.crop((0, 0, w, min(h, pad_t)))
                top_pad = ImageOps.flip(top_strip)
                if top_pad.height < pad_t:
                    top_pad = top_pad.resize((w, pad_t), Image.NEAREST)
                v_img.paste(top_pad, (0, 0))
                
            if pad_b > 0:
                # Flip bottom strip
                bottom_strip = img.crop((0, h - min(h, pad_b), w, h))
                bottom_pad = ImageOps.flip(bottom_strip)
                if bottom_pad.height < pad_b:
                    bottom_pad = bottom_pad.resize((w, pad_b), Image.NEAREST)
                v_img.paste(bottom_pad, (0, pad_t + h))
                
            img = v_img

        return img

    def context_aware_crop_resize(self, img, roi_box, target_size=(224, 224)):
        """
        Applies Context-Aware Crop & Resize logic with Mirror Padding.
        
        Args:
            img (PIL.Image): Original full-size image.
            roi_box (tuple): (x, y, x+w, y+h) of the ROI.
            target_size (tuple): Desired output size (width, height).
            
        Returns:
            PIL.Image: Processed image of size target_size.
        """
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_box
        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        target_w, target_h = target_size
        img_w, img_h = img.size

        # Case 1: ROI is smaller than target size (Context Expansion)
        # Strict logic: NEVER upscale. Expand context relative to ROI center.
        if roi_w < target_w or roi_h < target_h:
            # Calculate ROI center
            center_x = roi_x1 + roi_w / 2
            center_y = roi_y1 + roi_h / 2
            
            # Ideal Crop Box (Centered on ROI, Size 224x224)
            # Coordinates can be negative or larger than image
            crop_x1 = int(round(center_x - target_w / 2))
            crop_y1 = int(round(center_y - target_h / 2))
            crop_x2 = crop_x1 + target_w
            crop_y2 = crop_y1 + target_h
            
            # Calculate Intersection with Actual Image
            # We only crop what exists. The rest is padding.
            valid_x1 = max(0, crop_x1)
            valid_y1 = max(0, crop_y1)
            valid_x2 = min(img_w, crop_x2)
            valid_y2 = min(img_h, crop_y2)
            
            # Calculate Padding needed for each side
            # If crop_x1 is negative (e.g. -10), we need 10px padding on left.
            pad_left = max(0, valid_x1 - crop_x1)
            pad_top = max(0, valid_y1 - crop_y1)
            pad_right = max(0, crop_x2 - valid_x2)
            pad_bottom = max(0, crop_y2 - valid_y2)
            
            # Perform the valid crop
            # If valid area is invalid (e.g. image completely outside? impossible if ROI is inside), handle gracefully
            if valid_x2 <= valid_x1 or valid_y2 <= valid_y1:
                 # Should not happen if ROI is inside image
                 return img.resize(target_size, Image.BICUBIC)

            crop_img = img.crop((valid_x1, valid_y1, valid_x2, valid_y2))
            
            # Apply Mirror Padding if needed
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                crop_img = self.pad_reflection(crop_img, pad_left, pad_top, pad_right, pad_bottom)
            
            # Final Safety Check: ensure exact size match (e.g. rounding errors)
            if crop_img.size != target_size:
                crop_img = crop_img.resize(target_size, Image.BICUBIC)
                
            return crop_img

        # Case 2: ROI is larger than or equal to target size (Downscaling)
        else:
            # Crop the ROI
            crop_img = img.crop(roi_box)
            # Resize using BICUBIC for downscaling
            return crop_img.resize(target_size, Image.BICUBIC)

    def img_loader(self, img_path):
        """
        Robust image loader with Context-Aware logic.
        """
        try:
            # Check for JSON file with bounding box info first to know how to crop if needed
            json_path = os.path.splitext(img_path)[0] + '.json'
            
            # Open Image with error handling
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                
                roi_box = None
                
                # Bounding Box Logic
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
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
                                                
                                                if w > 0 and h > 0:
                                                    roi_box = (x, y, x + w, y + h)
                                                    break
                                            except (ValueError, TypeError):
                                                continue
                    except Exception:
                        pass
                
                if roi_box:
                    # Apply Context-Aware Crop & Resize
                    final_img = self.context_aware_crop_resize(img, roi_box, self.resize_dim)
                else:
                    # Fallback: simple resize if no ROI found
                    # BUT enforce padding if image is smaller than target to avoid upscaling
                    if img.width < self.resize_dim[0] or img.height < self.resize_dim[1]:
                        # Center and pad reflection
                        # Treat the whole image as ROI
                        final_img = self.context_aware_crop_resize(img, (0, 0, img.width, img.height), self.resize_dim)
                    else:
                        final_img = img.resize(self.resize_dim, Image.BICUBIC)

                return np.asarray(final_img).astype(np.uint8)
                
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
