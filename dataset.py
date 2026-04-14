import os
import glob
import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

#Normalization statistics
S1_MEAN = [-12.6017, -20.2717]
S1_STD = [5.1401, 5.7527]

class Sen3ClassesDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # 1. Read Image
            img = tifffile.imread(path).astype(np.float32)
            
            # Sanitize: Replace NaN/Inf with 0.0 (common in SAR processing)
            if np.isnan(img).any() or np.isinf(img).any():
                img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        except Exception as e:
            print(f"Error loading {path}: {e}")
            img = np.zeros((64, 64, 2), dtype=np.float32)

        # 2. Transpose to (C, H, W)
        if img.ndim == 3 and img.shape[2] == 2:
            img = np.moveaxis(img, 2, 0)
        
        img_tensor = torch.from_numpy(img)

        # 3. Apply Transforms
        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label

def prepare_sen3classes_splits(data_root, train_size=0.8, random_state=42, binary=False):
    """
    
    Args:
        binary (bool): If True, merges BusStand and RailwayStation into label 1.
                       If False, provides 3 separate classes (0, 1, 2).
    """
    image_paths = []
    labels = []
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Root directory not found at: {data_root}")

    
    if binary:
        
        target_class_mapping = {
            'Airports': 0,
            'BusStand': 1,
            'RailwayStation': 1
        }
        class_names = ['airports', 'non_airport']
    else:
        
        target_class_mapping = {
            'Airports': 0,
            'BusStand': 1,
            'RailwayStation': 2
        }
        class_names = ['airports', 'busstand', 'railwaystation']

    print(f"--- Preparing {'Binary' if binary else 'Multiclass'} Dataset ---")

    # Collect paths
    for folder_name, target_label in target_class_mapping.items():
        cls_dir = os.path.join(data_root, folder_name)
        
        
        if not os.path.exists(cls_dir):
            cls_dir = os.path.join(data_root, folder_name.lower())
        
        if not os.path.exists(cls_dir):
            print(f"Warning: Folder '{folder_name}' not found. Skipping.")
            continue

        files = glob.glob(os.path.join(cls_dir, "*.tif"))
        for f in files:
            image_paths.append(f)
            labels.append(target_label)

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    if len(image_paths) == 0:
        raise ValueError(f"No .tif files found in {data_root}. Check your folder names.")

    # Stratified Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, 
        train_size=train_size, 
        stratify=labels, 
        random_state=random_state
    )

    print(f"Total images: {len(image_paths)} | Train: {len(train_paths)} | Val: {len(val_paths)}")
    
    return train_paths, val_paths, train_labels, val_labels, class_names

def get_sen3classes_transforms(input_size=224):
    """
    Standard SAR transforms with normalization and augmentation.
    """
    normalize = transforms.Normalize(mean=S1_MEAN, std=S1_STD)

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        normalize
    ])

    return train_transform, val_transform