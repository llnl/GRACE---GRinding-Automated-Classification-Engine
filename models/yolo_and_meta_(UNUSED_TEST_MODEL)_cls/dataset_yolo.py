import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GrindingDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 metadata_csv: str,
                 filename_col: str = "filename",
                 metadata_cols: list = None,        # <--- NEW
                 transform=None,
                 aug_factor: int = 1,
                 meta_mean=None,
                 meta_std=None):
        """
        Args:
            images_dir: Directory with images.
            metadata_csv: Path to CSV with metadata and labels.
            filename_col: Column in CSV with image filenames.
            metadata_cols: List of metadata column names to use (e.g., ['grit_time', 'grit_step']).
            transform: Image transformations.
            aug_factor: Number of augmentation repeats per sample.
            meta_mean: Mean values (from training set) for metadata normalization.
            meta_std: Std values (from training set) for metadata normalization.
        """
        self.images_dir = images_dir
        self.df = pd.read_csv(metadata_csv)
        self.filename_col = filename_col
        self.aug_factor = aug_factor
        self.metadata_cols = metadata_cols or ['grit_time', 'grit_step']

        if self.filename_col not in self.df.columns:
            raise KeyError(f"Column '{self.filename_col}' not in CSV")
        for col in self.metadata_cols:
            if col not in self.df.columns:
                raise KeyError(f"Metadata column '{col}' not in CSV")

        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

        # Store normalization stats as torch tensors (or None)
        if meta_mean is not None:
            self.meta_mean = torch.tensor(meta_mean, dtype=torch.float32)
            assert len(self.meta_mean) == len(self.metadata_cols), \
                "meta_mean length must match metadata_cols"
        else:
            self.meta_mean = None

        if meta_std is not None:
            self.meta_std = torch.tensor(meta_std, dtype=torch.float32)
            assert len(self.meta_std) == len(self.metadata_cols), \
                "meta_std length must match metadata_cols"
        else:
            self.meta_std = None

    def __len__(self):
        return len(self.df) * self.aug_factor

    def __getitem__(self, idx):
        orig_idx = idx % len(self.df)
        row = self.df.iloc[orig_idx]
        raw = row[self.filename_col]

        if isinstance(raw, (float, int)):
            filename = f"{int(raw)}.jpg"
        else:
            filename = str(raw)

        class_label = str(int(row['label']))
        img_path = os.path.join(self.images_dir, class_label, filename)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Extract and normalize metadata
        meta = torch.tensor([row[col] for col in self.metadata_cols], dtype=torch.float32)
        if self.meta_mean is not None and self.meta_std is not None:
            meta = (meta - self.meta_mean) / self.meta_std

        lbl = torch.tensor(row['label'], dtype=torch.long)
        return image, meta, lbl
    

import torchvision.transforms as transforms
import random
import numpy as np
from seed import SEED

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
import torchvision.transforms as T

class SingleImageTransform:
    def __init__(self, image_size=(640, 640), train=True, flip_prob=0.5, rotation_degrees=10, jitter_params=None):
        self.train = train
        self.resize = T.Resize(image_size)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if jitter_params is None:
            jitter_params = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.color_jitter = T.ColorJitter(**jitter_params)
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees

        if self.train:
            self.augment = T.Compose([
                T.RandomHorizontalFlip(p=self.flip_prob),
                T.RandomRotation(degrees=self.rotation_degrees, fill=(128,128,128)),
                self.color_jitter
            ])
        else:
            self.augment = T.Compose([])

    def __call__(self, img):
        img = self.resize(img)
        if self.train:
            img = self.augment(img)
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img