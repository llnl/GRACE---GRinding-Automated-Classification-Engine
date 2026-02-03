"""
Author: Sean Jackson
Date Last Modified: 08.08.25
Description:
    Dataset and Transform classes for the Delta Fusion Model
"""

# Image mean and std calculated from the training set
# This can be found in the EDA notebook
IMG_MEAN = 0.7329108720275311
IMG_STD = 0.19402230922580405


import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class PairwiseGrindingDataset(Dataset):
    def __init__(self,
                 images_dir: str,                               # The directory containing the images
                 pairs_csv: str,                                # The directory containing the metadata
                 filename1_col: str = "img_id1",                # The column for the 1st image (consecutively)
                 filename2_col: str = "img_id2",                # The column for the 2nd image (consecutively)
                 meta1_cols = ["grit_time1", "grit_step1"],     # The columns to use from the metadata csv (1st image)
                 meta2_cols = ["grit_time2", "grit_step2"],     # The columns to use from the metadata csv (2nd image)
                 label_col: str = "label2",                     # The main label to compare for good and bad
                 transform=None,                                # The transform to augment the images
                 preload_images: bool = True,                   # Whether to preload the images into memory or not
                 oversample_factor: int = 1,                    # Number of augmentations per image
                 class1_col: str = "label1",                    # The label for the 1st image (needed for image loading)
                 class2_col: str = "label2",                    # The label for the 2nd image
                 meta1_mean=None,                               # The mean for the 1st metadata
                 meta1_std=None,                                # The std  for the 1st metadata
                 meta2_mean=None,                               # The mean for the 2nd metadata
                 meta2_std=None                                 # The std  for the 1st metadata
                 ):
        self.images_dir     = images_dir                
        self.df             = pd.read_csv(pairs_csv)    # The dataframe containing the metadata
        self.filename1_col  = filename1_col             # The column for the 1st image (consecutively)
        self.filename2_col  = filename2_col             # The column for the 2nd image (consecutively)
        self.meta1_cols     = meta1_cols                # The 1st metadata column (consecutively)
        self.meta2_cols     = meta2_cols                # The 2nd metadata column (consecutively)
        self.label_col      = label_col                 # The label (good/bad)
        self.transform      = transform                 # The transform to pass in
        self.preload_images = preload_images            # Preload images? You should
        self.oversample_factor = oversample_factor      # Number of augmentations per image
        self.class1_col     = class1_col                # Column for the good/bad label for the 1st image
        self.class2_col     = class2_col                # Column for the good/bad label for the 2nd image

        # Store normalization stats as tensors, if provided
        self.meta1_mean = torch.tensor(meta1_mean, dtype=torch.float32) if meta1_mean is not None else None
        self.meta1_std  = torch.tensor(meta1_std,  dtype=torch.float32) if meta1_std  is not None else None
        self.meta2_mean = torch.tensor(meta2_mean, dtype=torch.float32) if meta2_mean is not None else None
        self.meta2_std  = torch.tensor(meta2_std,  dtype=torch.float32) if meta2_std  is not None else None

        # Basic validation
        for col in [filename1_col, filename2_col, label_col, class1_col, class2_col] + meta1_cols + meta2_cols:
            if col not in self.df.columns:
                raise KeyError(f"Column '{col}' not found in CSV.")

        # If no transform is provided, use a simple default (just resize)
        if transform is None:
            self.transform = lambda img1, img2: (
                transforms.ToTensor()(img1.resize((640, 640))),
                transforms.ToTensor()(img2.resize((640, 640))),
            )
        else:
            self.transform = transform

        # Preload images if requested
        # (create a cache (dictionary) that holds all of the preloaded data)
        if self.preload_images:
            self.image_cache = {}
            for _, row in self.df.iterrows():
                img_id1 = self._normalize_id(row[self.filename1_col])
                class1  = self._normalize_id(row[self.class1_col])
                img_id2 = self._normalize_id(row[self.filename2_col])
                class2  = self._normalize_id(row[self.class2_col])
                for img_id, class_label in [(img_id1, class1), (img_id2, class2)]:
                    key = f"{class_label}/{img_id}"
                    if key not in self.image_cache:
                        img_path = os.path.join(self.images_dir, class_label, f"{img_id}.jpg")
                        try:
                            img = Image.open(img_path).convert("RGB")
                            self.image_cache[key] = img
                        except Exception as e:
                            raise RuntimeError(f"Failed to load image: {img_path}\n{e}")
        else:
            self.image_cache = None

    def __len__(self):
        # number of training samples * number of augmentations
        return len(self.df) * self.oversample_factor

    # Ensures image IDs and labels are always strings
    # This is needed for file loading
    def _normalize_id(self, raw):
        if isinstance(raw, (float, int)):
            return str(int(raw))
        else:
            return str(raw)

    # Return an image (from cache or from file)
    def _get_image(self, img_id, class_label):
        img_id = self._normalize_id(img_id)
        class_label = self._normalize_id(class_label)
        key = f"{class_label}/{img_id}"
        if self.image_cache is not None:
            return self.image_cache[key].copy()
        else:
            img_path = os.path.join(self.images_dir, class_label, f"{img_id}.jpg")
            return Image.open(img_path).convert("RGB")

    # Finally, get all of the data for train / val item (from cache or from file)
    def __getitem__(self, idx):
        base_idx = idx % len(self.df)
        row = self.df.iloc[base_idx]

        img1 = self._get_image(row[self.filename1_col], row[self.class1_col])
        img2 = self._get_image(row[self.filename2_col], row[self.class2_col])

        if self.transform:
            img1, img2 = self.transform(img1, img2)

        meta1 = torch.tensor([row[col] for col in self.meta1_cols], dtype=torch.float32)
        meta2 = torch.tensor([row[col] for col in self.meta2_cols], dtype=torch.float32)

        # Normalize meta1 and meta2 independently, if stats provided
        if self.meta1_mean is not None and self.meta1_std is not None:
            meta1 = (meta1 - self.meta1_mean) / self.meta1_std
        if self.meta2_mean is not None and self.meta2_std is not None:
            meta2 = (meta2 - self.meta2_mean) / self.meta2_std

        label = torch.tensor(row[self.label_col], dtype=torch.long)

        return img1, meta1, img2, meta2, label


import torchvision.transforms as transforms
import random
from seed import SEED

import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
import torchvision.transforms as T

import torchvision.transforms.functional as F

def _get_range(val, center=1.0):
    # If val is a float, interpret as (center - val, center + val)
    if isinstance(val, (float, int)):
        return center - val, center + val
    elif isinstance(val, (tuple, list)) and len(val) == 2:
        return val
    else:
        raise ValueError(f"Invalid jitter param: {val}")

def get_color_jitter_params(brightness, contrast, saturation, hue, generator):
    fn_idx = torch.randperm(4, generator=generator)
    b_min, b_max = _get_range(brightness)
    c_min, c_max = _get_range(contrast)
    s_min, s_max = _get_range(saturation)
    h_min, h_max = _get_range(hue, center=0.0)
    b = torch.empty(1).uniform_(b_min, b_max, generator=generator).item() if b_max - b_min > 0 else 1.0
    c = torch.empty(1).uniform_(c_min, c_max, generator=generator).item() if c_max - c_min > 0 else 1.0
    s = torch.empty(1).uniform_(s_min, s_max, generator=generator).item() if s_max - s_min > 0 else 1.0
    h = torch.empty(1).uniform_(h_min, h_max, generator=generator).item() if h_max - h_min > 0 else 0.0
    return fn_idx, b, c, s, h

def apply_color_jitter(img, fn_idx, b, c, s, h):
    for fn_id in fn_idx:
        if fn_id == 0 and b != 0:
            img = F.adjust_brightness(img, b)
        elif fn_id == 1 and c != 0:
            img = F.adjust_contrast(img, c)
        elif fn_id == 2 and s != 0:
            img = F.adjust_saturation(img, s)
        elif fn_id == 3 and h != 0:
            img = F.adjust_hue(img, h)
    return img

class PairwiseTransform:
    def __init__(
        self,
        image_size=(640, 640),
        train=True,
        flip_prob=0.5,
        rotation_degrees=10,
        jitter_params=None,
        generator=None,  # Optional torch.Generator for reproducibility
    ):
        self.train = train
        self.resize = transforms.Resize(image_size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[IMG_MEAN, IMG_MEAN, IMG_MEAN],
            std=[IMG_STD, IMG_STD, IMG_STD]
        )
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees
        if jitter_params is None:
            jitter_params = dict(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0)
        self.jitter_params = jitter_params  # Store for later use
        filtered_jitter_params = {k: v for k, v in jitter_params.items() if v != 0.0}
        self.color_jitter = transforms.ColorJitter(**filtered_jitter_params)
        self.generator = generator or torch.Generator()

    def __call__(self, img1, img2):
        img1 = self.resize(img1)
        img2 = self.resize(img2)

        if self.train:
            # Use torch.rand for shared randomness
            if torch.rand(1, generator=self.generator).item() < self.flip_prob:
                img1 = F.hflip(img1)
                img2 = F.hflip(img2)

            angle = float(torch.empty(1).uniform_(
                -self.rotation_degrees, self.rotation_degrees
            ).item())
            bg = IMG_MEAN * 255
            img1 = F.rotate(img1, angle, fill=(bg, bg, bg))
            img2 = F.rotate(img2, angle, fill=(bg, bg, bg))

            # Color jitter is a bit trickier, as the built-in transform is stateful.
            # We can use its get_params method to generate shared parameters:
            fn_idx, b, c, s, h = get_color_jitter_params(
                self.jitter_params.get('brightness', 0.0),
                self.jitter_params.get('contrast', 0.0),
                self.jitter_params.get('saturation', 0.0),
                self.jitter_params.get('hue', 0.0),
                self.generator
            )
            img1 = apply_color_jitter(img1, fn_idx, b, c, s, h)
            img2 = apply_color_jitter(img2, fn_idx, b, c, s, h)

        img1 = self.normalize(self.to_tensor(img1))
        img2 = self.normalize(self.to_tensor(img2))

        return img1, img2