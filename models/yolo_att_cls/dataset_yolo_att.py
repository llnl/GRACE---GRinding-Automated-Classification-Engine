"""
Author: Sean Jackson
Date Last Modified: 08.08.25
Description:
    Dataset and Transform classes for the Attention-Gating Model
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


class GrindingDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 metadata_csv: str,
                 filename_col: str = "filename",
                 metadata_cols: list = None,
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

        Note: The metadata mean is calculated from the training split.
        """

        # Initialize Constructor Variables
        self.images_dir = images_dir
        self.df = pd.read_csv(metadata_csv)
        self.filename_col = filename_col
        self.aug_factor = aug_factor
        self.metadata_cols = metadata_cols or ['grit_time', 'grit_step']

        # Make sure the file columns exist
        if self.filename_col not in self.df.columns:
            raise KeyError(f"Column '{self.filename_col}' not in CSV")
        for col in self.metadata_cols:
            if col not in self.df.columns:
                raise KeyError(f"Metadata column '{col}' not in CSV")


        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize([IMG_MEAN, IMG_MEAN, IMG_MEAN],
                                 [IMG_STD, IMG_STD, IMG_STD])
        ])

        # Store normalization stats as torch tensors
        # Do some error checking to ensure correctness
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

    # Return the length (the number of training samples * number of augmentations)
    def __len__(self):
        return len(self.df) * self.aug_factor

    # Function to return a single sample:
    def __getitem__(self, idx):
        # Index for the non-augmented dataset
        orig_idx = idx % len(self.df)
        # Obtain an image at this index:
        row = self.df.iloc[orig_idx]
        raw = row[self.filename_col]

        # Get the training image
        if isinstance(raw, (float, int)):
            filename = f"{int(raw)}.jpg"
        else:
            filename = str(raw)

        # Get the class label (good/bad)
        class_label = str(int(row['label']))
        # Obtain the complete image path (from the directory structure)
        img_path = os.path.join(self.images_dir, class_label, filename)
        # Convert the image to RGB (just duplicates greyscale channel 3 times)
        image = Image.open(img_path).convert("RGB")
        # Apply the transform (resize, to tensor, augment)
        image = self.transform(image)

        # Extract and normalize metadata
        meta = torch.tensor([row[col] for col in self.metadata_cols], dtype=torch.float32)
        if self.meta_mean is not None and self.meta_std is not None:
            meta = (meta - self.meta_mean) / self.meta_std

        lbl = torch.tensor(row['label'], dtype=torch.long)
        return image, meta, lbl
    

import torchvision.transforms as transforms
import torchvision.transforms as T
import random
import numpy as np
from seed import SEED

# Ensure all random seeds are set to the most recent
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

class SingleImageTransform:
    def __init__(self, image_size=(640, 640), train=True, flip_prob=0.5, rotation_degrees=10, jitter_params=None):
        self.train = train # Applies augmentations if true
        self.resize = T.Resize(image_size)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[IMG_MEAN, IMG_MEAN, IMG_MEAN],
            std=[IMG_STD, IMG_STD, IMG_STD]
        )
        if jitter_params is None:
            # saturation is useless for greyscale, hue is undefined
            jitter_params = dict(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0)
        self.color_jitter = T.ColorJitter(**jitter_params) # fancy dictionary unpacker
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees

        # Apply the augmentations and fill corners with a neutral grey
        bg_channel = IMG_MEAN*255
        if self.train:
            self.augment = T.Compose([
                T.RandomHorizontalFlip(p=self.flip_prob),
                T.RandomRotation(degrees=self.rotation_degrees, fill=(bg_channel,bg_channel,bg_channel)),
                self.color_jitter
            ])
        else:
            self.augment = T.Compose([]) # DO NOT augment is train is false

    # Apply all steps sequentially when called.
    def __call__(self, img):
        img = self.resize(img)
        if self.train:
            img = self.augment(img)
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img