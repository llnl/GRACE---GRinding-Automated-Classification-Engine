"""
Author: Sean Jackson
Date Last Modified: 08.08.25
Description:
    Dataset and Transform classes for the Multi-Task Model
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
    def __init__(
        self,
        images_dir: str,
        metadata_csv: str,
        grit_step2idx: dict, # grit step value (float) to index
        grit_time2idx: dict, # grit time value (float) to index
        filename_col: str = "image_id",
        transform=None,
        aug_factor: int = 1
    ):
        self.images_dir = images_dir
        self.df = pd.read_csv(metadata_csv)
        self.filename_col = filename_col
        self.aug_factor = aug_factor

        # Save mappings for categorical metadata (grit step and grind time)
        self.grit_step2idx = grit_step2idx
        self.grit_time2idx = grit_time2idx

        # Check that column exist, and do some error checking
        if self.filename_col not in self.df.columns:
            raise KeyError(f"Column '{self.filename_col}' not in CSV")

        # Instantiate the transform (class below)
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize([IMG_MEAN, IMG_MEAN, IMG_MEAN],
                                 [IMG_STD, IMG_STD, IMG_STD])
        ])

    def __len__(self):
        # number of data samples * number of augmentations per sample
        return len(self.df) * self.aug_factor

    def __getitem__(self, idx):
        orig_idx = idx % len(self.df)
        row = self.df.iloc[orig_idx]
        raw = row[self.filename_col]

        # Handle numeric or string image_id
        if isinstance(raw, (float, int)):
            filename = f"{int(raw)}.jpg"
        else:
            filename = str(raw) + ".jpg"

        class_label = int(row['label'])
        img_path = os.path.join(self.images_dir, str(class_label), filename)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Categorical targets: data to index
        grit_step_idx = self.grit_step2idx[row['grit_step']]
        grit_time_idx = self.grit_time2idx[row['grit_time']]

        # Return all targets as integers
        return image, (class_label, grit_step_idx, grit_time_idx)
    
    
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
        self.train = train # Applies augmentations if true
        self.resize = T.Resize(image_size)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[IMG_MEAN, IMG_MEAN, IMG_MEAN],
            std=[IMG_STD, IMG_STD, IMG_STD]
        )
        if jitter_params is None:
            # saturation is useless for greyscale, hue is undefined
            jitter_params = dict(brightness=0.1, contrast=0.1, saturation=0, hue=0)
        self.color_jitter = T.ColorJitter(**jitter_params)
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
            self.augment = T.Compose([])

    def __call__(self, img):
        img = self.resize(img)
        if self.train:
            img = self.augment(img)
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img