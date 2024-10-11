import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose, Resize, HorizontalFlip, VerticalFlip
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, data, transform=None, h=270, w=400, orig_h=540, orig_w=800):
        self.data = data
        self.transform = transform
        self.h = h
        self.w = w
        self.orig_h = orig_h
        self.orig_w = orig_w

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['Image Name']
        image_path = "/kaggle/input/biometry-detect/Task - Landmark/images/" + image_name
        img = Image.open(image_path)
        img = img.resize((self.w, self.h))
        img = np.array(img) / 255.0  # Normalize

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = torch.tensor(img).unsqueeze(0).float()
        biometry_points = self.data.iloc[idx, 1:].values.astype(float)
        biometry_points[0::2] *= (self.w / self.orig_w)
        biometry_points[1::2] *= (self.h / self.orig_h)
        return img, biometry_points

def get_dataloaders(labels_df, batch_size=8, h=270, w=400):
    transform_train = Compose([Resize(h, w), HorizontalFlip(p=0.5), VerticalFlip(p=0.5)])
    transform_val = Resize(h, w)

    train_data, test_data = train_test_split(labels_df, test_size=0.15, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(train_data, transform=transform_train, h=h, w=w)
    val_dataset = CustomDataset(val_data, transform=transform_val, h=h, w=w)
    test_dataset = CustomDataset(test_data, transform=transform_val, h=h, w=w)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
