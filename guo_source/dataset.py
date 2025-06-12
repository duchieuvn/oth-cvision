import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T

class VOCDataset(Dataset):
    def __init__(self, img_dir, mask_dir, image_list, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(os.path.join(self.img_dir, img_name + ".jpg")).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_name + ".png")).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = T.ToTensor()(mask).long().squeeze(0)
        
        return image, mask
