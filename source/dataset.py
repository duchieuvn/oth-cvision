from torchvision.datasets import VOCSegmentation
from medsegbench import DynamicNuclearMSBench
from torchvision import transforms
import torch
from pathlib import Path
from PIL import Image
import numpy as np

def binary_class_data_transform(img, mask, model_input_size=(224, 224)):
    img = img.resize(model_input_size)
    mask = mask.resize(model_input_size, resample=Image.NEAREST)
    img = transforms.ToTensor()(img)
    mask = torch.as_tensor(np.array(mask), dtype=torch.long)
    return img, mask

class VOCSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set='train', year='2012'):
        self.dataset = VOCSegmentation(
            root=root,
            image_set=image_set,
            year=year,
            download=False
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        return self.data_transform(img, mask)
    
class BUSIDataset(torch.utils.data.Dataset):
    def __init__(self, root, subset='train_folder'):
        self.images = []
        self.masks = []
        self.transform = binary_class_data_transform

        img_path = Path(root) / subset / 'img'
        for filename in sorted(img_path.glob("*.png")):
            self.images.append(filename)

        mask_path = Path(root) / subset / 'label'
        for filename in sorted(mask_path.glob("*.png")):
            self.masks.append(filename)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
       
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask
        img, mask = self.transform(img, mask)
        
        # Convert to binary mask: 0 for background, 1 for lesion
        mask = (mask > 0).long()
        return img, mask, img_path.stem  # Return image name for reference 

class DynamicNucDataset(torch.utils.data.Dataset):
    def __init__(self, root, subset='train'):
        self.images = []
        self.masks = []
        self.transform = binary_class_data_transform

        img_path = Path(root) / subset / 'images'
        print(img_path)
        for filename in sorted(img_path.glob("*.png")):
            self.images.append(filename)

        mask_path = Path(root) / subset / 'masks'
        print(mask_path)
        for filename in sorted(mask_path.glob("*.png")):
            self.masks.append(filename)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
       
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask
        img, mask = self.transform(img, mask)
        
        # Convert to binary mask: 0 for background, 1 for lesion
        mask = (mask > 0).long()
        return img, mask, img_path.stem

class UsForKidneyDataset(torch.utils.data.Dataset):
    def __init__(self, root, subset='train'):
        self.images = []
        self.masks = []
        self.transform = binary_class_data_transform

        img_path = Path(root) / subset / 'images'
        mask_path = Path(root) / subset / 'masks'
        print(f"Loading UsForKidney: {img_path}, {mask_path}")

        self.images = sorted(img_path.glob("*.png"))
        self.masks = sorted(mask_path.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        img, mask = self.transform(img, mask)
        mask = (mask > 0).long()
        return img, mask, self.images[idx].stem


class Covid19RadioDataset(torch.utils.data.Dataset):
    def __init__(self, root, subset='train'):
        self.images = []
        self.masks = []
        self.transform = binary_class_data_transform

        img_path = Path(root) / subset / 'images'
        mask_path = Path(root) / subset / 'masks'
        print(f"Loading Covid19Radio: {img_path}, {mask_path}")

        self.images = sorted(img_path.glob("*.png"))
        self.masks = sorted(mask_path.glob("*.png"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        img, mask = self.transform(img, mask)
        mask = (mask > 0).long()
        return img, mask, self.images[idx].stem

