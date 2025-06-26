from torchvision.datasets import VOCSegmentation
from medsegbench import DynamicNuclearMSBench
from torchvision import transforms
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix    # pip install scikit-learn


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


def cityscapes_transform(img, mask, size=(224, 224)):
    transform_img = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Resize mask using NEAREST to preserve class IDs
    mask = mask.resize(size, resample=Image.NEAREST)
    mask = torch.as_tensor(np.array(mask), dtype=torch.long)
    
    return transform_img(img), mask



class CityscapesDataset(torch.utils.data.Dataset):
    def __init__(self, root, subset='train', size=(224, 224)):
        self.images = []
        self.masks = []
        self.size = size

        img_root = Path(root) / 'leftImg8bit_trainvaltest'/ 'leftImg8bit' / subset
        mask_root = Path(root) / 'gtFine_trainvaltest' / 'gtFine' / subset

        for city in img_root.iterdir():
            for img_path in city.glob("*_leftImg8bit.png"):
                stem = img_path.stem.replace("_leftImg8bit", "")
                label_path = mask_root / city.name / f"{stem}_gtFine_labelIds.png"
                if label_path.exists():
                    self.images.append(img_path)
                    self.masks.append(label_path)
                else:
                    print(f"[Warning] Missing label for {img_path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        img, mask = cityscapes_transform(img, mask, size=self.size)
        return img, mask, self.images[idx].stem

def compute_iou(preds, masks, num_classes):  # nhớ sửa num_classes cho đúng model của bạn
    ious = []
    preds = torch.argmax(preds, dim=1)
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (masks == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            continue
        ious.append(intersection / union)
    if len(ious) == 0:
        return 0
    return np.mean(ious)

def dice_score(pred, target, num_classes):
    dice = 0
    for i in range(1, num_classes):  # ignore background
        pred_i = (pred == i).float()
        target_i = (target == i).float()
        inter = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum()
        if union > 0:
            dice += 2 * inter / union
    return dice / (num_classes - 1)

def f1_score(pred, target, num_classes):
    f1 = 0
    for i in range(1, num_classes):  # ignore background
        pred_i = (pred == i).float()
        target_i = (target == i).float()
        tp = (pred_i * target_i).sum()
        fp = (pred_i * (1 - target_i)).sum()
        fn = ((1 - pred_i) * target_i).sum()
        if tp + fp + fn > 0:
            f1 += 2 * tp / (2 * tp + fp + fn)
    return f1 / (num_classes - 1)

def update_cm(cm, preds, targets, num_classes):
    """In-place update of a confusion-matrix tensor/ndarray."""
    cm += confusion_matrix(
        targets.view(-1).cpu().numpy(),
        preds.view(-1).cpu().numpy(),
        labels=list(range(num_classes))
    )
    return cm
