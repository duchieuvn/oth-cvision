from torchvision.datasets import VOCSegmentation
from torchvision import transforms
import torch
from pathlib import Path
from PIL import Image
import numpy as np


def data_transform(img, mask, model_input_size=(224, 224)):
    img = img.resize(input_size)
    mask = mask.resize(input_size, resample=Image.NEAREST)
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
        return data_transform(img, mask)
    
class BUSIDataset(torch.utils.data.Dataset):
    def __init__(self, root, subset='train_folder'):
        self.images = []
        self.masks = []
        self.transform = data_transform  

        img_path = Path(root) / subset / 'img'
        for filename in sorted(img_path.glob("*.png")):
            self.images.append(filename)

        mask_path = Path(root) / subset / 'mask'
        for filename in sorted(mask_path.glob("*.png")):
            self.masks.append(mask_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')  # Grayscale mask

        img, mask = self.transform(img, mask, model_input_size=(512, 512))

        # Convert to binary mask: 0 for background, 1 for lesion
        mask = (mask > 0).long()

        return img, mask
    
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