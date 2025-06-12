import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import torch
from model import UNetConcat

import time
import json
import os
from tqdm import tqdm


input_size = (128, 128)

def voc_transform(img, mask):
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
        return voc_transform(img, mask)


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


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = VOCSegmentationDataset(root='../', image_set='train')
    val_data = VOCSegmentationDataset(root='../', image_set='val')
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)


    model = UNetConcat(out_channels=21).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)


    model_path = './results/train/UNetConcat_best_model.pth'



    patience = 5  # số epoch không cải thiện liên tiếp để dừng
    best_loss = float('inf')
    counter = 0

    metrics_records = []

    # List of keys for clarity (not strictly needed for JSON, but helps with consistency)
    keys = ['epoch', 'train_loss', 'val_loss', 'train_iou', 'val_iou', 'epoch_time']


    for epoch in range(300):
        model.train()
        total_train_loss = 0
        total_train_iou = 0
        train_batches = 0
        start_time = time.time()

        # Iterate through training data
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for imgs, masks in train_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_iou += compute_iou(outputs, masks, num_classes=21)
            train_batches += 1

            train_bar.set_postfix(loss=loss.item())

        # Calculate average training metrics
        avg_train_loss = total_train_loss / train_batches
        avg_train_iou = total_train_iou / train_batches

        # Validation loop
        model.eval()
        total_val_loss = 0
        total_val_iou = 0
        val_batches = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)

                loss = criterion(outputs, masks)
                total_val_loss += loss.item()
                total_val_iou += compute_iou(outputs, masks, num_classes=21)
                val_batches += 1

        # Calculate average validation metrics
        avg_val_loss = total_val_loss / val_batches
        avg_val_iou = total_val_iou / val_batches

        epoch_time = time.time() - start_time
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f} - Time: {epoch_time:.2f}s")

        # Save the best model based on validation loss
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), model_path)
            print("✅ Saved best model.")
        else:
            counter += 1
            print(f"⚠️ No improvement for {counter} epochs.")
            if counter >= patience:
                print("🛑 Early stopping triggered!")
                break

        # Append metrics for the current epoch
        metrics_records.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_iou': avg_train_iou,
            'val_iou': avg_val_iou,
            'epoch_time': epoch_time
        })

    metrics_path = './results/train/metrics.json'
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # Save metrics to a JSON file for structured and human-readable storage
    with open(metrics_path, 'w') as f:
        json.dump(metrics_records, f, indent=2)

























