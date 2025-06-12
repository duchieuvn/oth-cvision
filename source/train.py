import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch
import time
import json
import os
from pathlib import Path
from tqdm import tqdm

import utils
from models import UNetConcat, MonaiUNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_root = '../Hi-gMISnet_all_dataset/BUSI'
train_data = utils.BUSIDataset(root=data_root, subset='train_folder')
val_data = utils.BUSIDataset(root=data_root, subset='val_folder')
test_data = utils.BUSIDataset(root=data_root, subset='test_folder')

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False)
test_loader = DataLoader(test_data, batch_size=4, shuffle=False)

def train():
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetConcat(out_channels=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    result_path = Path('../results/busi/train')
    result_path.mkdir(parents=True, exist_ok=True)

    model_path = result_path / 'UNetConcat_best_model.pth'



    patience = 5  # for early stopping
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
            total_train_iou += utils.compute_iou(outputs, masks, num_classes=num_classes)
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
                total_val_iou += utils.compute_iou(outputs, masks, num_classes=num_classes)
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

    metrics_path = result_path / 'metrics.json'

    # Save metrics to a JSON file for structured and human-readable storage
    with open(metrics_path, 'w') as f:
        json.dump(metrics_records, f, indent=2)


if __name__ == "__main__":
    
    train()
    print("✅ Training complete!")
    print("Model and metrics saved in './results/train/'")






















