import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import time
import json
from pathlib import Path
from tqdm import tqdm
import yaml
import utils
from models import UNetConcat, MonaiUnet, BasicUNetPlusPlus, UNetSum

def train(model_name, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATASET_NAME = 'BUSI'
    busi_cfg = config['datasets'][DATASET_NAME]
    DATASET_ROOT = busi_cfg['data_root']
    DATASET_TRAIN = busi_cfg['train_folder']
    DATASET_VAL = busi_cfg['val_folder']
    NUM_CLASSES = busi_cfg['num_classes']

    train_cfg = config['training']
    NUM_EPOCHS = train_cfg['num_epochs']
    LEARNING_RATE = train_cfg['learning_rate']
    RESULT_PATH = '../results/{model_name}/train/{DATASET_NAME}'
    TRAIN_BATCH_SIZE = train_cfg['batch_size']['train']
    VAL_BATCH_SIZE = train_cfg['batch_size']['val']
    EARLY_STOP_PATIENCE = train_cfg['early_stopping_patience']
    
    # OUTPUT PATHS
    result_path = Path(RESULT_PATH)
    result_path.mkdir(parents=True, exist_ok=True)
    model_path = result_path / '{model_name}_{DATASET_NAME}_best_model.pth'
    metrics_path = result_path / '{model_name}_metrics.json'

    # DATASETS
    train_data = utils.BUSIDataset(root=DATASET_ROOT, subset=DATASET_TRAIN)
    val_data = utils.BUSIDataset(root=DATASET_ROOT, subset=DATASET_VAL)

    # train_data = utils.DynamicNucDataset("train", size=256)
    # val_data   = utils.DynamicNucDataset("val",   size=256)

    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False)

    # MODEL & TRAINING CONFIGURATION
    model = UNetConcat(out_channels=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if model_name == 'unet_sum':
        model = UNetSum(out_channels=NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # early stopping 
    best_loss = float('inf')
    early_stopping_count = 0
    metrics_records = []

    # TRAINING LOOP
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        total_train_iou = 0
        train_batches = 0
        start_time = time.time()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for imgs, masks, _ in train_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_iou += utils.compute_iou(outputs, masks, num_classes=NUM_CLASSES)
            train_batches += 1

            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / train_batches
        avg_train_iou = total_train_iou / train_batches

        # Validation
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
                total_val_iou += utils.compute_iou(outputs, masks, num_classes=NUM_CLASSES)
                val_batches += 1

        avg_val_loss = total_val_loss / val_batches
        avg_val_iou = total_val_iou / val_batches
        epoch_time = time.time() - start_time

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f} | "
              f"Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f} - Time: {epoch_time:.2f}s")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stopping_count = 0
            torch.save(model.state_dict(), model_path)
            print("✅ Saved best model.")
        else:
            early_stopping_count += 1
            print(f"⚠️ No improvement for {early_stopping_count} epochs.")
            if early_stopping_count >= EARLY_STOP_PATIENCE:
                print("🛑 Early stopping triggered!")
                break

        metrics_records.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_iou': avg_train_iou,
            'val_iou': avg_val_iou,
            'epoch_time': epoch_time
        })

    with open(metrics_path, 'w') as f:
        json.dump(metrics_records, f, indent=2)

    print(" Training complete!")
    print(f"Model and metrics saved in {RESULT_PATH}")

if __name__ == "__main__":

    # Load config.yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    for model_name in ['unet_concat', 'unet_sum']:
       train(model_name, config)
