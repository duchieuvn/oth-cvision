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
from models import UNetConcat, MonaiUnet, BasicUNetPlusPlusSum, BasicUNetPlusPlus


# Load config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Shortcuts for YAML structure
train_cfg = config['training']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_data = utils.BUSIDataset(root=config['data_root'], subset=config['train_folder'])
val_data = utils.BUSIDataset(root=config['data_root'], subset=config['val_folder'])
test_data = utils.BUSIDataset(root=config['data_root'], subset=config['test_folder'])

train_loader = DataLoader(train_data, batch_size=train_cfg['batch_size']['train'], shuffle=True)
val_loader = DataLoader(val_data, batch_size=train_cfg['batch_size']['eval'], shuffle=False)
test_loader = DataLoader(test_data, batch_size=train_cfg['batch_size']['eval'], shuffle=False)


def train():
    num_classes = train_cfg['num_classes']
    #model = UNetConcat(out_channels=num_classes).to(device)
    
    model = BasicUNetPlusPlusSum(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_classes,
        features=(16, 32, 64, 128, 256, 16),
        deep_supervision=False,
        act=("ReLU", {"inplace": True}),
        norm=("batch", {"affine": True})
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])

    result_path = Path(config['results_path'])
    result_path.mkdir(parents=True, exist_ok=True)

    model_path = result_path / 'UNetPlusPlusSum_best_model.pth'
    metrics_path = result_path / 'UNetPlusPlusSum_metrics.json'

    patience = train_cfg['early_stopping_patience']
    best_loss = float('inf')
    early_stopping_count = 0
    metrics_records = []

    for epoch in range(train_cfg['num_epochs']):
        model.train()
        total_train_loss = 0
        total_train_iou = 0
        train_batches = 0
        start_time = time.time()

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for imgs, masks in train_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            #outputs = model(imgs) unet
            outputs = model(imgs)[0] # unetpp

            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_iou += utils.compute_iou(outputs, masks, num_classes=num_classes)
            train_batches += 1

            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / train_batches
        avg_train_iou = total_train_iou / train_batches

        model.eval()
        total_val_loss = 0
        total_val_iou = 0
        val_batches = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                #outputs = model(imgs) unet
                outputs = model(imgs)[0] # unetpp

                loss = criterion(outputs, masks)
                total_val_loss += loss.item()
                total_val_iou += utils.compute_iou(outputs, masks, num_classes=num_classes)
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
            if early_stopping_count >= patience:
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


def tensor_to_obj(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: tensor_to_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_obj(v) for v in obj]
    else:
        return obj



def evaluate_on_test(model_path, result_path):
    num_classes = train_cfg['num_classes']
    #model = UNetConcat(out_channels=num_classes).to(device)
    model = BasicUNetPlusPlusSum(
        spatial_dims=2,
        in_channels=3,
        out_channels=num_classes,
        features=(16, 32, 64, 128, 256, 16),
        deep_supervision=False,
        act=("ReLU", {"inplace": True}),
        norm=("batch", {"affine": True})
    ).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss   = 0
    total_iou    = 0
    total_dice   = 0
    total_px     = 0
    total_correct = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing", leave=False)
        for imgs, masks in test_bar:
            imgs, masks = imgs.to(device), masks.to(device)

            logits = model(imgs)
            loss   = criterion(logits, masks)

            preds  = torch.argmax(logits, dim=1)

            # aggregate
            total_loss += loss.item()
            total_iou  += utils.compute_iou(logits, masks, num_classes)
            total_dice += utils.dice_score(preds, masks, num_classes)
            total_correct += (preds == masks).sum().item()
            total_px      += masks.numel()
            cm = utils.update_cm(cm, preds, masks, num_classes)

    n = len(test_loader)
    metrics = {
        "loss"       : total_loss / n,
        "mean_iou"   : total_iou  / n,
        "mean_dice"  : total_dice / n,
        "pixel_acc"  : total_correct / total_px,
        "conf_matrix": cm.tolist(),                      # JSON-friendly
        "iou_per_cls": (cm.diagonal() / cm.sum(1).clamp(min=1)).tolist()
    }

    # save
    with open(result_path / 'UNetPlusPlusSum_test_metrics.json', 'w') as f:
        json.dump(tensor_to_obj(metrics), f, indent=2)

    # pretty print
    print("\n  Test-set results")
    for k, v in metrics.items():
        if k not in {"conf_matrix", "iou_per_cls"}:
            print(f"  {k:12s}: {v:0.4f}")
    print("  IoU / class :", ["{:.3f}".format(x) for x in metrics['iou_per_cls']])


if __name__ == "__main__":
    train()
    print(" Training complete!")
    print(f"Model and metrics saved in '{config['results_path']}'")
    best_model_path = Path(config['results_path']) / 'UNetPlusPlusSum_best_model.pth'
    evaluate_on_test(best_model_path, Path(config['results_path']))

    print("All done!  Metrics stored in UNetPlusPlusSum_test_metrics.json")