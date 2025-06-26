import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
from tqdm import tqdm
import yaml
from monai.losses import DiceLoss
from models import UNetConcat, UNetSum, BasicUNetPlusPlus, BasicUNetPlusPlusSum
from datetime import datetime
import utils
import dataset as ds


def get_train_dataloaders(dataset_name, config):
    dataset_cfg = config['datasets'][dataset_name]
    DATASET_ROOT = dataset_cfg['data_root']
    DATASET_TRAIN = dataset_cfg['train_folder']
    DATASET_VAL = dataset_cfg['val_folder']

    train_cfg = config['training']
    TRAIN_BATCH_SIZE = train_cfg['batch_size']['train']
    VAL_BATCH_SIZE = train_cfg['batch_size']['val']

    # DATASETS
    if dataset_name == 'BUSI':
        train_data = ds.BUSIDataset(root=DATASET_ROOT, subset=DATASET_TRAIN)
        val_data = ds.BUSIDataset(root=DATASET_ROOT, subset=DATASET_VAL)
    elif dataset_name == 'DynamicNuclear':
        train_data = ds.DynamicNucDataset(root=DATASET_ROOT, subset=DATASET_TRAIN)
        val_data = ds.DynamicNucDataset(root=DATASET_ROOT, subset=DATASET_VAL)
    elif dataset_name == 'UsForKidney':
        train_data = ds.UsForKidneyDataset(root=DATASET_ROOT, subset=DATASET_TRAIN)
        val_data = ds.UsForKidneyDataset(root=DATASET_ROOT, subset=DATASET_VAL)
    elif dataset_name == 'Covid19Radio':
        train_data = ds.Covid19RadioDataset(root=DATASET_ROOT, subset=DATASET_TRAIN)
        val_data = ds.Covid19RadioDataset(root=DATASET_ROOT, subset=DATASET_VAL)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False)
    return train_loader, val_loader

def train(model_name, dataset_name, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_cfg = config['datasets'][dataset_name]
    NUM_CLASSES = dataset_cfg['num_classes']

    train_cfg = config['training']
    NUM_EPOCHS = train_cfg['num_epochs']
    LEARNING_RATE = train_cfg['learning_rate']
    RESULT_PATH = f'../results/train/{dataset_name}'
    EARLY_STOP_PATIENCE = train_cfg['early_stopping_patience']
    
    # OUTPUT PATHS
    timestamp = int(datetime.now().strftime("%d%m%M%S"))
    result_path = Path(RESULT_PATH)
    result_path.mkdir(parents=True, exist_ok=True)
    model_path = result_path / f'{model_name}_best_model_{timestamp}.pth'
    metrics_path = result_path / f'{model_name}_metrics_{timestamp}.json'

    # DATASETS
    train_loader, val_loader = get_train_dataloaders(dataset_name, config)

    # MODEL SELECTION
    if model_name == 'unetpp_concat':
        model = BasicUNetPlusPlus(spatial_dims=2, in_channels=3, out_channels=NUM_CLASSES).to(device)
    elif model_name == 'unetpp_sum':
        model = BasicUNetPlusPlusSum(spatial_dims=2, in_channels=3, out_channels=NUM_CLASSES).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    criterion = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stoppinng
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

        train_bar = tqdm(train_loader, desc=f"{model_name} - {dataset_name} - Epoch {epoch+1}", leave=False)
        for imgs, masks, _ in train_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            print('outputs len:', len(outputs))

            dice_loss_masks = masks.unsqueeze(1)
            dice_loss_outputs = outputs[0]
            print('dice_loss_outputs len:', len(dice_loss_outputs))
            loss = criterion(dice_loss_outputs, dice_loss_masks)
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
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)

                dice_loss_masks = masks.unsqueeze(1)    
                loss = criterion(outputs, dice_loss_masks)
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

    print(f"*** Training complete! Model and metrics saved in: {result_path} ***")


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_names = ['unetpp_concat', 'unetpp_sum']  
    dataset_names = list(config['datasets'].keys())  

    for dataset_name in dataset_names:
        for model_name in model_names:
            print(f"\n🚀 Starting training: {model_name} on {dataset_name}")
            train(model_name, dataset_name, config)

