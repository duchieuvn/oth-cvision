import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import yaml
from PIL import Image

import utils
from models import UNetConcat, MonaiUnet, BasicUNetPlusPlus

def convert_to_correct_type(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_correct_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_correct_type(v) for v in obj]
    else:
        return obj

def evaluate_on_test(model_path, config):
    test_cfg = config['testing']
    result_path = Path(test_cfg['results_path'])

    #test_data = utils.BUSIDataset(root=config['data_root'], subset=config['test_folder'])
    test_data = utils.DynamicNucDataset(root=config['data_root_dn'], subset=config['test_dn'])
    test_loader = DataLoader(test_data, batch_size=test_cfg['batch_size'], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = test_cfg['num_classes']
    model = UNetConcat(out_channels=num_classes).to(device)
    if device.type == 'cuda':
        model.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss   = 0
    total_iou    = 0
    total_dice   = 0
    total_px     = 0
    total_correct = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    # Create folder to save prediction masks
    save_pred_dir = result_path / f"{config['model_name']}_predictions"
    save_pred_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Testing", leave=False)
        for imgs, masks, org_filenames in test_bar:
            imgs, masks = imgs.to(device), masks.to(device)

            logits = model(imgs)
            loss   = criterion(logits, masks)
            preds  = torch.argmax(logits, dim=1)

            # aggregate metrics
            total_loss += loss.item()
            total_iou  += utils.compute_iou(logits, masks, num_classes)
            total_dice += utils.dice_score(preds, masks, num_classes)
            total_correct += (preds == masks).sum().item()
            total_px      += masks.numel()
            cm = utils.update_cm(cm, preds, masks, num_classes)

            # save prediction masks
            for i in range(preds.shape[0]):
                pred_mask = preds[i].cpu().numpy().astype(np.uint8)
                out_path = save_pred_dir / f"pred_{org_filenames[i]}.png"
                Image.fromarray(pred_mask * (255 // (num_classes - 1))).save(out_path)

    n = len(test_loader)
    metrics = {
        "loss"       : total_loss / n,
        "mean_iou"   : total_iou  / n,
        "mean_dice"  : total_dice / n,
        "pixel_acc"  : total_correct / total_px,
        "conf_matrix": cm.tolist(),
        "iou_per_cls": (cm.diagonal() / cm.sum(1).clamp(min=1)).tolist()
    }

    with open(result_path / f'{config["model_name"]}_test_metrics.json', 'w') as f:
        json.dump(convert_to_correct_type(metrics), f, indent=2)

    print("\n  Test-set results")
    for k, v in metrics.items():
        if k not in {"conf_matrix", "iou_per_cls"}:
            print(f"  {k:12s}: {v:0.4f}")
    print("  IoU/class :", ["{:.3f}".format(x) for x in metrics['iou_per_cls']])
    print(f"\nSaved prediction masks to: {save_pred_dir}")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    best_model_path = Path(config['training']['results_path']) / f'{config["model_name"]}_best_model.pth'
    evaluate_on_test(best_model_path, config)

    print("All done!  Metrics and masks saved.")
