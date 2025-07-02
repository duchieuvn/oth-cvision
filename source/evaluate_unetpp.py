import torch
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import yaml
from monai.losses import DiceLoss

from models import BasicUNetPlusPlus, BasicUNetPlusPlusSum
import utils
import dataset as ds


def convert_to_correct_type(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_correct_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_correct_type(v) for v in obj]
    else:
        return obj


def get_test_dataset(dataset_name, dataset_cfg):
    root = dataset_cfg['data_root']
    subset = dataset_cfg['test_folder']

    if dataset_name == 'BUSI':
        return ds.BUSIDataset(root=root, subset=subset)
    elif dataset_name == 'DynamicNuclear':
        return ds.DynamicNucDataset(root=root, subset=subset)
    elif dataset_name == 'UsForKidney':
        return ds.UsForKidneyDataset(root=root, subset=subset)
    elif dataset_name == 'Covid19Radio':
        return ds.Covid19RadioDataset(root=root, subset=subset)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")


def get_model(model_name, num_classes):
    if model_name == 'unetpp_concat':
        return BasicUNetPlusPlus(spatial_dims=2, in_channels=3, out_channels=num_classes)
    elif model_name == 'unetpp_sum':
        return BasicUNetPlusPlusSum(spatial_dims=2, in_channels=3, out_channels=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def evaluate(model, test_loader, criterion, device, result_path, model_tag, num_classes):
    model.eval()
    total_loss = 0
    total_iou = 0
    total_dice = 0
    total_px = 0
    total_correct = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    save_pred_dir = result_path / f"{model_tag}_predictions"
    save_pred_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        test_bar = tqdm(test_loader, desc=f"Testing: {model_tag}", leave=False)
        for imgs, masks, filenames in test_bar:
            imgs, masks = imgs.to(device), masks.to(device)

            logits = model(imgs)[0]  # <-- UNet++ output is a tuple
            loss = criterion(logits, masks.unsqueeze(1))
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            total_iou += utils.compute_iou(logits, masks, num_classes)
            total_dice += utils.dice_score(preds, masks, num_classes)
            total_correct += (preds == masks).sum().item()
            total_px += masks.numel()
            cm = utils.update_cm(cm, preds, masks, num_classes)

            for i in range(preds.shape[0]):
                pred_mask = preds[i].cpu().numpy().astype(np.uint8)
                out_path = save_pred_dir / f"pred_{filenames[i]}.png"
                Image.fromarray(pred_mask * (255 // (num_classes - 1))).save(out_path)

    n = len(test_loader)
    metrics = {
        "loss": total_loss / n,
        "mean_iou": total_iou / n,
        "mean_dice": total_dice / n,
        "pixel_acc": total_correct / total_px,
        "conf_matrix": cm.tolist(),
        "iou_per_cls": (cm.diagonal() / cm.sum(1).clamp(min=1)).tolist()
    }

    with open(result_path / f'{model_tag}_test_metrics.json', 'w') as f:
        json.dump(convert_to_correct_type(metrics), f, indent=2)

    print(f"\n✅ Results for {model_tag}")
    for k, v in metrics.items():
        if k not in {"conf_matrix", "iou_per_cls"}:
            print(f"  {k:12s}: {v:0.4f}")
    print("  IoU/class :", ["{:.3f}".format(x) for x in metrics['iou_per_cls']])
    print(f"Saved prediction masks to: {save_pred_dir}\n")


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_names = ['unetpp_concat', 'unetpp_sum']
    dataset_names = list(config['datasets'].keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in dataset_names:
        dataset_cfg = config['datasets'][dataset_name]
        num_classes = dataset_cfg['num_classes']
        test_dataset = get_test_dataset(dataset_name, dataset_cfg)
        
        result_path = Path(f"../results/inference/{dataset_name}")
        result_path.mkdir(parents=True, exist_ok=True)

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['testing']['batch_size'],
            shuffle=False
        )

        for model_name in model_names:
            model_tag = f"{model_name}_{dataset_name}"
            print(f"🔍 Evaluating {model_tag}")

            model = get_model(model_name, num_classes).to(device)


            model_dir = Path(f'../results/train/{dataset_name}')
            model_files = list(model_dir.glob(f"{model_name}_best_model_*.pth"))

            
            if not model_files:
                print(f"❌ Model checkpoint not found for {model_tag}")
                continue

            model_path = model_files[0]
            print(f"📦 Loading model from: {model_path.name}")
            model.load_state_dict(torch.load(model_path, map_location=device))

            criterion = DiceLoss(to_onehot_y=True, softmax=True)
            evaluate(model, test_loader, criterion, device, result_path, model_tag, num_classes)

    print("🎉 All UNet++ evaluations complete.")
