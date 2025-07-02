import torch
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix    

def compute_iou(preds, masks, num_classes): 
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
