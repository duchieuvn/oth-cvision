import torch
import matplotlib.pyplot as plt

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

def plot_curves(train_losses, val_scores, save_path="training.png"):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_scores, label="Val Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Progress")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
