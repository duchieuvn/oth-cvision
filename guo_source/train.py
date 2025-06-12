import os
import torch
import torchvision.transforms as T
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
from monai.networks.nets import SwinUNETR
from utils import dice_score, plot_curves

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Dataset using torchvision
# ---------------------------
class VOC2DSegmentation(torch.utils.data.Dataset):
    def __init__(self, root, image_set="train", transform=None):
        self.voc = VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=True
        )
        self.transform = transform
        self.target_size = (128, 128)

    def __len__(self):
        return len(self.voc)

    def __getitem__(self, idx):
        image, target = self.voc[idx]

        # Resize and transform
        image = self.transform(image)
        target = target.resize(self.target_size, resample=Image.NEAREST)
        target = torch.from_numpy(np.array(target)).long()
        return image, target

# ---------------------------
# Transforms
# ---------------------------
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

# ---------------------------
# Load and Subset Dataset
# ---------------------------
full_dataset = VOC2DSegmentation(root="data", image_set="train", transform=transform)
subset_size = int(0.04 * len(full_dataset))
subset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])

# Train/Val split
train_size = int(0.8 * len(subset))
train_set, val_set = random_split(subset, [train_size, len(subset) - train_size])

train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
val_loader = DataLoader(val_set, batch_size=2)

# ---------------------------
# Swin UNETR model
# ---------------------------
model = SwinUNETR(
    img_size=(128, 128),       # now inferred dynamically in MONAI >= 1.3
    in_channels=3,
    out_channels=21,
    feature_size=24,
    spatial_dims=2             # for 2D images
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)

# ---------------------------
# Training Loop
# ---------------------------
epochs = 10
train_losses, val_scores = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))
    print("min:", masks.min().item(), "max:", masks.max().item())
    # Evaluation
    model.eval()
    with torch.no_grad():
        dices = []
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = torch.argmax(model(imgs), dim=1)
            dice = dice_score(preds, masks, num_classes=21)
            dices.append(dice)
        val_scores.append(sum(dices) / len(dices))

    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f}, Dice: {val_scores[-1]:.4f}")

# ---------------------------
# Save outputs
# ---------------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/swinunetr_voc_tiny.pth")

plot_curves(train_losses, val_scores, save_path="training_metrics.png")
