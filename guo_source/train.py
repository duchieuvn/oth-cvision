import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from PIL import Image
import numpy as np
from monai.networks.nets import SwinUNETR
from utils import dice_score, plot_curves
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Hugging Face dataset
dataset = load_dataset("merve/pascal-voc", split="train")
subset_size = int(0.04 * len(dataset))  # use 4%
dataset = dataset.select(range(subset_size))

# Transform
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.transform(sample["image"])
        mask = np.array(sample["annotation"]["segmentation"], dtype=np.uint8)
        mask = Image.fromarray(mask).resize((128, 128), resample=Image.NEAREST)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

# Wrap
voc_dataset = VOCDataset(dataset, transform)

# Split train/val
train_size = int(0.8 * len(voc_dataset))
train_set, val_set = torch.utils.data.random_split(voc_dataset, [train_size, len(voc_dataset) - train_size])
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
val_loader = DataLoader(val_set, batch_size=2)

# Model
model = SwinUNETR(
    img_size=(128, 128, 1),
    in_channels=3,
    out_channels=21,
    feature_size=24,
    spatial_dims=2
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Training
epochs = 10
train_losses, val_scores = [], []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        imgs = imgs.unsqueeze(2)  # fake depth dim for 3D model
        outputs = model(imgs)
        loss = loss_fn(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    # Eval
    model.eval()
    with torch.no_grad():
        dices = []
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            imgs = imgs.unsqueeze(2)
            preds = torch.argmax(model(imgs), dim=1)
            dice = dice_score(preds, masks, num_classes=21)
            dices.append(dice)
        val_scores.append(sum(dices) / len(dices))

    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f}, Dice: {val_scores[-1]:.4f}")

# Save model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/swinunetr_voc_tiny.pth")

# Save plot
plot_curves(train_losses, val_scores, save_path="training_metrics.png")
