import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
import torch

from medsegbench import DynamicNuclearMSBench, USforKidneyMSBench, Covid19RadioMSBench

# ==== Dataset Conversion Helpers ====

def save_png(arr, path, replicate=False):
    if replicate:
        arr = np.stack([arr] * 3, axis=2)  # grayscale -> RGB
    Image.fromarray(arr).save(path, compress_level=0)

def convert_split(dataset_cls, split, out_root, size, category=None):
    try:
        if category:
            ds = dataset_cls(split=split, size=size, category=category, download=True)
            split_folder = f"{split}_{category}"
        else:
            ds = dataset_cls(split=split, size=size, download=True)
            split_folder = split
    except (AssertionError, ValueError) as e:
        print(f"⚠️ Skipping invalid split: {split}{'_'+category if category else ''} – {e}")
        return

    
    img_dir = os.path.join(out_root, split_folder, "images")
    msk_dir = os.path.join(out_root, split_folder, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    for idx, (img_pil, msk_np) in enumerate(tqdm(ds, desc=f"{split_folder:15s}")):
        img_np = np.array(img_pil, dtype=np.uint8)
        msk_np = (msk_np > 0).astype(np.uint8)

        fname = f"{idx:05d}.png"
        save_png(img_np, os.path.join(img_dir, fname), replicate=True)
        save_png(msk_np * 255, os.path.join(msk_dir, fname))


class DynamicNucDataset(torch.utils.data.Dataset):
    def __init__(self, root, subset='train'):
        self.images = sorted((Path(root) / subset / 'images').glob("*.png"))
        self.masks = sorted((Path(root) / subset / 'masks').glob("*.png"))
        self.transform = lambda x, y: (torch.tensor(np.array(x)).permute(2, 0, 1) / 255.0,
                                       torch.tensor(np.array(y)))

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        img, mask = self.transform(img, mask)
        mask = (mask > 0).long()
        return img, mask, self.images[idx].stem

# Reuse same structure for the other two
UsForKidneyDataset = DynamicNucDataset
Covid19RadioDataset = DynamicNucDataset

# ==== Test Loader ====

def test_dataset(dataset_class, root, split, label):
    print(f"\n🔍 Testing {label}: {split}")
    try:
        dataset = dataset_class(root=root, subset=split)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        sample = next(iter(loader))
        print(f"✅ Loaded {label} [{split}]: {len(dataset)} samples, shape: {sample[0].shape}, {sample[1].shape}")
    except Exception as e:
        print(f"❌ Failed to load {label} [{split}]: {e}")

# ==== Main ====

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True, help="Root directory to save all PNG datasets")
    parser.add_argument("--size", type=int, default=256, choices=[128, 256, 512])
    args = parser.parse_args()

    datasets = {
        "DynamicNuclear": DynamicNuclearMSBench,
        "UsForKidney": USforKidneyMSBench,
        "Covid19Radio": Covid19RadioMSBench
    }

    covid_categories = ['C1', 'C2', 'C3', 'C4']
    standard_splits = ['train', 'val', 'test']

    print("📥 Exporting datasets...")
    for name, dataset_cls in datasets.items():
        print(f"\n📦 Processing {name}")
        dataset_root = os.path.join(args.out_root, name)

        for split in standard_splits:
            convert_split(dataset_cls, split, dataset_root, args.size)

    print("\n🧪 Testing dataset loading...")

    test_dataset(DynamicNucDataset, os.path.join(args.out_root, "DynamicNuclear"), 'train', "DynamicNuclear")
    test_dataset(DynamicNucDataset, os.path.join(args.out_root, "DynamicNuclear"), 'val', "DynamicNuclear")

    test_dataset(UsForKidneyDataset, os.path.join(args.out_root, "UsForKidney"), 'train', "UsForKidney")
    test_dataset(UsForKidneyDataset, os.path.join(args.out_root, "UsForKidney"), 'val', "UsForKidney")

    test_dataset(Covid19RadioDataset, os.path.join(args.out_root, "Covid19Radio"), 'train', "Covid19Radio")
    test_dataset(Covid19RadioDataset, os.path.join(args.out_root, "Covid19Radio"), 'val', "Covid19Radio")

    for cat in ['C1', 'C2', 'C3', 'C4']:
        split = f"train_{cat}"
        test_dataset(Covid19RadioDataset, os.path.join(args.out_root, "Covid19Radio"), split, f"Covid19Radio-{cat}")

    print("\n✅ All done!")
