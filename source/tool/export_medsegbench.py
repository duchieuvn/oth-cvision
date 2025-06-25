"""
Convert MedSegBench datasets to BUSIDataset-style PNG folders
-------------------------------------------------------------

- Output: <out_root>/<dataset>/<split>/{images|masks}/xxxxx.png
- Replicates grayscale input to RGB (for UNet with in_channels=3).
- Converts mask 255 → 1 (foreground) for use with CrossEntropyLoss.
- Handles category-specific splits for Covid19Radio.
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

from medsegbench import DynamicNuclearMSBench, UsForKidneyBench, Covid19RadioMSBench

def save_png(arr, path, replicate=False):
    if replicate:
        arr = np.stack([arr] * 3, axis=2)  # grayscale → RGB
    Image.fromarray(arr).save(path, compress_level=0)

def convert_split(dataset_cls, split, out_root, size, category=None):
    # Instantiate dataset, with or without category
    if category:
        ds = dataset_cls(split=split, size=size, download=True, category=category)
        split_folder = f"{split}_{category}"
    else:
        ds = dataset_cls(split=split, size=size, download=True)
        split_folder = split

    img_dir = os.path.join(out_root, split_folder, "images")
    msk_dir = os.path.join(out_root, split_folder, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    for idx, (img_pil, msk_np) in enumerate(tqdm(ds, desc=f"{split_folder:15s}")):
        img_np = np.array(img_pil, dtype=np.uint8)
        msk_np = (msk_np > 0).astype(np.uint8)  # 0/255 → 0/1

        fname = f"{idx:05d}.png"
        save_png(img_np, os.path.join(img_dir, fname), replicate=True)
        save_png(msk_np * 255, os.path.join(msk_dir, fname))  # use 0/255 for visual clarity

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True, help="Root directory for output PNG folders")
    parser.add_argument("--size", type=int, default=256, choices=[128, 256, 512], help="Resize images to this size")
    args = parser.parse_args()

    # Dataset class mappings
    datasets = {
        "DynamicNuclear": DynamicNuclearMSBench,
        "UsForKidney": UsForKidneyBench,
        "Covid19Radio": Covid19RadioMSBench,
    }

    covid_categories = ["C1", "C2", "C3", "C4"]
    standard_splits = ["train", "val", "test"]

    for name, dataset_cls in datasets.items():
        print(f"\n Processing {name}")
        dataset_root = os.path.join(args.out_root, name)

        if name == "Covid19Radio":
            # Export global splits
            for split in standard_splits:
                convert_split(dataset_cls, split, dataset_root, args.size)
            # Export category-specific splits
            for category in covid_categories:
                for split in standard_splits:
                    convert_split(dataset_cls, split, dataset_root, args.size, category=category)
        else:
            for split in standard_splits:
                convert_split(dataset_cls, split, dataset_root, args.size)

    print("\n✅ All MedSegBench datasets exported.")
