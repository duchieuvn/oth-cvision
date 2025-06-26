from PIL import Image
import numpy as np
from pathlib import Path
import random
import os

# Define paths
VOC_ROOT = Path("../VOCdevkit/VOC2012")
INPUT_MASK_DIR = VOC_ROOT / "SegmentationClass"
OUTPUT_MASK_DIR = VOC_ROOT / "SegmentationClass5"
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

# Mapping: VOC class label (0–20) → new 5-class label
VOC2FIVE_CLASS = {
    15: 1,  # Person
    3: 2, 8: 2, 10: 2, 12: 2, 13: 2, 17: 2,  # Animals
    1: 3, 2: 3, 4: 3, 6: 3, 7: 3, 14: 3, 19: 3,  # Vehicles
    5: 4, 9: 4, 11: 4, 16: 4, 18: 4, 20: 4  # Objects
    # Background (0) is left unchanged
}

def convert_to_5class_mask(input_path: Path, output_path: Path):
    """Convert original VOC mask to new 5-class mask."""
    mask = np.array(Image.open(input_path))
    new_mask = np.zeros_like(mask, dtype=np.uint8)

    for voc_label, new_class in VOC2FIVE_CLASS.items():
        new_mask[mask == voc_label] = new_class

    Image.fromarray(new_mask).save(output_path)

def process_segmentation_files():
    """Shuffle and split val.txt into val/test 50/50."""
    seg_dir = VOC_ROOT / "ImageSets/Segmentation"
    val_file = seg_dir / "val.txt"

    with open(val_file) as f:
        val_list = f.read().splitlines()

    random.seed(42)
    random.shuffle(val_list)

    mid = len(val_list) // 2
    new_val = sorted(val_list[:mid])
    new_test = sorted(val_list[mid:])

    with open(seg_dir / "val.txt", "w") as f:
        f.write("\n".join(new_val) + "\n")

    with open(seg_dir / "test.txt", "w") as f:
        f.write("\n".join(new_test) + "\n")

    print(f"✅ Split complete:")
    print(f"New val: {len(new_val)} images")
    print(f"New test: {len(new_test)} images")

def main():
    """Run mask conversion and dataset split. Rename folders after completion."""
    process_segmentation_files()
    
    # Convert each mask to 5-class format
    for mask_path in sorted(INPUT_MASK_DIR.glob("*.png")):
        output_path = OUTPUT_MASK_DIR / mask_path.name
        convert_to_5class_mask(mask_path, output_path)

    print(f"✅ Done! Converted masks saved in: {OUTPUT_MASK_DIR}")

    # Rename original mask folder to preserve it
    original_dir = VOC_ROOT / "SegmentationClass"
    backup_dir = VOC_ROOT / "SegmentationClassOriginal"
    new_dir = VOC_ROOT / "SegmentationClass"

    if not backup_dir.exists():
        original_dir.rename(backup_dir)
        print("📁 Renamed SegmentationClass → SegmentationClassOriginal")

    # Replace original with 5-class version
    if not new_dir.exists():
        OUTPUT_MASK_DIR.rename(new_dir)
        print("📁 Renamed SegmentationClass5 → SegmentationClass")

if __name__ == "__main__":
    main()
