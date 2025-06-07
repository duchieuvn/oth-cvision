from PIL import Image
import numpy as np
from pathlib import Path
import random

# Define paths
VOC_ROOT = Path("../VOCdevkit/VOC2012")
INPUT_MASK_DIR = VOC_ROOT / "SegmentationClassOriginal"  # Updated name
OUTPUT_MASK_DIR = VOC_ROOT / "SegmentationClass"  # Updated name
OUTPUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

# Mapping VOC labels (0–20) to new class groups (0–4)
VOC2FIVE_CLASS = {
    15: 1,  # Person class
    3: 2, 8: 2, 10: 2, 12: 2, 13: 2, 17: 2,  # Animals
    1: 3, 2: 3, 4: 3, 6: 3, 7: 3, 14: 3, 19: 3,  # Vehicles
    5: 4, 9: 4, 11: 4, 16: 4, 18: 4, 20: 4  # Objects
    # Background remains as class 0
}

def convert_to_5class_mask(input_path: Path, output_path: Path):
    """Convert segmentation mask from VOC classes to 5-class representation."""
    mask = np.array(Image.open(input_path))
    new_mask = np.zeros_like(mask, dtype=np.uint8)

    for voc_label, new_class in VOC2FIVE_CLASS.items():
        new_mask[mask == voc_label] = new_class

    Image.fromarray(new_mask).save(output_path)

def process_segmentation_files():
    """Split validation dataset into new val and test sets (50/50 randomly)."""
    seg_dir = VOC_ROOT / "ImageSets/Segmentation"
    val_file = seg_dir / "val.txt"

    with open(val_file) as f:
        val_list = f.read().splitlines()

    # Shuffle images randomly and split into two equal parts
    random.seed(42)
    random.shuffle(val_list)

    mid = len(val_list) // 2
    new_val = sorted(val_list[:mid])
    new_test = sorted(val_list[mid:])

    # Save new val and test datasets
    with open(seg_dir / "val.txt", "w") as f:
        f.write("\n".join(new_val) + "\n")

    with open(seg_dir / "test.txt", "w") as f:
        f.write("\n".join(new_test) + "\n")

    print(f"✅ Validation dataset split:")
    print(f"New val set: {len(new_val)} images")
    print(f"New test set: {len(new_test)} images")

def main():
    """Execute dataset processing and mask conversion."""
    process_segmentation_files()
    
    for mask_path in sorted(INPUT_MASK_DIR.glob("*.png")):
        output_path = OUTPUT_MASK_DIR / mask_path.name
        convert_to_5class_mask(mask_path, output_path)

    print(f"✅ Done! Converted masks saved in: {OUTPUT_MASK_DIR}")

if __name__ == "__main__":
    main()