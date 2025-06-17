"""
Convert MedSegBench DynamicNuclearMSBench -> BUSIDataset-style PNG folders
--------------------------------------------------------------------------

* Creates   <out_root>/<split>/{images|masks}/xxxxx.png
* Replicates the single-channel input to RGB so BasicUNetPlusPlus(in_channels=3) works.
* Converts mask 255 → 1 (foreground) so CrossEntropyLoss expects {0,1} labels.
"""

import os, argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from medsegbench import DynamicNuclearMSBench

def save_png(arr, path, replicate=False):
    if replicate:                      # grayscale -> RGB
        arr = np.stack([arr]*3, axis=2)
    Image.fromarray(arr).save(path, compress_level=0)

def convert_split(split, out_root, size):
    ds = DynamicNuclearMSBench(split=split, size=size, download=True)
    img_dir  = os.path.join(out_root, split, "images")
    msk_dir  = os.path.join(out_root, split, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    for idx, (img_pil, msk_np) in enumerate(tqdm(ds, desc=f"{split:5s}")):
        img_np = np.array(img_pil, dtype=np.uint8)
        msk_np = (msk_np > 0).astype(np.uint8)          # 0/255 → 0/1

        fname = f"{idx:05d}.png"
        save_png(img_np, os.path.join(img_dir, fname), replicate=True)
        save_png(msk_np*255, os.path.join(msk_dir, fname))  # keep 0/255 for nice viewing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True, help="where the PNG folders will be written")
    parser.add_argument("--size", type=int, default=256, choices=[128,256,512])
    args = parser.parse_args()

    for split in ["train", "val", "test"]:
        convert_split(split, args.out_root, args.size)

    print("✅ DynamicNuclear dataset exported in BUSI format!")
