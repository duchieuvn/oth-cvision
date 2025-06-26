import random
from pathlib import Path

# Đường dẫn chứa file val.txt
seg_dir = Path("../VOCdevkit/VOC2012/ImageSets/Segmentation")

# Đọc file val.txt ban đầu
with open(seg_dir / "val.txt") as f:
    val_list = f.read().splitlines()

# Chia 50/50 ngẫu nhiên
random.seed(42)
random.shuffle(val_list)

mid = len(val_list) // 2
new_val = sorted(val_list[:mid])
new_test = sorted(val_list[mid:])

# Ghi lại file
with open(seg_dir / "val.txt", "w") as f:
    f.write("\n".join(new_val) + "\n")

with open(seg_dir / "test.txt", "w") as f:
    f.write("\n".join(new_test) + "\n")

print(f"✅ Validation set is splitted:")
print(f"New val: {len(new_val)} images")
print(f"New test: {len(new_test)} images")
