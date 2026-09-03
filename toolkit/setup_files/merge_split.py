import os
import shutil
import random
from pathlib import Path

# Source directories
raw_dir = Path("C:/Users/Alvian/OneDrive/Dokumen/GitHub/tugasakhir/data/raw")
merged_dir = raw_dir / "merged"
merged_dir.mkdir(exist_ok=True)

# Merge all frames
count = 0
for subdir in ["KIRI-7", "KIRI-9", "TENGAH-7", "TENGAH-9"]:
    src = raw_dir / subdir
    if src.exists():
        files = list(src.glob("*.jpg"))
        for f in files:
            dst = merged_dir / f"merged_{count:05d}.jpg"
            shutil.copy2(f, dst)
            count += 1
        print(f"Merged {len(files)} frames from {subdir}")

print(f"Total merged: {count} frames")

# Split to train/val
images = list(merged_dir.glob("*.jpg"))
random.shuffle(images)

train_ratio = 0.8
split_idx = int(len(images) * train_ratio)

train_files = images[:split_idx]
val_files = images[split_idx:]

# Create directories
train_img = Path("C:/Users/Alvian/OneDrive/Dokumen/GitHub/tugasakhir/data/annotated/images/train")
val_img = Path("C:/Users/Alvian/OneDrive/Dokumen/GitHub/tugasakhir/data/annotated/images/val")
train_lbl = Path("C:/Users/Alvian/OneDrive/Dokumen/GitHub/tugasakhir/data/annotated/labels/train")
val_lbl = Path("C:/Users/Alvian/OneDrive/Dokumen/GitHub/tugasakhir/data/annotated/labels/val")

for d in [train_img, val_img, train_lbl, val_lbl]:
    d.mkdir(parents=True, exist_ok=True)

# Copy train
for f in train_files:
    shutil.copy2(f, train_img / f.name)
    (train_lbl / (f.stem + ".txt")).touch()

# Copy val
for f in val_files:
    shutil.copy2(f, val_img / f.name)
    (val_lbl / (f.stem + ".txt")).touch()

print(f"Train: {len(train_files)} images")
print(f"Val: {len(val_files)} images")
print("Done!")
