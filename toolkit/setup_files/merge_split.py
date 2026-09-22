"""
Script untuk merge dan split dataset
"""
import os
import random
import shutil
from pathlib import Path

def merge_datasets(source_dirs, output_dir):
    """Merge beberapa dataset menjadi satu"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)
    
    total_images = 0
    
    for source_dir in source_dirs:
        source = Path(source_dir)
        img_dir = source / "images"
        lbl_dir = source / "labels"
        
        if not img_dir.exists():
            print(f"Warning: {img_dir} tidak ditemukan")
            continue
        
        for img_file in img_dir.glob("*.jpg"):
            shutil.copy2(img_file, images_dir / img_file.name)
            lbl_file = lbl_dir / (img_file.stem + ".txt")
            if lbl_file.exists():
                shutil.copy2(lbl_file, labels_dir / lbl_file.name)
            total_images += 1
    
    print(f"Total gambar yang di-merge: {total_images}")
    return total_images

def split_dataset(source_dir, output_dir, train_ratio=0.8):
    """Split dataset menjadi train/val"""
    source = Path(source_dir)
    output = Path(output_dir)
    
    images_dir = source / "images"
    labels_dir = source / "labels"
    
    images = list(images_dir.glob("*.jpg"))
    random.shuffle(images)
    
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Create directories
    for split in ["train", "val"]:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for img_file in train_images:
        shutil.copy2(img_file, output / "images" / "train" / img_file.name)
        lbl_file = labels_dir / (img_file.stem + ".txt")
        if lbl_file.exists():
            shutil.copy2(lbl_file, output / "labels" / "train" / lbl_file.name)
    
    for img_file in val_images:
        shutil.copy2(img_file, output / "images" / "val" / img_file.name)
        lbl_file = labels_dir / (img_file.stem + ".txt")
        if lbl_file.exists():
            shutil.copy2(lbl_file, output / "labels" / "val" / lbl_file.name)
    
    print(f"Train: {len(train_images)} gambar")
    print(f"Val: {len(val_images)} gambar")

def main():
    print("=== MERGE & SPLIT DATASET ===")
    # Tambahkan kode sesuai kebutuhan

if __name__ == "__main__":
    main()
