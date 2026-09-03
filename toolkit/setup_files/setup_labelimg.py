"""LabelImg Setup Guide for Vehicle Annotation
Panduan instalasi dan penggunaan LabelImg untuk anotasi dataset
"""

import os
import subprocess
import sys
from pathlib import Path


def check_labelimg():
    """Check if LabelImg is installed."""
    try:
        result = subprocess.run([sys.executable, "-m", "labelImg", "--help"],
                              capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False


def install_labelimg():
    """Install LabelImg."""
    print("=" * 60)
    print("INSTALLING LABELIMG")
    print("=" * 60)

    print("\n[INFO] Installing LabelImg...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "labelImg"
        ])
        print("[OK] LabelImg installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Installation failed: {e}")
        return False


def create_labelimg_config():
    """Create LabelImg configuration for vehicle classes."""
    print("\n[INFO] Creating LabelImg configuration...")

    # Create predefined_classes.txt for LabelImg
    classes = [
        "motor",
        "mobil",
        "bus",
        "truk"
    ]

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    classes_file = config_dir / "predefined_classes.txt"
    with open(classes_file, "w") as f:
        f.write("\n".join(classes))

    print(f"[OK] Classes file created: {classes_file}")
    print(f"[INFO] Classes: {classes}")

    return classes_file


def show_guide():
    """Show LabelImg usage guide."""
    print("\n" + "=" * 60)
    print("LABELIMG USAGE GUIDE")
    print("Vehicle Annotation for YOLO Format")
    print("=" * 60)

    guide = """
1. INSTALLATION
   pip install labelImg

2. LAUNCH LABELImg
   labelImg

   Atau dengan direktori default:
   labelImg data/annotated/images/train

3. PENGATURAN AWAL
   - Klik "Change Save Dir" -> pilih folder labels
   - Pastikan format: YOLO (bukan PascalVOC)
   - Klik "Edit" -> "Change Default Saved Annotation Dir"
     -> pilih "data/annotated/labels/train"

4. KEYBOARD SHORTCUTS
   W         : Create new rectangle box
   D         : Next image
   A         : Previous image
   Del/Delete: Delete selected box
   Ctrl+S    : Save
   Ctrl+u    : Load all images from dir
   Ctrl+shift+S: Change save dir

5. PROSES ANOTASI
   a. Load gambar dari folder
   b. Tekan W untuk membuat bounding box
   c. Gambar rectangle di sekitar kendaraan
   d. Pilih label: motor, mobil, bus, atau truk
   e. Tekan D untuk lanjut ke gambar berikutnya
   f. Ulangi sampai semua gambar selesai
   g. Tekan Ctrl+S untuk save

6. FORMAT OUTPUT (YOLO)
   File .txt akan dibuat otomatis:
   <class_id> <x_center> <y_center> <width> <height>
   
   Contoh:
   0 0.5 0.3 0.1 0.2
   1 0.7 0.6 0.15 0.25

7. CLASS MAPPING
   0 = motor
   1 = mobil
   2 = bus
   3 = truk

8. TIPS
   - Anotasi harus tepat (tight) di sekitar objek
   - Jangan potong kendaraan
   - Anotasi semua kendaraan yang terlihat
   - Simpan secara berkala (Ctrl+S)
   - Gunakan "Verified" untuk menandai sudah dicek

9. STRUKTUR FILE
   data/annotated/
   ├── images/
   │   ├── train/
   │   │   ├── img_00001.jpg
   │   │   └── ...
   │   └── val/
   │       ├── img_00100.jpg
   │       └── ...
   └── labels/
       ├── train/
       │   ├── img_00001.txt  (dibuat otomatis oleh LabelImg)
       │   └── ...
       └── val/
           ├── img_00100.txt
           └── ...

10. VERIFIKASI
    Setelah anotasi, jalankan:
    python src/dataset_prepare.py --action validate
"""
    print(guide)

    # Save guide to file
    guide_path = Path("docs/LABELIMG_GUIDE.txt")
    guide_path.parent.mkdir(parents=True, exist_ok=True)
    with open(guide_path, "w") as f:
        f.write(guide)
    print(f"[OK] Guide saved to: {guide_path}")


def launch_labelimg():
    """Launch LabelImg with correct settings."""
    print("\n[INFO] Launching LabelImg...")

    # Ensure directories exist
    Path("data/annotated/images/train").mkdir(parents=True, exist_ok=True)
    Path("data/annotated/labels/train").mkdir(parents=True, exist_ok=True)

    try:
        subprocess.Popen([
            sys.executable, "-m", "labelImg",
            "data/annotated/images/train",
            "data/annotated/labels/train",
        ])
        print("[OK] LabelImg launched!")
    except Exception as e:
        print(f"[ERROR] Failed to launch: {e}")
        print("[INFO] Try running manually: labelImg")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LabelImg Setup Tool")
    parser.add_argument("--action", type=str, default="guide",
                       choices=["install", "config", "guide", "launch", "check"],
                       help="Action to perform")
    args = parser.parse_args()

    if args.action == "check":
        if check_labelimg():
            print("[OK] LabelImg is installed")
        else:
            print("[WARNING] LabelImg is not installed")
            print("[INFO] Run: python setup_labelimg.py --action install")

    elif args.action == "install":
        install_labelimg()

    elif args.action == "config":
        create_labelimg_config()

    elif args.action == "guide":
        create_labelimg_config()
        show_guide()

    elif args.action == "launch":
        if not check_labelimg():
            print("[INFO] LabelImg not found, installing...")
            install_labelimg()
        launch_labelimg()


if __name__ == "__main__":
    main()
