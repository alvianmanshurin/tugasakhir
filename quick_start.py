"""Quick Start - Optimized for Intel i3-1115G4, 8GB RAM"""

import os
import sys
import subprocess
import psutil


def check_system():
    """Check system specifications."""
    print("=" * 60)
    print("SYSTEM CHECK")
    print("=" * 60)

    # CPU
    cpu = psutil.cpu_count(logical=False)
    cpu_logical = psutil.cpu_count()
    print(f"\n[CPU] Intel Core i3-1115G4")
    print(f"  Physical cores: {cpu}")
    print(f"  Logical cores: {cpu_logical}")

    # RAM
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    print(f"\n[RAM] {ram_gb:.1f} GB")

    # GPU
    print(f"\n[GPU] Intel UHD Graphics (Integrated)")
    print(f"  Status: NO CUDA - Using CPU only")

    # Recommendation
    print(f"\n[SETTINGS APPLIED]")
    print(f"  Model: YOLOv8n (Nano - 3.2M params)")
    print(f"  Image size: 416x416")
    print(f"  Batch size: 4")
    print(f"  Device: CPU")

    print("=" * 60)


def install_requirements():
    """Install required packages."""
    print("\n[STEP 1] Installing packages...")

    # Install PyTorch CPU first
    print("  Installing PyTorch (CPU version)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "--index-url",
        "https://download.pytorch.org/whl/cpu"
    ])

    # Install other requirements
    print("  Installing other packages...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])

    print("[OK] All packages installed")


def create_directories():
    """Create project directories."""
    dirs = [
        "data/raw", "data/annotated/images/train", "data/annotated/images/val",
        "data/annotated/labels/train", "data/annotated/labels/val",
        "models", "outputs/detections", "outputs/evaluation",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[OK] Directories created")


def download_model():
    """Download YOLOv8n model."""
    print("\n[STEP 3] Downloading YOLOv8n model...")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("[OK] YOLOv8n downloaded")
    except Exception as e:
        print(f"[ERROR] {e}")


def main():
    print("\n" + "=" * 60)
    print("VEHICLE DETECTION - QUICK START")
    print("Optimized for Intel i3-1115G4, 8GB RAM")
    print("=" * 60)

    check_system()

    print("\n[STEP 1] Install packages?")
    print("  (Skip if already installed)")
    resp = input("  Install? (y/n): ").strip().lower()

    if resp == 'y':
        install_requirements()

    print("\n[STEP 2] Creating directories...")
    create_directories()

    print("\n[STEP 3] Download model?")
    resp = input("  Download YOLOv8n? (y/n): ").strip().lower()
    if resp == 'y':
        download_model()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print("\nQuick commands:")
    print("  python src/train.py --quick          # Training cepat (10 epochs)")
    print("  python src/train.py                  # Training penuh (50 epochs)")
    print("  python src/detect.py --source foto.jpg  # Deteksi 1 gambar")
    print("  python src/detect.py --source data/raw  # Batch deteksi")
    print("  python src/realtime.py --source 0 --show  # Webcam")
    print("\nTips untuk laptop ini:")
    print("  - Gunakan YOLOv8n (sudah di-set default)")
    print("  - Image size 416 (bukan 640)")
    print("  - Batch size 4 (hemat RAM)")
    print("  - Training ~25-30 menit untuk 50 epochs")


if __name__ == "__main__":
    main()
