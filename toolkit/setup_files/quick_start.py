"""
Script setup cepat untuk proyek deteksi kendaraan
"""
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
    print(f"\n[CPU]")
    print(f"  Physical cores: {cpu}")
    print(f"  Logical cores: {cpu_logical}")

    # RAM
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    print(f"\n[RAM] {ram_gb:.1f} GB")

    # GPU
    print(f"\n[GPU]")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  {torch.cuda.get_device_name(0)}")
        else:
            print("  No CUDA - Using CPU only")
    except:
        print("  Unable to detect GPU")

    # Recommendation
    print(f"\n[SETTINGS APPLIED]")
    print(f"  Model: YOLOv11n (Nano - 2.6M params)")
    print(f"  Image size: 416x416")
    print(f"  Batch size: 4")
    print(f"  Device: CPU")

    print("=" * 60)


def install_requirements():
    """Instal requirements yang diperlukan"""
    print("\n=== INSTALASI REQUIREMENTS ===")

    print(f"Python: {sys.version}")

    packages = [
        "ultralytics",
        "opencv-python",
        "pyyaml",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "psutil",
        "Pillow",
    ]

    for pkg in packages:
        print(f"Installing {pkg}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  [OK] {pkg}")
        except subprocess.CalledProcessError:
            print(f"  [ERROR] Failed to install {pkg}")

    print("\n=== INSTALASI SELESAI ===")


def create_directories():
    """Buat struktur direktori proyek"""
    print("\n=== MEMBUAT DIREKTORI ===")

    dirs = [
        "data/raw",
        "data/annotated/images/train",
        "data/annotated/images/val",
        "data/annotated/labels/train",
        "data/annotated/labels/val",
        "models/vehicle_detection/weights",
        "outputs/detections",
        "outputs/evaluation",
        "runs",
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  [OK] {d}")

    print("=== DIREKTORI SELESAI ===")


def download_model():
    """Download YOLOv11n model."""
    print("\n=== DOWNLOAD MODEL ===")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov11n.pt")
        print("[OK] YOLOv11n downloaded")
    except Exception as e:
        print(f"[ERROR] {e}")


def main():
    print("\n" + "=" * 60)
    print("VEHICLE DETECTION - QUICK START")
    print("Optimized for Intel i3-1115G4, 8GB RAM")
    print("=" * 60)

    # Step 1: Check system
    print("\n[STEP 1] Checking system...")
    check_system()

    # Step 2: Install packages
    print("\n[STEP 2] Install packages?")
    resp = input("  Install? (y/n): ").strip().lower()
    if resp == 'y':
        install_requirements()

    # Step 3: Create directories
    print("\n[STEP 3] Creating directories...")
    create_directories()

    # Step 4: Download model
    print("\n[STEP 4] Download model?")
    resp = input("  Download YOLOv11n? (y/n): ").strip().lower()
    if resp == 'y':
        download_model()

    # Summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("ALL AVAILABLE COMMANDS (via unified CLI: python main.py <cmd>)")
    print("-" * 60)

    print("\n[TRAINING]")
    print("  python main.py train --quick              # Quick training (10 epochs)")
    print("  python main.py train                      # Full training (50 epochs)")
    print("  python main.py train --check              # Check system resources")
    print("  python main.py realtime --source 0        # Real-time detection (--no-show utk headless)")

    print("\n[DETECTION]")
    print("  python main.py detect --source image.jpg  # Detect single image")
    print("  python main.py detect --source data/raw   # Batch detection")
    print("  python main.py pipeline --source 0 --show # Detect+track+count")
    print("  python main.py gui                        # Launch GUI application")

    print("\n[EVALUATION]")
    print("  python main.py evaluate --task all        # Full evaluation")
    print("  python main.py evaluate --task fps        # FPS benchmark only")
    print("  python main.py compare --help             # Compare reports")

    print("\n[DATASET]")
    print("  python main.py extract --action list              # List videos")
    print("  python main.py extract --action extract-all       # Extract all videos")
    print("  python main.py workflow --list                    # Rencana alur dataset")
    print("  python main.py workflow                           # stage+annotate+validate+yaml")
    print("  python main.py workflow --rebuild                 # + split --clean (hati-hati)")
    print("  python main.py dataset --action validate          # Validate dataset")
    print("  python main.py dataset --action split-report      # Laporan leakage split")
    print("  python main.py annotate --image-dir ... --label-dir ...")

    print("\n[EXPORT]")
    print("  python main.py export --help              # ONNX/TFLite/TorchScript/dll")

    print("\n[HELP]")
    print("  python main.py --help                     # Daftar semua command")
    print("  (cara lama python src/<script>.py tetap didukung)")

    print("\n" + "-" * 60)
    print("TIPS FOR THIS LAPTOP")
    print("-" * 60)
    print("  1. Use YOLOv11n (default, fastest)")
    print("  2. Image size 416 (not 640)")
    print("  3. Batch size 4 (saves RAM)")
    print("  4. Close other apps during training")
    print("  5. Use --quick for testing")
    print("  6. Training ~25-30 min for 50 epochs")
    print("  7. Webcam runs at ~5-8 FPS")
    print("  8. Augmentasi optimized (erasing OFF)")


if __name__ == "__main__":
    main()
