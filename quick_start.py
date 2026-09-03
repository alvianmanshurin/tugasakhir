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
    print(f"  Model: YOLOv11n (Nano - 2.6M params)")
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
        "models", "outputs/detections", "outputs/evaluation", "docs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[OK] Directories created")


def download_model():
    """Download YOLOv11n model."""
    print("\n[STEP 3] Downloading YOLOv11n model...")
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

    check_system()

    print("\n[STEP 1] Install packages?")
    print("  (Skip if already installed)")
    resp = input("  Install? (y/n): ").strip().lower()

    if resp == 'y':
        install_requirements()

    print("\n[STEP 2] Creating directories...")
    create_directories()

    print("\n[STEP 3] Download model?")
    resp = input("  Download YOLOv11n? (y/n): ").strip().lower()
    if resp == 'y':
        download_model()

    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)

    print("\n" + "-" * 60)
    print("ALL AVAILABLE COMMANDS")
    print("-" * 60)

    print("\n[TRAINING]")
    print("  python src/train.py --quick              # Quick training (10 epochs)")
    print("  python src/train.py                      # Full training (50 epochs)")
    print("  python src/train.py --check              # Check system resources")
    print("  python src/monitor.py --action watch     # Monitor training progress")

    print("\n[DETECTION]")
    print("  python src/detect.py --source image.jpg  # Detect single image")
    print("  python src/detect.py --source data/raw   # Batch detection")
    print("  python src/realtime.py --source 0 --show # Webcam real-time")

    print("\n[EVALUATION]")
    print("  python src/evaluate.py                   # Full evaluation")
    print("  python src/evaluate.py --task fps        # FPS benchmark only")
    print("  python src/comparison.py --action compare --image-dir data/raw --manual-json manual_counts.json")

    print("\n[DATASET - VIDEO]")
    print("  python src/extract_frames.py --action list              # List videos")
    print("  python src/extract_frames.py --action extract-all      # Extract all videos")
    print("  python src/extract_frames.py --action extract --video path/to/video.MOV")
    print("  python src/extract_frames.py --action split            # Split to train/val")

    print("\n[DATASET]")
    print("  python src/dataset_prepare.py --action validate  # Validate dataset")
    print("  python src/dataset_prepare.py --action split --source data/raw")
    print("  python src/dataset_collect.py --action guide     # Collection guide")
    print("  python src/dataset_collect.py --action webcam    # Capture from webcam")

    print("\n[ANNOTATION]")
    print("  python setup_labelimg.py --action install  # Install LabelImg")
    print("  python setup_labelimg.py --action launch   # Launch LabelImg")
    print("  python setup_labelimg.py --action guide    # Annotation guide")

    print("\n[GUI]")
    print("  python src/gui_app.py                    # Launch GUI application")

    print("\n[PIPELINE]")
    print("  python src/pipeline.py --action status      # Check project status")
    print("  python src/pipeline.py --action full        # Run full pipeline")
    print("  python src/pipeline.py --action from-train  # Train after annotation")

    print("\n[EXPORT]")
    print("  python src/export_model.py --action export --format onnx tflite")

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


if __name__ == "__main__":
    main()
