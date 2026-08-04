"""Training Script - OPTIMIZED FOR LOW-END LAPTOP (CPU)
Hardware: Intel i3-1115G4, 8GB RAM, Intel UHD Graphics
"""

import os
import yaml
import argparse
import time
import psutil
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_system_resources():
    """Check available system resources before training."""
    print("\n" + "=" * 60)
    print("SYSTEM RESOURCE CHECK")
    print("=" * 60)

    # CPU info
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"\n[CPU]")
    print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"  Logical cores: {cpu_count}")
    print(f"  Max frequency: {cpu_freq.max:.0f} MHz" if cpu_freq else "  Max frequency: N/A")

    # RAM info
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    ram_available = ram.available / (1024**3)
    print(f"\n[RAM]")
    print(f"  Total: {ram_gb:.1f} GB")
    print(f"  Available: {ram_available:.1f} GB")
    print(f"  Used: {ram.percent}%")

    # Recommendation
    print(f"\n[RECOMMENDATION]")
    if ram_gb < 8:
        print("  WARNING: RAM < 8GB, use batch_size=2")
    elif ram_gb < 16:
        print("  OK: 8GB RAM, use batch_size=4")
    else:
        print("  OK: Sufficient RAM")

    print("=" * 60)

    return {
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "ram_available": ram_available,
    }


def train_model(config):
    """Train YOLOv8n model optimized for CPU."""
    cfg = config["training"]
    model_cfg = config["model"]

    print("\n" + "=" * 60)
    print("VEHICLE DETECTION MODEL TRAINING")
    print("OPTIMIZED FOR CPU - Intel i3-1115G4")
    print("=" * 60)

    # Check resources
    resources = check_system_resources()

    # Adjust batch size based on available RAM
    recommended_batch = 4
    if resources["ram_gb"] < 6:
        recommended_batch = 2
        print("\n[WARNING] Low RAM detected, reducing batch_size to 2")
    elif resources["ram_gb"] >= 16:
        recommended_batch = 8
        print("\n[INFO] High RAM detected, increasing batch_size to 8")

    # Load model
    model_name = model_cfg["architecture"]
    print(f"\n[INFO] Model: {model_name} (Nano - optimized for CPU)")
    print(f"[INFO] Image size: {cfg['image_size']}x{cfg['image_size']}")
    print(f"[INFO] Batch size: {recommended_batch}")
    print(f"[INFO] Epochs: {cfg['epochs']}")
    print(f"[INFO] Device: CPU (no CUDA)")

    # Estimated training time
    est_time_per_epoch = 120  # rough estimate for CPU with 416x416
    total_est = est_time_per_epoch * cfg['epochs'] / 60
    print(f"[INFO] Estimated training time: ~{total_est:.0f} minutes")

    # Start training
    print("\n[INFO] Starting training...")
    start_time = time.time()

    model = YOLO(model_name)

    results = model.train(
        data=config["dataset"]["yaml_path"],
        epochs=cfg["epochs"],
        imgsz=cfg["image_size"],
        batch=recommended_batch,
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],
        warmup_momentum=cfg["warmup_momentum"],
        warmup_bias_lr=cfg["warmup_bias_lr"],
        close_mosaic=cfg["close_mosaic"],
        patience=cfg["patience"],
        save_period=cfg["save_period"],
        workers=cfg["workers"],
        optimizer=cfg["optimizer"],
        device="cpu",
        amp=False,
        cache=False,
        verbose=True,
        project="models",
        name="vehicle_detection",
        exist_ok=True,
    )

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    print(f"\n[INFO] Training completed!")
    print(f"[INFO] Total time: {elapsed_min:.1f} minutes")
    print(f"[INFO] Best model: models/vehicle_detection/weights/best.pt")
    print(f"[INFO] Last model: models/vehicle_detection/weights/last.pt")

    return results


def quick_train(config):
    """Quick training with minimal epochs for testing."""
    print("\n[INFO] QUICK TRAIN MODE (10 epochs)")
    cfg = config["training"]
    cfg["epochs"] = 10
    cfg["patience"] = 5
    return train_model(config)


def resume_training(config, last_model):
    """Resume training from last checkpoint."""
    print(f"\n[INFO] Resuming training from: {last_model}")

    model = YOLO(last_model)
    results = model.train(
        data=config["dataset"]["yaml_path"],
        epochs=config["training"]["epochs"],
        imgsz=config["training"]["image_size"],
        batch=4,
        device="cpu",
        amp=False,
        project="models",
        name="vehicle_detection",
        exist_ok=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for Vehicle Detection (CPU Optimized)"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Quick training (10 epochs)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--check", action="store_true", help="Check system resources only")
    args = parser.parse_args()

    if args.check:
        check_system_resources()
        return

    config = load_config(args.config)

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch:
        config["training"]["batch_size"] = args.batch
    if args.imgsz:
        config["training"]["image_size"] = args.imgsz

    if args.resume:
        resume_training(config, args.resume)
    elif args.quick:
        quick_train(config)
    else:
        train_model(config)


if __name__ == "__main__":
    main()
