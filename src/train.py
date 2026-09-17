"""
Script Training - DIOPTIMALKAN UNTUK LAPTOP LOW-END (CPU)
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
    """Memuat file konfigurasi YAML dari path yang diberikan."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_system_resources():
    """
    Mengecek sumber daya sistem yang tersedia sebelum training.
    
    Informasi yang dicek:
    - Jumlah core CPU (fisik dan logis)
    - Frekuensi CPU
    - Total RAM dan yang tersedia
    - Rekomendasi batch size berdasarkan RAM
    
    Returns:
        Dict berisi cpu_count, ram_gb, ram_available
    """
    print("\n" + "=" * 60)
    print("CEK SUMBER DAYA SISTEM")
    print("=" * 60)

    # Info CPU
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"\n[CPU]")
    print(f"  Core fisik: {psutil.cpu_count(logical=False)}")
    print(f"  Core logis: {cpu_count}")
    print(f"  Frekuensi maks: {cpu_freq.max:.0f} MHz" if cpu_freq else "  Frekuensi maks: N/A")

    # Info RAM
    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    ram_available = ram.available / (1024**3)
    print(f"\n[RAM]")
    print(f"  Total: {ram_gb:.1f} GB")
    print(f"  Tersedia: {ram_available:.1f} GB")
    print(f"  Terpakai: {ram.percent}%")

    # Rekomendasi
    print(f"\n[REKOMENDASI]")
    if ram_gb < 8:
        print("  PERINGATAN: RAM < 8GB, gunakan batch_size=2")
    elif ram_gb < 16:
        print("  OK: 8GB RAM, gunakan batch_size=4")
    else:
        print("  OK: RAM cukup")

    print("=" * 60)

    return {
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "ram_available": ram_available,
    }


def train_model(config):
    """
    Melatih model YOLOv11n yang dioptimalkan untuk CPU.
    
    Proses training:
    1. Cek sumber daya sistem
    2. Sesuaikan batch size berdasarkan RAM
    3. Muat model YOLOv11n (Nano)
    4. Jalankan training dengan parameter dari config
    5. Simpan model best.pt dan last.pt
    
    Args:
        config: dict konfigurasi dari config.yaml
        
    Returns:
        Hasil training dari ultralytics
    """
    cfg = config["training"]
    model_cfg = config["model"]

    print("\n" + "=" * 60)
    print("TRAINING MODEL DETEKSI KENDARAAN")
    print("DIOPTIMALKAN UNTUK CPU - Intel i3-1115G4")
    print("=" * 60)

    # Cek sumber daya
    resources = check_system_resources()

    # Sesuaikan batch size berdasarkan RAM yang tersedia
    recommended_batch = 4
    if resources["ram_gb"] < 6:
        recommended_batch = 2
        print("\n[PERINGATAN] RAM rendah, mengurangi batch_size ke 2")
    elif resources["ram_gb"] >= 16:
        recommended_batch = 8
        print("\n[INFO] RAM tinggi, meningkatkan batch_size ke 8")

    # Muat model
    model_name = model_cfg["architecture"]
    print(f"\n[INFO] Model: {model_name} (Nano - dioptimalkan untuk CPU)")
    print(f"[INFO] Ukuran gambar: {cfg['image_size']}x{cfg['image_size']}")
    print(f"[INFO] Batch size: {recommended_batch}")
    print(f"[INFO] Epochs: {cfg['epochs']}")
    print(f"[INFO] Device: CPU (tanpa CUDA)")

    # Estimasi waktu training
    est_time_per_epoch = 120  # estimasi kasar untuk CPU dengan 416x416
    total_est = est_time_per_epoch * cfg['epochs'] / 60
    print(f"[INFO] Estimasi waktu training: ~{total_est:.0f} menit")

    # Muat augmentasi dari config
    aug_cfg = config.get("augmentation", {})
    print(f"\n[AUGMENTASI]")
    print(f"  Flip horizontal: {aug_cfg.get('fliplr', 0.5)}")
    print(f"  Mosaic: {aug_cfg.get('mosaic', 1.0)}")
    print(f"  HSV Brightness: {aug_cfg.get('hsv_v', 0.4)}")
    print(f"  Shear: {aug_cfg.get('shear', 5.0)}")
    print(f"  Blur: {aug_cfg.get('blur', 0.01)}")
    print(f"  Grayscale: {aug_cfg.get('grayscale', 0.1)}")
    print(f"  Erasing: {aug_cfg.get('erasing', 0.0)} ← DIMATIKAN untuk counting")

    # Mulai training
    print("\n[INFO] Memulai training...")
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
        # === AUGMENTASI PARAMETERS ===
        fliplr=aug_cfg.get("fliplr", 0.5),
        flipud=aug_cfg.get("flipud", 0.0),
        mosaic=aug_cfg.get("mosaic", 1.0),
        hsv_h=aug_cfg.get("hsv_h", 0.015),
        hsv_s=aug_cfg.get("hsv_s", 0.7),
        hsv_v=aug_cfg.get("hsv_v", 0.4),
        degrees=aug_cfg.get("degrees", 0.0),
        translate=aug_cfg.get("translate", 0.1),
        scale=aug_cfg.get("scale", 0.5),
        shear=aug_cfg.get("shear", 5.0),
        perspective=aug_cfg.get("perspective", 0.001),
        blur=aug_cfg.get("blur", 0.01),
        erasing=aug_cfg.get("erasing", 0.0),
        grayscale=aug_cfg.get("grayscale", 0.1),
        mixup=aug_cfg.get("mixup", 0.0),
        copy_paste=aug_cfg.get("copy_paste", 0.0),
        crop_fraction=aug_cfg.get("crop_fraction", 1.0),
    )

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    print(f"\n[INFO] Training selesai!")
    print(f"[INFO] Total waktu: {elapsed_min:.1f} menit")
    print(f"[INFO] Model terbaik: models/vehicle_detection/weights/best.pt")
    print(f"[INFO] Model terakhir: models/vehicle_detection/weights/last.pt")

    return results


def quick_train(config):
    """
    Training cepat dengan epoch minimal untuk pengujian.
    Menggunakan augmentasi yang sama dengan training penuh.
    
    Args:
        config: dict konfigurasi
        
    Returns:
        Hasil training
    """
    print("\n[INFO] MODE TRAINING CEPAT (10 epochs)")
    print("[INFO] Augmentasi: Menggunakan konfigurasi yang sama dengan training penuh")
    cfg = config["training"]
    cfg["epochs"] = 10
    cfg["patience"] = 5
    return train_model(config)


def resume_training(config, last_model):
    """
    Melanjutkan training dari checkpoint terakhir.
    Menggunakan augmentasi yang sama dengan training penuh.
    
    Args:
        config: dict konfigurasi
        last_model: path ke model last.pt
        
    Returns:
        Hasil training
    """
    print(f"\n[INFO] Melanjutkan training dari: {last_model}")
    print("[INFO] Augmentasi: Menggunakan konfigurasi yang sama dengan training penuh")

    # Muat augmentasi dari config
    aug_cfg = config.get("augmentation", {})
    cfg = config["training"]

    model = YOLO(last_model)
    results = model.train(
        data=config["dataset"]["yaml_path"],
        epochs=cfg["epochs"],
        imgsz=cfg["image_size"],
        batch=4,
        device="cpu",
        amp=False,
        project="models",
        name="vehicle_detection",
        exist_ok=True,
        # === AUGMENTASI PARAMETERS (sama dengan train_model) ===
        fliplr=aug_cfg.get("fliplr", 0.5),
        flipud=aug_cfg.get("flipud", 0.0),
        mosaic=aug_cfg.get("mosaic", 1.0),
        hsv_h=aug_cfg.get("hsv_h", 0.015),
        hsv_s=aug_cfg.get("hsv_s", 0.7),
        hsv_v=aug_cfg.get("hsv_v", 0.4),
        degrees=aug_cfg.get("degrees", 0.0),
        translate=aug_cfg.get("translate", 0.1),
        scale=aug_cfg.get("scale", 0.5),
        shear=aug_cfg.get("shear", 5.0),
        perspective=aug_cfg.get("perspective", 0.001),
        blur=aug_cfg.get("blur", 0.01),
        erasing=aug_cfg.get("erasing", 0.0),
        grayscale=aug_cfg.get("grayscale", 0.1),
        mixup=aug_cfg.get("mixup", 0.0),
        copy_paste=aug_cfg.get("copy_paste", 0.0),
        crop_fraction=aug_cfg.get("crop_fraction", 1.0),
    )
    return results


def main():
    """Fungsi utama untuk menjalankan script training dari command line."""
    parser = argparse.ArgumentParser(
        description="Training YOLOv11 untuk Deteksi Kendaraan (Dioptimalkan untuk CPU)"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path file konfigurasi")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Jumlah epoch")
    parser.add_argument("--batch", type=int, default=None,
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None,
                       help="Ukuran gambar")
    parser.add_argument("--quick", action="store_true",
                       help="Training cepat (10 epochs)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Lanjutkan dari checkpoint")
    parser.add_argument("--check", action="store_true",
                       help="Cek sumber daya sistem saja")
    args = parser.parse_args()

    if args.check:
        check_system_resources()
        return

    config = load_config(args.config)

    # Override konfigurasi dari command line
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
