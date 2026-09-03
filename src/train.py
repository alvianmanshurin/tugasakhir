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
    
    Args:
        config: dict konfigurasi
        
    Returns:
        Hasil training
    """
    print("\n[INFO] MODE TRAINING CEPAT (10 epochs)")
    cfg = config["training"]
    cfg["epochs"] = 10
    cfg["patience"] = 5
    return train_model(config)


def resume_training(config, last_model):
    """
    Melanjutkan training dari checkpoint terakhir.
    
    Args:
        config: dict konfigurasi
        last_model: path ke model last.pt
        
    Returns:
        Hasil training
    """
    print(f"\n[INFO] Melanjutkan training dari: {last_model}")

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
