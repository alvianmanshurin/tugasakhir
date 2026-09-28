"""
Script Training - DIOPTIMALKAN UNTUK LAPTOP LOW-END (CPU)
Hardware: Intel i3-1115G4, 8GB RAM, Intel UHD Graphics
"""

import time
import psutil
from pathlib import Path
from ultralytics import YOLO

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.paths import load_config, ROOT  # noqa: E402


# Argumen augmentasi yang diteruskan ke ultralytics. Hanya key di sini yang
# valid untuk task deteksi - blur/grayscale/erasing tidak didukung (lihat
# config/config.yaml untuk penjelasan).
AUGMENT_KEYS = (
    "fliplr", "flipud", "mosaic",
    "hsv_h", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "shear", "perspective",
    "mixup", "copy_paste",
)


def check_system_resources():
    """
    Mengecek sumber daya sistem yang tersedia sebelum training.

    Returns:
        Dict berisi cpu_count, ram_gb, ram_available, recommended_batch
    """
    print("\n" + "=" * 60)
    print("CEK SUMBER DAYA SISTEM")
    print("=" * 60)

    cpu_count = psutil.cpu_count()
    print(f"\n[CPU]")
    print(f"  Core fisik: {psutil.cpu_count(logical=False)}")
    print(f"  Core logis: {cpu_count}")

    ram = psutil.virtual_memory()
    ram_gb = ram.total / (1024**3)
    print(f"\n[RAM]")
    print(f"  Total: {ram_gb:.1f} GB")
    print(f"  Tersedia: {ram.available / (1024**3):.1f} GB")
    print(f"  Terpakai: {ram.percent}%")

    recommended_batch = recommend_batch(ram_gb)
    print(f"\n[REKOMENDASI] batch_size={recommended_batch}")

    print("=" * 60)

    return {
        "cpu_count": cpu_count,
        "ram_gb": ram_gb,
        "ram_available": ram.available / (1024**3),
        "recommended_batch": recommended_batch,
    }


def recommend_batch(ram_gb: float) -> int:
    """Batch size menurut RAM. Satu fungsi dipakai check + training."""
    if ram_gb < 6:
        return 2
    if ram_gb < 16:
        return 4
    return 8


def build_train_kwargs(config, batch=None, epochs=None):
    """
    Susun kwargs untuk ultralytics train() dari config.

    Dipakai oleh train_model() dan resume_training() supaya keduanya
    tidak mungkin berbeda setting.
    """
    cfg = config["training"]
    aug_cfg = config.get("augmentation", {})
    model_cfg = config["model"]

    kwargs = {
        "data": config["dataset"]["yaml_path"],
        "epochs": epochs if epochs is not None else cfg["epochs"],
        "imgsz": cfg["image_size"],
        "batch": batch,
        "lr0": cfg["lr0"],
        "lrf": cfg["lrf"],
        "momentum": cfg["momentum"],
        "weight_decay": cfg["weight_decay"],
        "warmup_epochs": cfg["warmup_epochs"],
        "warmup_momentum": cfg["warmup_momentum"],
        "warmup_bias_lr": cfg["warmup_bias_lr"],
        "close_mosaic": cfg["close_mosaic"],
        "patience": cfg["patience"],
        "save_period": cfg["save_period"],
        "workers": cfg["workers"],
        "optimizer": cfg["optimizer"],
        "device": model_cfg.get("device", "cpu"),
        "amp": cfg.get("amp", False),
        "cache": cfg.get("cache", False),
        "verbose": cfg.get("verbose", True),
        "project": str(ROOT / "runs" / "detect" / "models"),
        "name": "vehicle_detection",
        "exist_ok": True,
    }

    for key in AUGMENT_KEYS:
        if key in aug_cfg:
            kwargs[key] = aug_cfg[key]

    return kwargs


def train_model(config, batch=None, epochs=None):
    """
    Melatih model YOLOv11n yang dioptimalkan untuk CPU.

    Args:
        config: dict konfigurasi dari config.yaml
        batch: override batch size (None = dari RAM)
        epochs: override jumlah epoch (None = dari config)

    Returns:
        Hasil training dari ultralytics
    """
    cfg = config["training"]
    model_cfg = config["model"]

    print("\n" + "=" * 60)
    print("TRAINING MODEL DETEKSI KENDARAAN")
    print("DIOPTIMALKAN UNTUK CPU - Intel i3-1115G4")
    print("=" * 60)

    resources = check_system_resources()
    if batch is None:
        batch = resources["recommended_batch"]

    epochs_final = epochs if epochs is not None else cfg["epochs"]

    print(f"\n[INFO] Model: {model_cfg['architecture']} (Nano - dioptimalkan untuk CPU)")
    print(f"[INFO] Ukuran gambar: {cfg['image_size']}x{cfg['image_size']}")
    print(f"[INFO] Batch size: {batch}")
    print(f"[INFO] Epochs: {epochs_final}")
    print(f"[INFO] Device: {model_cfg.get('device', 'cpu')}")

    total_est = 120 * epochs_final / 60
    print(f"[INFO] Estimasi waktu training: ~{total_est:.0f} menit")

    print("\n[AUGMENTASI]")
    for key in AUGMENT_KEYS:
        if key in config.get("augmentation", {}):
            print(f"  {key}: {config['augmentation'][key]}")

    print("\n[INFO] Memulai training...")
    start_time = time.time()

    # Arsitektur dasar (yolo11n.pt) dipakai sebagai starting point. Kalau file
    # ada di root project, pakai path absolut; kalau tidak (mis. dipanggil
    # lewat CWD lain), Ultralytics akan mencarinya sendiri lewat nama.
    architecture = model_cfg["architecture"]
    arch_path = Path(architecture)
    if not arch_path.is_absolute() and (ROOT / arch_path).is_file():
        architecture = str(ROOT / arch_path)
    print(f"[INFO] Arsitektur dasar: {architecture}")
    model = YOLO(architecture)
    results = model.train(**build_train_kwargs(config, batch=batch, epochs=epochs_final))

    elapsed_min = (time.time() - start_time) / 60

    print(f"\n[INFO] Training selesai!")
    print(f"[INFO] Total waktu: {elapsed_min:.1f} menit")
    print(f"[INFO] Model terbaik: {ROOT / 'runs/detect/models/vehicle_detection/weights/best.pt'}")
    print(f"[INFO] Model terakhir: {ROOT / 'runs/detect/models/vehicle_detection/weights/last.pt'}")

    return results


def quick_train(config):
    """Training cepat (10 epochs) untuk smoke test konfigurasi."""
    print("\n[INFO] MODE TRAINING CEPAT (10 epochs)")
    print("[INFO] Augmentasi: konfigurasi sama dengan training penuh")
    return train_model(config, epochs=10)


def resume_training(config, last_model):
    """
    Lanjutkan training dari checkpoint terakhir.

    Memakai kwargs yang sama persis dengan train_model() agar fase
    close_mosaic tidak ter-reset.
    """
    print(f"\n[INFO] Melanjutkan training dari: {last_model}")

    model = YOLO(last_model)
    kwargs = build_train_kwargs(config, batch=recommend_batch(psutil.virtual_memory().total / (1024**3)))
    kwargs["resume"] = True
    return model.train(**kwargs)


def main():
    """Fungsi utama untuk menjalankan script training dari command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Training YOLOv11 untuk Deteksi Kendaraan (Dioptimalkan untuk CPU)"
    )
    parser.add_argument("--config", type=str, default=None,
                       help="Path file konfigurasi (default: config/config.yaml)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Jumlah epoch")
    parser.add_argument("--batch", type=int, default=None,
                       help="Batch size")
    parser.add_argument("--check", action="store_true",
                       help="Cek sumber daya sistem saja")
    parser.add_argument("--quick", action="store_true",
                       help="Training cepat (10 epochs)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Lanjutkan dari checkpoint")
    args = parser.parse_args()

    if args.check:
        check_system_resources()
        return

    config = load_config(args.config)

    if args.resume:
        resume_training(config, args.resume)
    elif args.quick:
        quick_train(config)
    else:
        train_model(config, batch=args.batch, epochs=args.epochs)


if __name__ == "__main__":
    main()
