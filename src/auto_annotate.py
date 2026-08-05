"""
Script auto-annotation menggunakan YOLOv8 pretrained (bobot COCO)
Pemetaan kelas COCO ke kelas proyek:
  COCO: car(2), motorcycle(3), bus(5), truck(7)
  Proyek: motor(0), mobil(1), bus(2), truk(3)
"""

import os
import cv2
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO

# Pemetaan ID kelas COCO -> ID kelas proyek
COCO_TO_PROJECT = {
    3: 0,   # motorcycle -> motor
    2: 1,   # car -> mobil
    5: 2,   # bus -> bus
    7: 3,   # truck -> truk
}

# Nama kelas proyek berdasarkan ID
PROJECT_NAMES = {0: "motor", 1: "mobil", 2: "bus", 3: "truk"}


def load_config(config_path="config/config.yaml"):
    """Memuat file konfigurasi YAML dari path yang diberikan."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def auto_annotate(
    image_dir,
    label_dir,
    model_name="yolov8n.pt",
    conf_threshold=0.35,
    iou_threshold=0.45,
    img_size=640,
):
    """
    Fungsi utama untuk auto-annotation gambar menggunakan YOLOv8 pretrained.
    
    Proses:
    1. Memuat semua gambar dari image_dir
    2. Menjalankan inferensi YOLOv8 pada setiap gambar
    3. Memetakan hasil deteksi COCO ke kelas proyek (motor/mobil/bus/truk)
    4. Mengkonversi koordinat ke format YOLO (normalized)
    5. Menyimpan label .txt dalam format YOLO
    
    Args:
        image_dir: direktori berisi gambar yang akan di-annotate
        label_dir: direktori output untuk menyimpan file label .txt
        model_name: nama model YOLOv8 (default: yolov8n.pt)
        conf_threshold: ambang batas confidence (0-1)
        iou_threshold: ambang batas IoU untuk NMS
        img_size: ukuran gambar untuk inferensi
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Mengumpulkan semua file gambar (jpg, jpeg, png)
    image_files = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )

    if not image_files:
        print(f"[ERROR] Tidak ada gambar ditemukan di {image_dir}")
        return

    print(f"[INFO] Ditemukan {len(image_files)} gambar di {image_dir}")
    print(f"[INFO] Memuat model: {model_name}")

    # Memuat model YOLOv8
    model = YOLO(model_name)

    total_boxes = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    skipped = 0

    # Proses setiap gambar
    for i, img_path in enumerate(image_files):
        # Menjalankan inferensi pada gambar
        results = model.predict(
            source=str(img_path),
            imgsz=img_size,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        label_lines = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            img_h, img_w = r.orig_shape

            for box in r.boxes:
                coco_cls = int(box.cls[0])
                # Hanya proses kelas yang ada di proyek
                if coco_cls not in COCO_TO_PROJECT:
                    continue

                proj_cls = COCO_TO_PROJECT[coco_cls]

                # Mengkonversi koordinat bbox ke format YOLO (normalized 0-1)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                # Memastikan nilai dalam rentang 0-1
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                # Menambahkan baris label dalam format YOLO
                label_lines.append(f"{proj_cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                class_counts[proj_cls] += 1
                total_boxes += 1

        # Menyimpan file label .txt
        lbl_file = label_dir / (img_path.stem + ".txt")
        with open(lbl_file, "w") as f:
            f.write("\n".join(label_lines))

        if not label_lines:
            skipped += 1

        # Menampilkan progress setiap 50 gambar
        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"  [{i+1}/{len(image_files)}] diproses... ({total_boxes} box sejauh ini)")

    # Menampilkan ringkasan hasil
    print("\n" + "=" * 60)
    print("AUTO-ANNOTATION SELESAI")
    print("=" * 60)
    print(f"  Total gambar:  {len(image_files)}")
    print(f"  Total box:     {total_boxes}")
    print(f"  Gambar kosong: {skipped}")
    print(f"\n  Rincian per kelas:")
    for cls_id, name in PROJECT_NAMES.items():
        print(f"    {cls_id} ({name}): {class_counts[cls_id]}")
    print(f"\n  Label tersimpan di: {label_dir}")


def main():
    """Fungsi utama untuk menjalankan script auto-annotation dari command line."""
    parser = argparse.ArgumentParser(description="Auto-annotate gambar dengan YOLOv8 pretrained")
    parser.add_argument("--image-dir", type=str, required=True, help="Direktori gambar")
    parser.add_argument("--label-dir", type=str, required=True, help="Direktori output label")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model YOLOv8 (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Ambang batas confidence (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.45, help="Ambang batas IoU (default: 0.45)")
    parser.add_argument("--img-size", type=int, default=640, help="Ukuran gambar inferensi (default: 640)")
    args = parser.parse_args()

    auto_annotate(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
