"""Dataset Collection Guide for Vehicle Detection
Panduan pengambilan gambar kendaraan di gerbang masuk Kampus ITERA
"""

import os
import cv2
import yaml
import argparse
import time
import json
from pathlib import Path
from datetime import datetime


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DatasetCollector:
    """Helper untuk pengumpulan dataset kendaraan."""

    def __init__(self, config):
        self.config = config
        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_from_webcam(self, output_dir=None, interval=2):
        """Capture gambar dari webcam secara berkala."""
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("DATASET CAPTURE FROM WEBCAM")
        print("=" * 60)
        print(f"\n[INFO] Output directory: {self.output_dir}")
        print(f"[INFO] Capture interval: {interval} seconds")
        print(f"\n[CONTROLS]")
        print(f"  SPACE  - Capture manual")
        print(f"  s      - Start/Stop auto capture")
        print(f"  q      - Quit")
        print(f"\n[INFO] Starting webcam...")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return

        auto_capture = False
        last_capture = time.time()
        count = len(list(self.output_dir.glob("*.jpg")))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Display
            display = frame.copy()
            status = "AUTO" if auto_capture else "MANUAL"
            cv2.putText(display, f"Mode: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Captured: {count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display, "SPACE: Capture | S: Auto | Q: Quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Dataset Capture", display)

            # Auto capture
            if auto_capture and (time.time() - last_capture) >= interval:
                filename = f"img_{count:05d}.jpg"
                cv2.imwrite(str(self.output_dir / filename), frame)
                count += 1
                last_capture = time.time()
                print(f"  Captured: {filename}")

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                filename = f"img_{count:05d}.jpg"
                cv2.imwrite(str(self.output_dir / filename), frame)
                count += 1
                print(f"  Captured: {filename}")
            elif key == ord('s'):
                auto_capture = not auto_capture
                print(f"  Auto capture: {'ON' if auto_capture else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[OK] Total images captured: {count}")

    def capture_from_video(self, video_path, output_dir=None, interval=30):
        """Extract frames dari video."""
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("EXTRACT FRAMES FROM VIDEO")
        print("=" * 60)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"\n[INFO] Video: {video_path}")
        print(f"[INFO] FPS: {fps:.1f}")
        print(f"[INFO] Total frames: {total_frames}")
        print(f"[INFO] Duration: {duration:.1f}s")
        print(f"[INFO] Extract every {interval} frames")

        frame_count = 0
        extracted = 0
        count = len(list(self.output_dir.glob("*.jpg")))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                filename = f"img_{count:05d}.jpg"
                cv2.imwrite(str(self.output_dir / filename), frame)
                count += 1
                extracted += 1

                if extracted % 10 == 0:
                    print(f"  Extracted: {extracted} frames")

            frame_count += 1

        cap.release()
        print(f"\n[OK] Extracted {extracted} frames to: {self.output_dir}")

    def create_split(self, source_dir, train_ratio=0.8):
        """Split dataset menjadi train/val."""
        import random

        source = Path(source_dir)
        images = list(source.glob("*.jpg")) + list(source.glob("*.png"))

        print(f"\n[SPLIT] Total images: {len(images)}")

        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)

        train_files = images[:split_idx]
        val_files = images[split_idx:]

        # Create directories
        train_img_dir = Path("data/annotated/images/train")
        val_img_dir = Path("data/annotated/images/val")
        train_lbl_dir = Path("data/annotated/labels/train")
        val_lbl_dir = Path("data/annotated/labels/val")

        for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Copy files
        import shutil
        for f in train_files:
            shutil.copy2(f, train_img_dir / f.name)
            # Create empty label
            (train_lbl_dir / (f.stem + ".txt")).touch()

        for f in val_files:
            shutil.copy2(f, val_img_dir / f.name)
            # Create empty label
            (val_lbl_dir / (f.stem + ".txt")).touch()

        print(f"[OK] Train: {len(train_files)} images")
        print(f"[OK] Val: {len(val_files)} images")

    def show_guidelines(self):
        """Tampilkan panduan pengambilan gambar."""
        print("\n" + "=" * 60)
        print("PANDUAN PENGAMBILAN DATASET")
        print("Gerbang Masuk Kampus ITERA")
        print("=" * 60)

        guidelines = """
1. KONDISI PENCAHAYAAN
   - Ambil gambar pada kondisi SIANG HARI (08.00 - 16.00)
   - Hindari kondisi backlight (matahari di belakang kamera)
   - Pastikan pencahayaan merata

2. SUDUT PENGAMBILAN
   - Sudut ideal: 30-45 derajat dari horizontal
   - Posisi kamera harus melihat gerbang masuk
   - Pastikan seluruh gerbang terlihat dalam frame

3. RESOLUSI MINIMAL
   - Minimal 640x640 pixels
   - Disarankan: 1280x720 atau 1920x1080
   - Format: JPG atau PNG

4. VARIASI KENDARAAN
   - Pastikan semua kelas terwakili:
     * Motor (berbagai jenis dan warna)
     * Mobil (berbagai jenis dan warna)
     * Bus (jika ada)
     * Truk (jika ada)
   - Minimal 100 gambar per kelas untuk hasil optimal

5. KONDISI LALU LINTAS
   - Variasi: sepi, ramai, sangat ramai
   - Berbagai waktu: pagi, siang, sore
   - Berbagai cuaca: cerah, mendung

6. YANG HARUS DIHINDARI
   - Gambar buram/blur
   - Gambar terlalu gelap/terang
   - Objek terpotong
   - Terlalu banyak objek non-kendaraan

7. JUMLAH MINIMAL
   - Total gambar: minimal 500 gambar
   - Per kelas: minimal 100 gambar
   - Train:Val = 80:20

8. LOKASI PENGAMBILAN
   - Gerbang masuk Kampus ITERA
   - Arah masuk dan keluar
   - Berbagai kondisi lalu lintas
"""

        print(guidelines)

        # Save as file
        guide_path = Path("data/DATASET_GUIDE.txt")
        with open(guide_path, "w") as f:
            f.write(guidelines)
        print(f"[OK] Panduan disimpan di: {guide_path}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Collection Tool")
    parser.add_argument("--action", type=str, required=True,
                       choices=["webcam", "video", "split", "guide"],
                       help="Action to perform")
    parser.add_argument("--source", type=str, help="Video source path")
    parser.add_argument("--output", type=str, default="data/raw",
                       help="Output directory")
    parser.add_argument("--interval", type=int, default=30,
                       help="Frame interval for video extraction")
    parser.add_argument("--ratio", type=float, default=0.8,
                       help="Train split ratio")
    args = parser.parse_args()

    config = load_config()
    collector = DatasetCollector(config)

    if args.action == "webcam":
        collector.capture_from_webcam(args.output)
    elif args.action == "video":
        collector.capture_from_video(args.source, args.output, args.interval)
    elif args.action == "split":
        collector.create_split(args.source, args.ratio)
    elif args.action == "guide":
        collector.show_guidelines()


if __name__ == "__main__":
    main()
