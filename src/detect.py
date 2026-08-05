"""
Script Deteksi - DIOPTIMALKAN UNTUK LAPTOP LOW-END (CPU)
Inferensi ringan untuk Intel i3-1115G4, 8GB RAM
"""

import os
import cv2
import yaml
import argparse
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path="config/config.yaml"):
    """Memuat file konfigurasi YAML dari path yang diberikan."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class LightweightDetector:
    """
    Detektor kendaraan ringan yang dioptimalkan untuk CPU.
    
    Fitur:
    - Deteksi single image atau batch directory
    - Gambar bounding box dan label pada hasil deteksi
    - Panel statistik jumlah kendaraan per kelas
    - Pengukuran FPS (Frames Per Second)
    """

    def __init__(self, config):
        """
        Inisialisasi detektor ringan.
        
        Args:
            config: dict konfigurasi dari config.yaml
        """
        self.config = config
        self.model_cfg = config["model"]
        self.det_cfg = config["detection"]
        self.class_names = config["dataset"]["names"]

        # Memuat model YOLOv8
        model_path = "models/vehicle_detection/weights/best.pt"
        if not os.path.exists(model_path):
            model_path = self.model_cfg["architecture"]
            print(f"[INFO] Menggunakan model pretrained: {model_path}")

        print(f"[INFO] Memuat model: {model_path}")
        self.model = YOLO(model_path)

        # Warna bounding box per kelas (BGR)
        self.colors = {
            "motor": (255, 0, 0),    # Biru
            "mobil": (0, 255, 0),    # Hijau
            "bus": (0, 0, 255),      # Merah
            "truk": (255, 255, 0),   # Cyan
        }

    def detect(self, source, save=True):
        """
        Mendeteksi kendaraan pada gambar atau direktori.
        
        Args:
            source: path gambar atau direktori
            save: simpan hasil deteksi ke file
            
        Returns:
            Hasil deteksi atau None jika gagal
        """
        source = Path(source)

        if source.is_dir():
            return self._detect_directory(source, save)
        elif source.is_file():
            return self._detect_single(source, save)
        else:
            print(f"[ERROR] Sumber tidak ditemukan: {source}")
            return None

    def _detect_single(self, image_path, save=True):
        """
        Mendeteksi kendaraan pada satu gambar.
        
        Proses:
        1. Baca gambar
        2. Jalankan inferensi YOLOv8
        3. Parse hasil deteksi
        4. Gambar bounding box dan label
        5. Simpan hasil jika diperlukan
        
        Args:
            image_path: path file gambar
            save: simpan hasil deteksi
            
        Returns:
            Dict berisi counts, total, dan time
        """
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[ERROR] Tidak dapat membaca: {image_path}")
            return None

        # Jalankan inferensi
        start = time.time()
        results = self.model(
            img,
            conf=self.model_cfg["confidence_threshold"],
            iou=self.model_cfg["iou_threshold"],
            imgsz=self.model_cfg["input_size"],
            verbose=False,
        )
        inference_time = time.time() - start

        # Parse hasil deteksi
        detections = []
        counts = {name: 0 for name in self.class_names.values()}

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    if cls_id < len(self.class_names):
                        cls_name = self.class_names[cls_id]
                        detections.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": xyxy,
                        })
                        counts[cls_name] += 1

        total = sum(counts.values())

        # Gambar hasil deteksi pada gambar
        img_result = self._draw_detections(img, detections, counts, total, inference_time)

        # Simpan hasil
        if save:
            output_dir = Path(self.det_cfg["output_dir"]) / "detections"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / f"{Path(image_path).stem}_detected.jpg"
            cv2.imwrite(str(save_path), img_result)
            print(f"[INFO] Tersimpan: {save_path}")

        # Tampilkan hasil
        print(f"\n[HASIL] {Path(image_path).name}")
        print(f"  Total: {total} kendaraan")
        for name, count in counts.items():
            if count > 0:
                print(f"  {name}: {count}")
        print(f"  Waktu: {inference_time*1000:.0f}ms | FPS: {1/inference_time:.1f}")

        return {"counts": counts, "total": total, "time": inference_time}

    def _detect_directory(self, dir_path, save=True):
        """
        Mendeteksi kendaraan pada semua gambar dalam direktori.
        
        Args:
            dir_path: path direktori berisi gambar
            save: simpan hasil deteksi
            
        Returns:
            Daftar hasil deteksi untuk setiap gambar
        """
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        images = []
        for ext in exts:
            images.extend(dir_path.glob(ext))

        if not images:
            print(f"[PERINGATAN] Tidak ada gambar ditemukan di: {dir_path}")
            return []

        print(f"\n[INFO] Memproses {len(images)} gambar...")

        results = []
        total_start = time.time()

        for idx, img_path in enumerate(sorted(images), 1):
            result = self._detect_single(img_path, save)
            if result:
                results.append(result)

            # Tampilkan progress
            if idx % 5 == 0:
                elapsed = time.time() - total_start
                fps = idx / elapsed if elapsed > 0 else 0
                print(f"  Progress: {idx}/{len(images)} | FPS: {fps:.1f}")

        # Ringkasan batch
        total_time = time.time() - total_start
        total_vehicles = sum(r["total"] for r in results)
        avg_fps = len(results) / total_time if total_time > 0 else 0

        print(f"\n{'='*50}")
        print(f"RINGKASAN BATCH")
        print(f"{'='*50}")
        print(f"  Gambar: {len(results)}")
        print(f"  Kendaraan: {total_vehicles}")
        print(f"  Waktu: {total_time:.1f}s")
        print(f"  FPS: {avg_fps:.1f}")

        return results

    def _draw_detections(self, img, detections, counts, total, inference_time):
        """
        Menggambar hasil deteksi pada gambar.
        
        Fitur yang digambar:
        - Bounding box dengan warna per kelas
        - Label nama kelas dan confidence
        - Panel jumlah kendaraan per kelas
        - FPS
        
        Args:
            img: gambar asli
            detections: daftar deteksi
            counts: jumlah per kelas
            total: jumlah total kendaraan
            inference_time: waktu inferensi
            
        Returns:
            Gambar dengan anotasi
        """
        result = img.copy()

        # Gambar bounding box dan label
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            color = self.colors.get(det["class_name"], (0, 255, 0))

            # Gambar bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Buat label dengan background
            label = f"{det['class_name']} {det['confidence']:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1-h-10), (x1+w, y1), color, -1)
            cv2.putText(result, label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Panel jumlah kendaraan
        y = 25
        cv2.putText(result, "JUMLAH KENDARAAN", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y += 25
        for name, count in counts.items():
            if count > 0:
                color = self.colors.get(name, (0,255,0))
                cv2.putText(result, f"{name}: {count}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 20
        cv2.putText(result, f"Total: {total}", (10, y+10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # Tampilkan FPS
        fps = 1/inference_time if inference_time > 0 else 0
        cv2.putText(result, f"FPS: {fps:.1f}", (10, result.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        return result


def main():
    """Fungsi utama untuk menjalankan script deteksi dari command line."""
    parser = argparse.ArgumentParser(
        description="Deteksi Kendaraan (Dioptimalkan untuk CPU)"
    )
    parser.add_argument("--source", type=str, required=True,
                       help="Path gambar atau direktori")
    parser.add_argument("--model", type=str, default=None,
                       help="Path model YOLOv8")
    parser.add_argument("--conf", type=float, default=None,
                       help="Ambang batas confidence")
    parser.add_argument("--no-save", action="store_true",
                       help="Jangan simpan hasil")
    args = parser.parse_args()

    config = load_config()

    # Override konfigurasi dari command line
    if args.model:
        config["model"]["architecture"] = args.model
    if args.conf:
        config["model"]["confidence_threshold"] = args.conf

    detector = LightweightDetector(config)
    detector.detect(args.source, save=not args.no_save)


if __name__ == "__main__":
    main()
