"""
Pipeline Deteksi Kendaraan + Pemfilteran ROI + Pelacakan Objek
Dioptimalkan untuk CPU (Intel i3-1115G4, 8GB RAM)
"""

import os
import sys
import cv2
import yaml
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from utils.roi_filter import ROIFilter, ROIConfig, ROIBoundary
from utils.tracker import ObjectTracker
from utils.counter import VehicleCounter


def load_config(config_path="config/config.yaml"):
    """Memuat file konfigurasi YAML dari path yang diberikan."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class VehicleDetectionPipeline:
    """
    Pipeline lengkap: Deteksi YOLOv11 + Filter ROI + Pelacak Objek + Penghitung.
    
    Alur kerja:
    1. Baca frame dari video/webcam
    2. Jalankan inferensi YOLOv11 untuk deteksi kendaraan
    3. Filter deteksi berdasarkan ROI (area jalan 20m)
    4. Lacak objek menggunakan ByteTrack-inspired tracker
    5. Hitung kendaraan yang melewati garis penghitung
    6. Gambar anotasi pada frame
    7. Tampilkan HUD (Heads-Up Display) dengan statistik
    """

    # Pemetaan ID kelas ke nama kelas
    CLASS_NAMES = {0: "motor", 1: "mobil", 2: "bus", 3: "truk"}
    
    # Warna bounding box per kelas (BGR)
    COLORS = {
        "motor": (255, 0, 0),    # Biru
        "mobil": (0, 255, 0),    # Hijau
        "bus": (0, 0, 255),      # Merah
        "truk": (255, 255, 0),   # Cyan
    }

    def __init__(self, config: dict, model_path: str = None):
        """
        Inisialisasi pipeline deteksi kendaraan.
        
        Args:
            config: dict konfigurasi dari config.yaml
            model_path: path ke model YOLOv11 .pt (opsional)
        """
        self.config = config
        model_cfg = config["model"]
        roi_cfg = config.get("roi", {})
        track_cfg = config.get("tracking", {})
        count_cfg = config.get("counting", {})

        # Memuat model YOLOv11
        if model_path is None:
            model_path = "D:/KULIAH/Tugas Akhir/tugasakhir/runs/detect/models/vehicle_detection/weights/best.pt"
            if not os.path.exists(model_path):
                model_path = model_cfg["architecture"]

        print(f"[INFO] Memuat model: {model_path}")
        self.model = YOLO(model_path)

        # Inisialisasi Filter ROI
        boundary_cfg = roi_cfg.get("boundary", {})
        boundary = ROIBoundary(
            top_left=tuple(boundary_cfg.get("top_left", [60, 497])),
            top_right=tuple(boundary_cfg.get("top_right", [390, 484])),
            bottom_left=tuple(boundary_cfg.get("bottom_left", [111, 1007])),
            bottom_right=tuple(boundary_cfg.get("bottom_right", [979, 822])),
        )
        self.roi_config = ROIConfig(
            enabled=roi_cfg.get("enabled", True),
            boundary=boundary,
            min_bbox_height=roi_cfg.get("min_bbox_height", 20),
        )
        self.roi_filter = ROIFilter(self.roi_config)
        self.show_roi = roi_cfg.get("draw_roi", True)

        # Inisialisasi Pelacak Objek
        self.tracker = ObjectTracker(
            max_age=track_cfg.get("max_age", 30),
            min_hits=track_cfg.get("min_hits", 3),
            max_distance=track_cfg.get("max_distance", 80.0),
            track_buffer=track_cfg.get("track_buffer", 50),
        )
        self.show_track_id = track_cfg.get("show_track_id", True)
        self.show_velocity = track_cfg.get("show_velocity", False)

        # Inisialisasi Penghitung Kendaraan (dual-line)
        self.counter = VehicleCounter(
            line1_position=count_cfg.get("line1_position", 0.35),
            line2_position=count_cfg.get("line2_position", 0.65),
            direction=count_cfg.get("direction", "both"),
            min_track_length=count_cfg.get("min_track_length", 5),
            max_lost_frames=count_cfg.get("max_lost_frames", 30),
        )

        # Pelacakan FPS
        self.fps_history = []
        self.total_detections = 0
        self.total_filtered = 0

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Memproses satu frame gambar.
        
        Proses:
        1. Resize frame untuk inferensi
        2. Jalankan YOLOv11 detection
        3. Konversi koordinat kembali ke ukuran asli
        4. Filter deteksi menggunakan ROI
        5. Lacak objek menggunakan tracker
        6. Perbarui penghitung kendaraan
        7. Gambar anotasi pada frame
        
        Args:
            frame: frame gambar asli (BGR)
            
        Returns:
            Tuple (frame_labeled, stats) dimana:
            - frame_labeled: frame dengan anotasi
            - stats: dict statistik (fps, jumlah deteksi, dll)
        """
        start = time.time()
        input_size = self.config["model"]["input_size"]
        conf_thresh = self.config["model"]["confidence_threshold"]
        iou_thresh = self.config["model"]["iou_threshold"]

        # Resize frame untuk inferensi
        frame_resized = cv2.resize(frame, (input_size, input_size))

        # Jalankan inferensi YOLOv11
        results = self.model(
            frame_resized,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
        )

        # Hitung faktor skala untuk konversi koordinat
        scale_x = frame.shape[1] / input_size
        scale_y = frame.shape[0] / input_size

        # Parsing hasil deteksi
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                # Hanya proses kelas yang dikenal
                if cls_id not in self.CLASS_NAMES:
                    continue
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                # Konversi koordinat kembali ke ukuran asli frame
                xyxy[0] *= scale_x
                xyxy[1] *= scale_y
                xyxy[2] *= scale_x
                xyxy[3] *= scale_y

                detections.append({
                    "class_id": cls_id,
                    "class_name": self.CLASS_NAMES[cls_id],
                    "confidence": conf,
                    "bbox": xyxy,
                })

        self.total_detections += len(detections)

        # Filter deteksi menggunakan ROI
        filtered = self.roi_filter.filter_detections(detections, frame.shape)
        self.total_filtered += len(filtered)

        # Lacak objek menggunakan tracker
        tracked = self.tracker.update(filtered)

        # Perbarui penghitung (untuk persilangan garis)
        counter_input = [
            {
                "class_id": t["class_id"],
                "class_name": t["class_name"],
                "confidence": t["confidence"],
                "bbox": t["bbox"],
            }
            for t in tracked
        ]
        self.counter.update(counter_input, frame.shape[0])

        # Hitung FPS
        inference_time = time.time() - start
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history[-30:])

        # Gambar hasil pada frame
        result_frame = frame.copy()

        # Gambar ROI
        if self.show_roi:
            result_frame = self.roi_filter.draw_roi(result_frame)

        # Gambar 2 garis penghitung (dual-line)
        h = frame.shape[0]
        w = frame.shape[1]
        
        # Garis 1 (atas) - biru
        line1_y = int(h * self.counter.line1_position)
        cv2.line(result_frame, (0, line1_y), (w, line1_y), (255, 100, 0), 2)
        cv2.putText(result_frame, "GARIS 1", (10, line1_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
        
        # Garis 2 (bawah) - kuning
        line2_y = int(h * self.counter.line2_position)
        cv2.line(result_frame, (0, line2_y), (w, line2_y), (0, 255, 255), 2)
        cv2.putText(result_frame, "GARIS 2", (10, line2_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Label area
        cv2.putText(result_frame, "ZONE COUNTING", (w//2 - 60, line1_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Gambar objek yang dilacak
        for t in tracked:
            x1, y1, x2, y2 = [int(c) for c in t["bbox"]]
            cls_name = t["class_name"]
            color = self.COLORS.get(cls_name, (0, 255, 0))

            # Gambar bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # Buat label
            track_id = t["track_id"]
            label_parts = [f"ID:{track_id} {cls_name}"]
            label_parts.append(f"{t['confidence']:.2f}")
            label = " ".join(label_parts)

            # Gambar label di atas bbox
            cv2.putText(result_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Gambar panah kecepatan (opsional)
            if self.show_velocity and "velocity" in t:
                vx, vy = t["velocity"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                end_x = int(cx + vx * 3)
                end_y = int(cy + vy * 3)
                cv2.arrowedLine(result_frame, (cx, cy), (end_x, end_y),
                              (255, 255, 255), 2)

        # Gambar HUD (Heads-Up Display)
        self._draw_hud(result_frame, tracked, avg_fps)

        stats = {
            "fps": avg_fps,
            "total_detections": len(detections),
            "after_roi": len(filtered),
            "tracked": len(tracked),
            "counted": self.counter.total_count,
        }

        return result_frame, stats

    def _draw_hud(self, frame, tracked, fps):
        """
        Menggambar Heads-Up Display (HUD) pada frame.
        
        HUD menampilkan:
        - FPS saat ini
        - Jumlah objek per kelas (motor, mobil, bus, truk)
        - Total kendaraan yang melewati garis penghitung
        - Jumlah track aktif
        
        Args:
            frame: frame yang akan digambar HUD
            tracked: daftar objek yang dilacak
            fps: FPS saat ini
        """
        h, w = frame.shape[:2]

        # Panel latar belakang HUD
        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 90), (100, 100, 100), 1)

        # Tampilkan FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Hitung jumlah objek per kelas
        counts = {}
        for t in tracked:
            name = t["class_name"]
            counts[name] = counts.get(name, 0) + 1

        # Tampilkan jumlah per kelas
        y_offset = 55
        for name in ["motor", "mobil", "bus", "truk"]:
            c = counts.get(name, 0)
            color = self.COLORS[name]
            cv2.putText(frame, f"{name}: {c}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 20

        # Tampilkan total kendaraan yang melewati garis
        cv2.putText(frame, f"Terhitung: {self.counter.total_count}",
                   (w - 200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 255, 255), 2)

        # Tampilkan jumlah track aktif
        cv2.putText(frame, f"Aktif: {len(tracked)}",
                   (w - 200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 1)

    def process_video(self, source, output_path=None, show=False, max_frames=None):
        """
        Memproses video file atau webcam.
        
        Args:
            source: path file video atau "0" untuk webcam
            output_path: path file video output (opsional)
            show: tampilkan preview window
            max_frames: jumlah frame maksimum yang diproses (opsional)
        """
        is_webcam = source in ["0", 0]
        cap = cv2.VideoCapture(0 if is_webcam else source)

        if not cap.isOpened():
            print(f"[ERROR] Tidak dapat membuka: {source}")
            return

        # Dapatkan properti video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

        print(f"\n[INFO] Sumber: {'Webcam' if is_webcam else source}")
        print(f"[INFO] Resolusi: {width}x{height}")
        print(f"[INFO] ROI: {'ON' if self.roi_config.enabled else 'OFF'}")
        print(f"[INFO] Pelacakan: {'ON' if self.tracker else 'OFF'}")
        print(f"[INFO] Tekan 'q' untuk keluar\n")

        # Inisialisasi VideoWriter jika perlu menyimpan output
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_src, (width, height))

        # Reset penghitung dan pelacak
        self.counter.reset()
        self.tracker.reset()
        self.fps_history = []
        frame_count = 0

        # Loop utama pemrosesan video
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if max_frames and frame_count > max_frames:
                break

            # Proses frame
            result_frame, stats = self.process_frame(frame)

            # Simpan frame ke video output
            if writer:
                writer.write(result_frame)

            # Tampilkan preview
            if show:
                cv2.imshow("Deteksi Kendaraan + ROI + Pelacakan", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Tampilkan progress setiap 30 frame
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count} | FPS: {stats['fps']:.1f} | "
                      f"Deteksi: {stats['total_detections']} -> ROI: {stats['after_roi']} | "
                      f"Dilacak: {stats['tracked']} | Terhitung: {stats['counted']}")

        # Ringkasan akhir
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        print(f"\n{'='*60}")
        print(f"PEMBERPROSESAN SELESAI")
        print(f"  Frame diproses:    {frame_count}")
        print(f"  FPS rata-rata:     {np.mean(self.fps_history):.1f}")
        print(f"  Total deteksi:     {self.total_detections}")
        print(f"  Setelah filter ROI: {self.total_filtered}")
        print(f"  Jumlah akhir:      {self.counter.total_count}")
        print(f"  Per kelas:         {dict(self.counter.class_counts)}")
        print(f"{'='*60}")

        return {
            "frames": frame_count,
            "fps": np.mean(self.fps_history),
            "total_detections": self.total_detections,
            "total_filtered": self.total_filtered,
            "count": self.counter.total_count,
            "by_class": dict(self.counter.class_counts),
        }


def main():
    """Fungsi utama untuk menjalankan pipeline deteksi kendaraan dari command line."""
    parser = argparse.ArgumentParser(
        description="Pipeline Deteksi Kendaraan + ROI + Pelacakan"
    )
    parser.add_argument("--source", type=str, default="0",
                       help="Path file video atau '0' untuk webcam")
    parser.add_argument("--model", type=str, default=None,
                       help="Path ke model YOLOv11 .pt")
    parser.add_argument("--output", type=str, default=None,
                       help="Path file video output")
    parser.add_argument("--show", action="store_true",
                       help="Tampilkan preview window")
    parser.add_argument("--conf", type=float, default=None,
                       help="Override ambang batas confidence")
    parser.add_argument("--roi-max-dist", type=float, default=None,
                       help="Override jarak maksimum ROI (meter)")
    parser.add_argument("--no-roi", action="store_true",
                       help="Nonaktifkan filter ROI")
    parser.add_argument("--no-track", action="store_true",
                       help="Nonaktifkan pelacakan objek")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Jumlah frame maksimum yang diproses")
    args = parser.parse_args()

    config = load_config()

    # Override konfigurasi dari command line
    if args.conf:
        config["model"]["confidence_threshold"] = args.conf
    if args.roi_max_dist:
        config["roi"]["max_distance_m"] = args.roi_max_dist
    if args.no_roi:
        config["roi"]["enabled"] = False
    if args.no_track:
        config["tracking"]["enabled"] = False

    pipeline = VehicleDetectionPipeline(config, model_path=args.model)
    pipeline.process_video(
        source=args.source,
        output_path=args.output,
        show=args.show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
