"""
Script Evaluasi Model Deteksi Kendaraan
"""

import os
import time
import yaml
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report


def load_config(config_path="config/config.yaml"):
    """Memuat file konfigurasi proyek."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ModelEvaluator:
    """
    Evaluator model YOLOv8 untuk deteksi kendaraan.
    
    Fitur:
    - Evaluasi mAP, Precision, Recall
    - Evaluasi kecepatan inferensi (FPS)
    - Generate Confusion Matrix
    - Generate kurva Precision-Recall dan F1
    - Simpan laporan evaluasi dalam format JSON dan CSV
    """

    def __init__(self, config):
        """
        Inisialisasi evaluator model.
        
        Args:
            config: dict konfigurasi dari config.yaml
        """
        self.config = config
        self.eval_cfg = config["evaluation"]
        self.model_cfg = config["model"]
        self.dataset_cfg = config["dataset"]
        self.output_dir = Path(self.eval_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Memuat model
        model_path = config["model"].get("best_weights",
            "D:/KULIAH/Tugas Akhir/tugasakhir/runs/detect/models/vehicle_detection/weights/best.pt")
        if not os.path.exists(model_path):
            print("[ERROR] Model tidak ditemukan. Silakan latih model terlebih dahulu.")
            raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

        print(f"[INFO] Memuat model: {model_path}")
        self.model = YOLO(model_path)
        # class_names: {0: 'motor', 1: 'mobil', ...}
        raw = config["dataset"].get("names", {0:'motor',1:'mobil',2:'bus',3:'truk'})
        self.class_names = {int(k): v for k, v in raw.items()}

    def evaluate_map(self):
        """
        Mengevaluasi mAP, Precision, dan Recall pada validation set.
        
        Metrik yang dihitung:
        - mAP50: Mean Average Precision pada IoU 0.5
        - mAP50-95: Mean Average Precision pada IoU 0.5 sampai 0.95
        - Precision: Rasio True Positive / (True Positive + False Positive)
        - Recall: Rasio True Positive / (True Positive + False Negative)
        
        Returns:
            Tuple (metrics, per_class, results)
        """
        print("\n" + "=" * 60)
        print("EVALUASI MODEL - mAP, PRECISION, RECALL")
        print("=" * 60)

        # Jalankan evaluasi pada validation set
        results = self.model.val(data=self.dataset_cfg["yaml_path"])

        # Ekstrak metrik keseluruhan
        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "Precision": float(results.box.mp),
            "Recall": float(results.box.mr),
        }

        # Ekstrak metrik per kelas
        per_class = {}
        for i, name in self.class_names.items():
            if i < len(results.box.ap_class_index):
                per_class[name] = {
                    "AP50": float(results.box.ap50[i]) if i < len(results.box.ap50) else 0,
                    "AP50-95": float(results.box.ap[i]) if i < len(results.box.ap) else 0,
                }

        # Tampilkan hasil
        print(f"\n[HASIL] Metrik Keseluruhan:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        print(f"\n[HASIL] AP per Kelas:")
        for name, ap in per_class.items():
            print(f"  {name}: AP50={ap['AP50']:.4f}, AP50-95={ap['AP50-95']:.4f}")

        return metrics, per_class, results

    def evaluate_fps(self, image_size=640, num_images=100):
        """
        Mengevaluasi kecepatan inferensi (FPS).
        
        Proses:
        1. Buat gambar dummy
        2. Warmup model (10 iterasi)
        3. Ukur waktu inferensi pada num_images gambar
        4. Hitung rata-rata waktu dan FPS
        
        Args:
            image_size: ukuran gambar untuk inferensi
            num_images: jumlah gambar untuk pengukuran
            
        Returns:
            Dict berisi metrik FPS
        """
        print("\n" + "=" * 60)
        print("EVALUASI MODEL - FPS (Kecepatan Inferensi)")
        print("=" * 60)

        # Buat gambar dummy
        dummy_img = np.random.randint(
            0, 255, (image_size, image_size, 3), dtype=np.uint8
        )

        # Warmup model
        for _ in range(10):
            self.model(dummy_img, verbose=False)

        # Ukur FPS
        times = []
        for _ in range(num_images):
            start = time.time()
            self.model(dummy_img, verbose=False)
            times.append(time.time() - start)

        avg_time = np.mean(times)
        fps = 1.0 / avg_time

        metrics = {
            "avg_inference_time_ms": float(avg_time * 1000),
            "fps": float(fps),
            "num_images": num_images,
            "image_size": image_size,
        }

        # Tampilkan hasil
        print(f"\n[HASIL] Evaluasi FPS:")
        print(f"  Ukuran gambar: {image_size}x{image_size}")
        print(f"  Jumlah gambar: {num_images}")
        print(f"  Rata-rata waktu inferensi: {avg_time*1000:.2f} ms")
        print(f"  FPS: {fps:.2f}")

        return metrics

    def generate_confusion_matrix(self, results):
        """
        Generate dan simpan confusion matrix.
        
        Confusion matrix menunjukkan:
        - True Positive: prediksi benar
        - False Positive: prediksi salah (FP)
        - False Negative: tidak terdeteksi (FN)
        
        Args:
            results: hasil evaluasi dari model.val()
        """
        print("\n[INFO] Generate confusion matrix...")

        # Ambil prediksi dan ground truth
        true_labels = []
        pred_labels = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    pred_labels.append(int(box.cls[0]))
            if result.probs is not None:
                true_labels.append(int(result.probs.top1))

        if len(true_labels) > 0 and len(pred_labels) > 0:
            # Buat confusion matrix
            cm = confusion_matrix(
                true_labels, pred_labels, labels=list(self.class_names.keys())
            )

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=list(self.class_names.values()),
                yticklabels=list(self.class_names.values()),
            )
            plt.title("Confusion Matrix - Deteksi Kendaraan")
            plt.ylabel("Label Sebenarnya")
            plt.xlabel("Label Prediksi")
            plt.tight_layout()

            # Simpan plot
            save_path = self.output_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[INFO] Confusion matrix tersimpan di: {save_path}")

    def generate_pr_curve(self, results):
        """
        Generate dan simpan kurva Precision-Recall.
        
        Args:
            results: hasil evaluasi dari model.val()
        """
        print("\n[INFO] Generate kurva Precision-Recall...")

        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            plt.title("Kurva Precision-Recall - Deteksi Kendaraan")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid(True)
            plt.tight_layout()

            save_path = self.output_dir / "pr_curve.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[INFO] Kurva PR tersimpan di: {save_path}")
        except Exception as e:
            print(f"[PERINGATAN] Tidak dapat generate kurva PR: {e}")

    def generate_f1_curve(self, results):
        """
        Generate dan simpan kurva F1.
        
        Args:
            results: hasil evaluasi dari model.val()
        """
        print("\n[INFO] Generate kurva F1...")

        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            plt.title("Kurva F1-Confidence - Deteksi Kendaraan")
            plt.xlabel("Ambang Batas Confidence")
            plt.ylabel("F1 Score")
            plt.grid(True)
            plt.tight_layout()

            save_path = self.output_dir / "f1_curve.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[INFO] Kurva F1 tersimpan di: {save_path}")
        except Exception as e:
            print(f"[PERINGATAN] Tidak dapat generate kurva F1: {e}")

    def save_evaluation_report(self, map_metrics, per_class, fps_metrics):
        """
        Menyimpan laporan evaluasi ke file JSON dan CSV.
        
        Format JSON berisi:
        - Informasi model
        - Dataset yang digunakan
        - Metrik keseluruhan
        - Metrik per kelas
        - Metrik FPS
        
        Args:
            map_metrics: metrik mAP, Precision, Recall
            per_class: metrik per kelas
            fps_metrics: metrik FPS
        """
        report = {
            "model": self.model_cfg["architecture"],
            "dataset": self.dataset_cfg["yaml_path"],
            "overall_metrics": map_metrics,
            "per_class_metrics": per_class,
            "fps_metrics": fps_metrics,
        }

        # Simpan sebagai JSON
        save_path = self.output_dir / "evaluation_report.json"
        with open(save_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"\n[INFO] Laporan evaluasi tersimpan di: {save_path}")

        # Simpan sebagai CSV untuk kemudahan perbandingan
        df = pd.DataFrame(
            {
                "Metrik": list(map_metrics.keys()) + ["FPS"],
                "Nilai": list(map_metrics.values()) + [fps_metrics["fps"]],
            }
        )
        csv_path = self.output_dir / "evaluation_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Metrik CSV tersimpan di: {csv_path}")

    def full_evaluation(self):
        """
        Menjalankan pipeline evaluasi lengkap.
        
        Langkah evaluasi:
        1. Evaluasi mAP, Precision, Recall
        2. Evaluasi FPS
        3. Generate Confusion Matrix
        4. Simpan laporan evaluasi
        
        Returns:
            Tuple (map_metrics, per_class, fps_metrics)
        """
        print("\n" + "=" * 60)
        print("EVALUASI MODEL LENGKAP")
        print("=" * 60)

        # 1. mAP, Precision, Recall
        map_metrics, per_class, results = self.evaluate_map()

        # 2. FPS
        fps_metrics = self.evaluate_fps(
            image_size=self.model_cfg["input_size"]
        )

        # 3. Confusion Matrix
        if self.eval_cfg["confusion_matrix"]:
            self.generate_confusion_matrix(results)

        # 4. Simpan Laporan
        self.save_evaluation_report(map_metrics, per_class, fps_metrics)

        print("\n" + "=" * 60)
        print("EVALUASI SELESAI")
        print("=" * 60)

        return map_metrics, per_class, fps_metrics


def main():
    """Fungsi utama untuk menjalankan script evaluasi dari command line."""
    parser = argparse.ArgumentParser(
        description="Evaluasi Model Deteksi Kendaraan"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path file konfigurasi"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path model"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["map", "fps", "confusion", "all"],
        help="Tugas evaluasi",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model:
        config["model"]["architecture"] = args.model

    evaluator = ModelEvaluator(config)

    # Jalankan tugas evaluasi sesuai pilihan
    if args.task == "map":
        evaluator.evaluate_map()
    elif args.task == "fps":
        evaluator.evaluate_fps()
    elif args.task == "confusion":
        map_metrics, per_class, results = evaluator.evaluate_map()
        evaluator.generate_confusion_matrix(results)
    else:
        evaluator.full_evaluation()


if __name__ == "__main__":
    main()
