"""
Script Evaluasi Model Deteksi Kendaraan (YOLO11n, CPU).

Perbaikan terhadap versi lama:
- Per-kelas memakai ``ap_class_index`` untuk memetakan class_id -> posisi
  metrik. Versi lama mengindeks ``results.box.ap50[i]`` dengan class_id,
  sehingga kelas kedua bisa membaca AP kelas pertama.
- Confusion Matrix dihitung dari val set dengan IoU matching sungguhan.
  Versi lama mencoba ``for result in results`` pada objek ``DetMetrics``
  (tidak iterable) dan membandingkan ``result.probs.top1`` - ``probs`` hanya
  ada pada klasifikasi, tidak pada deteksi, dan tidak ada pencocokan IoU
  sama sekali, sehingga angkanya tidak bermakna.
- Kurva PR/F1 memakai kurva asli dari Ultralytics (``model.val(plots=True)``).
  Versi lama hanya menggambar sumbu kosong tanpa data lalu menyimpannya
  sebagai "kurva".
- Flag ``--model`` benar-benar dipakai (dulu menulis ke ``architecture``
  sementara objek membaca ``best_weights``).
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: tidak boleh buka window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402

from utils.paths import get_class_names, get_model_path, load_config  # noqa: E402


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU antara dua set bbox [N,4] dan [M,4] (format xyxy)."""
    if pred.size == 0 or gt.size == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    px1, py1, px2, py2 = pred[:, None, 0], pred[:, None, 1], pred[:, None, 2], pred[:, None, 3]
    gx1, gy1, gx2, gy2 = gt[None, :, 0], gt[None, :, 1], gt[None, :, 2], gt[None, :, 3]
    ix1 = np.maximum(px1, gx1)
    iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2)
    iy2 = np.minimum(py2, gy2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_p = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
    area_g = np.clip(gx2 - gx1, 0, None) * np.clip(gy2 - gy1, 0, None)
    union = area_p + area_g - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def read_yolo_label(label_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Baca label YOLO. Returns (classes[N], boxes_xyxy_pixels[N,4])."""
    if not label_path.is_file():
        return np.zeros(0, dtype=int), np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows.append([float(p) for p in parts[:5]])
    if not rows:
        return np.zeros(0, dtype=int), np.zeros((0, 4), dtype=np.float32)

    arr = np.array(rows, dtype=np.float32)
    classes = arr[:, 0].astype(int)
    # normalize -> pixel
    cx, cy, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return classes, boxes


def xyxy_to_pixel(boxes_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    """Box ternormalisasi (0-1) -> pixel xyxy."""
    out = boxes_norm.copy()
    out[:, [0, 2]] *= width
    out[:, [1, 3]] *= height
    return out


class ModelEvaluator:
    """Evaluator model YOLO11 untuk deteksi kendaraan."""

    # Default di level kelas supaya instance yang dibuat tanpa __init__
    # (dipakai saat pengujian) tetap punya split yang valid.
    split: str = "val"

    def __init__(self, config: dict, model_path: Optional[str] = None,
                 split: Optional[str] = None):
        self.config = config
        self.eval_cfg = config.get("evaluation", {})
        self.model_cfg = config["model"]
        self.dataset_cfg = config["dataset"]
        self.output_dir = Path(self.eval_cfg.get("output_dir", "outputs/evaluation"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Split untuk mAP dan confusion matrix. Default "val" menjaga
        # perilaku lama; "test" dipakai untuk pengujian akhir karena test
        # set tidak boleh dipakai untuk tuning apa pun.
        self.split = (split or "val").lower()
        if self.split not in ("val", "test", "train"):
            raise ValueError(f"split tidak dikenal: {self.split} "
                             f"(pilihan: train, val, test)")

        self.model_path = get_model_path(config, explicit=model_path)
        print(f"[INFO] Memuat model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.runs_dir = str(self.model_cfg.get("runs_dir", "runs/detect"))

        self.class_names: Dict[int, str] = get_class_names(config)
        self.nc = len(self.class_names)

        self.imgsz = int(self.model_cfg.get("input_size", 416))
        self.device = self.model_cfg.get("device", "cpu")
        self.conf_threshold = float(self.model_cfg.get("confidence_threshold", 0.5))
        self.iou_threshold = float(self.model_cfg.get("iou_threshold", 0.45))

        # Ambang bawah saat menghitung AP. Ultralytics memotong kurva PR
        # pada nilai ini, jadi memakai ambang operasional (0.5) akan
        # menghasilkan AP yang jauh lebih rendah dan TIDAK sebanding dengan
        # mAP literatur / target skripsi. 0.001 adalah nilai bawaan
        # Ultralytics. Precision/Recall pada ambang operasional dilaporkan
        # terpisah lewat confusion matrix.
        self.map_conf = float(self.eval_cfg.get("map_conf", 0.001))

        self._val_run_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # 1. mAP / Precision / Recall
    # ------------------------------------------------------------------

    def evaluate_map(self, plots: bool = True):
        """
        Jalankan ``model.val()`` dan ekstrak metrik.

        Metrik memakai ``conf=self.map_conf`` (default 0.001) supaya
        Precision/Recall/F1 dan mAP punya arti standar. Split mengikuti
        ``self.split``, jadi ``--split test`` benar-benar mengevaluasi test
        set, bukan hanya confusion matrix.

        Returns:
            (metrics_overall, metrics_per_class, val_results, run_dir)
        """
        print("\n" + "=" * 60)
        print("EVALUASI MODEL - mAP, PRECISION, RECALL")
        print("=" * 60)
        print(f"[INFO] Split: {self.split}  conf>={self.map_conf}  "
              f"imgsz={self.imgsz}  NMS iou={self.iou_threshold}")

        results = self.model.val(
            data=self.dataset_cfg["yaml_path"],
            split=self.split,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.map_conf,
            iou=self.iou_threshold,
            plots=plots,
            verbose=False,
        )

        box = results.box
        metrics = {
            "mAP50": float(box.map50),
            "mAP50-95": float(box.map),
            "Precision": float(box.mp),
            "Recall": float(box.mr),
            "F1": float(np.mean(box.f1)) if len(box.f1) else 0.0,
        }

        # Per-kelas: array metrik diindeks POSISI, bukan class_id.
        # Posisi ke-i milik class_id == ap_class_index[i].
        per_class: Dict[str, dict] = {}
        for pos, class_id in enumerate(box.ap_class_index):
            name = self.class_names.get(int(class_id), f"class_{int(class_id)}")
            per_class[name] = {
                "class_id": int(class_id),
                "AP50": float(box.ap50[pos]) if pos < len(box.ap50) else 0.0,
                "AP50-95": float(box.ap[pos]) if pos < len(box.ap) else 0.0,
                "Precision": float(box.p[pos]) if pos < len(box.p) else 0.0,
                "Recall": float(box.r[pos]) if pos < len(box.r) else 0.0,
            }
            if pos < len(box.f1):
                per_class[name]["F1"] = float(box.f1[pos])

        # Kelas yang tidak ada di val set tetap dilaporkan sebagai 0 supaya
        # tabel per-kelas lengkap dan tidak menyesatkan.
        for class_id, name in self.class_names.items():
            per_class.setdefault(name, {
                "class_id": int(class_id), "AP50": 0.0, "AP50-95": 0.0,
                "Precision": 0.0, "Recall": 0.0, "F1": 0.0,
                "note": f"tidak ada instance kelas ini di split {self.split}",
            })

        run_dir = self._find_val_run_dir(results)

        print("\n[HASIL] Metrik Keseluruhan:")
        for metric, value in metrics.items():
            print(f"  {metric:12s}: {value:.4f}")

        print("\n[HASIL] Metrik per Kelas:")
        for name, m in sorted(per_class.items()):
            print(f"  {name:6s} AP50={m['AP50']:.4f}  AP50-95={m['AP50-95']:.4f}  "
                  f"P={m['Precision']:.4f}  R={m['Recall']:.4f}")

        return metrics, per_class, results, run_dir

    def _find_val_run_dir(self, results) -> Optional[Path]:
        """
        Temukan folder output ``val()`` yang baru saja dibuat.

        ``model.val()`` mengembalikan ``DetMetrics``, dan objek itu TIDAK
        punya atribut ``save_dir`` - versi lama memakai ``results.save_dir``
        di dalam ``try/except``, jadi selalu gagal diam-diam dan kurva
        (``PR_curve.png`` dll) tidak pernah ikut dikumpulkan ke laporan.

        Sumber yang benar ada di ``model.validator.save_dir``, yang di-set
        oleh Ultralytics selama validasi berjalan.
        """
        candidates = [
            getattr(getattr(self.model, "validator", None), "save_dir", None),
            getattr(results, "save_dir", None),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                path = Path(candidate)
            except TypeError:
                continue
            if path.is_dir():
                self._val_run_dir = path
                return path

        if self._val_run_dir is not None:
            return self._val_run_dir

        # Terakhir: cari run val terbaru berdasarkan waktu tulis.
        # Ultralytics memakai nama ``val``, ``val2``, ``val3``, ... (tanpa
        # tanda hubung), jadi glob ``val*`` wajib menyertakan keduanya.
        try:
            base = Path(self.runs_dir)
            if base.is_dir():
                candidates = [p for p in base.glob("val*") if p.is_dir()]
                latest = max(candidates, key=lambda p: p.stat().st_mtime,
                             default=None)
                if latest is not None:
                    self._val_run_dir = latest
                    return latest
        except Exception:  # noqa: BLE001
            pass
        return None

    # ------------------------------------------------------------------
    # 2. FPS
    # ------------------------------------------------------------------

    def evaluate_fps(self, num_images: int = 50, warmup: int = 10) -> dict:
        """
        Ukur kecepatan inferensi pada frame 1920x1080 yang di-letterbox ke
        ``imgsz``. Versi lama memakai gambar square 416x416, yang jauh lebih
        murah daripada frame asli 16:9 dan membuat FPS terlihat lebih baik
        dari kenyataan.
        """
        print("\n" + "=" * 60)
        print("EVALUASI MODEL - FPS (Kecepatan Inferensi)")
        print("=" * 60)

        # Resolensi asli dataset, lalu di-letterbox oleh Ultralytics
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        for _ in range(warmup):
            self.model.predict(frame, imgsz=self.imgsz, device=self.device, verbose=False)

        times = []
        for _ in range(num_images):
            t0 = time.perf_counter()
            self.model.predict(frame, imgsz=self.imgsz, device=self.device, verbose=False)
            times.append(time.perf_counter() - t0)

        avg_time = float(np.mean(times))
        metrics = {
            "avg_inference_time_ms": avg_time * 1000,
            "fps": 1.0 / avg_time if avg_time > 0 else 0.0,
            "num_images": num_images,
            "source_resolution": "1920x1080",
            "imgsz": self.imgsz,
            "device": self.device,
        }

        print(f"  Resolusi sumber : 1920x1080 (di-letterbox ke {self.imgsz})")
        print(f"  Device          : {self.device}")
        print(f"  Rata-rata waktu : {avg_time * 1000:.2f} ms")
        print(f"  FPS             : {metrics['fps']:.2f}")
        return metrics

    # ------------------------------------------------------------------
    # 3. Confusion Matrix (dihitung sendiri, IoU matching)
    # ------------------------------------------------------------------

    def generate_confusion_matrix(self, iou_thresh: float = 0.5,
                                  conf_thresh: Optional[float] = None,
                                  max_images: Optional[int] = None) -> Tuple[Optional[Path], dict]:
        """
        Confusion matrix untuk DETEKSI, dihitung dari val set.

        Matriks berukuran (nc+1) x (nc+1); baris/kolom terakhir adalah
        background:
          - diagonal          : True Positive
          - kolom background  : False Positive (terdeteksi, tidak ada GT)
          - baris background  : False Negative (ada GT, tidak terdeteksi)
          - diagonal background: prediksi benar untuk objectness, tidak dipakai

        Returns:
            (path_png, ringkasan) di mana ringkasan berisi tp/fp/fn,
            presisi, recall, dan matriks mentah. Angka inilah yang dipakai
            laporan - bukan dihitung ulang dari file label, karena
            menghitung ulang hanya menghasilkan jumlah ground truth dan
            membuat laporan salah (KeyError 'tp').
        """
        conf_thresh = self.conf_threshold if conf_thresh is None else conf_thresh
        split = self.split or "val"
        print(f"\n[INFO] Menghitung confusion matrix dari {split} set "
              f"(IoU>={iou_thresh}, conf>={conf_thresh})...")
        empty = (None, {})

        # Split yang diminta dibaca langsung dari folder annotated, BUKAN dari
        # dataset.yaml. yaml hanya pernah menunjuk ke val, sehingga tanpa
        # override ini test set tidak pernah bisa dievaluasi padahal sudah
        # disiapkan justru untuk pengujian akhir.
        images_root = Path(self.dataset_cfg["val_images"]).parent
        val_dir = images_root / split
        if not val_dir.is_dir():
            print(f"[PERINGATAN] Folder gambar {split} tidak ditemukan: "
                  f"{val_dir}")
            return empty

        images = sorted(
            p for p in val_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if not images:
            print(f"[PERINGATAN] Tidak ada gambar di {val_dir}")
            return empty
        if max_images:
            images = images[:max_images]

        import cv2  # lokal: hanya dipakai di fungsi ini

        # Struktur YOLO: <root>/images/<split>/x.jpg  ->  <root>/labels/<split>/x.txt
        # val_dir = .../data/annotated/images/val, jadi parent.parent = .../annotated
        labels_root = val_dir.parent.parent / "labels" / val_dir.name
        if not labels_root.is_dir():
            alt = val_dir.parent / "labels" / val_dir.name
            if alt.is_dir():
                labels_root = alt
            else:
                print(f"[PERINGATAN] Folder label tidak ditemukan untuk: {val_dir}")
                return empty

        bg = self.nc  # index background
        cm = np.zeros((self.nc + 1, self.nc + 1), dtype=np.int64)
        images_used = 0

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            images_used += 1

            label_path = labels_root / (img_path.stem + ".txt")
            gt_cls, gt_box_norm = read_yolo_label(label_path)
            gt_boxes = xyxy_to_pixel(gt_box_norm, w, h)

            pred = self.model.predict(img, conf=conf_thresh, iou=self.iou_threshold,
                                      imgsz=self.imgsz, device=self.device,
                                      verbose=False)[0]
            if pred.boxes is not None and len(pred.boxes) > 0:
                p_cls = pred.boxes.cls.cpu().numpy().astype(int)
                p_conf = pred.boxes.conf.cpu().numpy().astype(float)
                p_boxes = pred.boxes.xyxy.cpu().numpy().astype(np.float32)
            else:
                p_cls = np.zeros(0, dtype=int)
                p_conf = np.zeros(0, dtype=float)
                p_boxes = np.zeros((0, 4), dtype=np.float32)

            # GT dengan class_id di luar mapping diabaikan (tidak bisa dinilai)
            valid = (gt_cls >= 0) & (gt_cls < self.nc)
            gt_cls, gt_boxes = gt_cls[valid], gt_boxes[valid]

            ious = iou_matrix(p_boxes, gt_boxes)
            gt_matched = np.zeros(len(gt_cls), dtype=bool)

            # Prediksi diurutkan dari confidence tertinggi
            for pi in np.argsort(-p_conf):
                cls = int(p_cls[pi])
                if cls < 0 or cls >= self.nc:
                    cm[bg, bg] += 1
                    continue
                best_j, best_iou = -1, iou_thresh
                for gi in range(len(gt_cls)):
                    if gt_matched[gi] or int(gt_cls[gi]) != cls:
                        continue
                    if ious[pi, gi] >= best_iou:
                        best_iou, best_j = ious[pi, gi], gi
                if best_j >= 0:
                    gt_matched[best_j] = True
                    cm[cls, cls] += 1          # TP
                else:
                    cm[bg, cls] += 1           # FP

            for gi in range(len(gt_cls)):
                if not gt_matched[gi]:
                    cm[int(gt_cls[gi]), bg] += 1  # FN

        if images_used == 0 or cm.sum() == 0:
            print("[PERINGATAN] Tidak ada data untuk confusion matrix.")
            return empty

        labels = [self.class_names[i] for i in range(self.nc)] + ["background"]
        col_totals = cm.sum(axis=0)
        cm_norm = cm / np.maximum(col_totals, 1)

        fig, ax = plt.subplots(figsize=(1.1 * self.nc + 3, 1.0 * self.nc + 2.5))
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    cbar_kws={"label": "fraksi"})
        ax.set_title(f"Confusion Matrix (IoU>={iou_thresh}, conf>={conf_thresh})\n"
                     f"{images_used} gambar {self.split}")
        ax.set_ylabel("Label Sebenarnya")
        ax.set_xlabel("Label Prediksi")
        fig.tight_layout()
        # Nama file ikut split. Versi lama selalu menulis
        # "confusion_matrix.png", jadi evaluasi pada test menimpa hasil val
        # dan menyisakan satu gambar tanpa keterangan split mana.
        save_path = self.output_dir / f"confusion_matrix_{self.split}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Confusion matrix tersimpan: {save_path}")

        n_tp = int(np.trace(cm[: self.nc, : self.nc]))
        n_fp = int(cm[self.nc, : self.nc].sum())
        n_fn = int(cm[: self.nc, self.nc].sum())
        n_ignored = int(cm[self.nc, self.nc])
        p_cm = float(n_tp / max(n_tp + n_fp, 1))
        r_cm = float(n_tp / max(n_tp + n_fn, 1))
        f1_cm = float(2 * p_cm * r_cm / max(p_cm + r_cm, 1e-9))
        print(f"  TP={n_tp}  FP={n_fp}  FN={n_fn}  "
              f"Precision={p_cm:.4f}  Recall={r_cm:.4f}  F1={f1_cm:.4f}")

        summary = {
            "tp": n_tp,
            "fp": n_fp,
            "fn": n_fn,
            "precision": p_cm,
            "recall": r_cm,
            "f1": f1_cm,
            "images": int(images_used),
            "ground_truth_boxes": int(cm[: self.nc, self.nc].sum() + n_tp),
            "ignored_class_predictions": n_ignored,
            "iou_threshold": float(iou_thresh),
            "conf_threshold": float(conf_thresh),
            "matrix": cm.tolist(),
        }
        return save_path, summary

    # ------------------------------------------------------------------
    # 4. Kurva dari Ultralytics
    # ------------------------------------------------------------------

    def collect_curves(self, run_dir: Optional[Path]) -> List[Path]:
        """
        Salin kurva asli yang dibuat ``model.val(plots=True)``.

        File yang dicari: PR_curve.png, F1_curve.png, P_curve.png,
        R_curve.png. Versi lama membuat file PNG kosong (hanya sumbu tanpa
        garis) dan menyimpannya seolah-olah itu kurva.
        """
        if not run_dir or not run_dir.is_dir():
            print("[PERINGATAN] Folder run val() tidak ditemukan, kurva dilewati.")
            return []

        # Ultralytics terbaru menamai file kurva dengan prefiks tugas:
        # ``BoxPR_curve.png``, bukan ``PR_curve.png``. Daftar lama hanya
        # mencari nama tanpa prefiks, jadi tidak ada yang ketemu dan laporan
        # selalu claiming "kurva dilewati" padahal val(plots=True) sudah
        # membuat gambarnya. Karena itu kedua pola dicoba.
        stems = ["PR_curve", "F1_curve", "P_curve", "R_curve"]
        copied: List[Path] = []
        for stem in stems:
            src = None
            for candidate in (run_dir / f"{stem}.png",
                              run_dir / f"Box{stem}.png",
                              run_dir / f"{stem}_box.png"):
                if candidate.is_file():
                    src = candidate
                    break
            if src is None:
                continue
            dst = self.output_dir / f"{stem}.png"
            shutil.copy2(src, dst)
            copied.append(dst)
            print(f"[INFO] Kurva disalin: {src.name} -> {dst.name}")

        if not copied:
            tersedia = sorted(p.name for p in run_dir.glob("*.png"))
            print("[PERINGATAN] Tidak ada file kurva di folder run val(). "
                  f"File PNG yang tersedia: {tersedia or 'tidak ada'}")
        return copied

    # ------------------------------------------------------------------
    # 5. Laporan
    # ------------------------------------------------------------------

    def save_evaluation_report(self, map_metrics: dict, per_class: dict,
                               fps_metrics: dict, confusion: dict,
                               curves: List[Path]) -> Path:
        report = {
            "model": str(self.model_path),
            "architecture": self.model_cfg.get("architecture"),
            "dataset": self.dataset_cfg["yaml_path"],
            "map_split": self.split,
            "map_conf": self.map_conf,
            "confusion_split": self.split,
            "imgsz": self.imgsz,
            "device": self.device,
            "confidence_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "overall_metrics": map_metrics,
            "per_class_metrics": per_class,
            "fps_metrics": fps_metrics,
            "confusion_matrix": confusion,
            "curves": [p.name for p in curves],
        }
        # overall_metrics memakai conf=map_conf (0.001) sehingga mAP/P/R/F1
        # punya arti standar; confusion_matrix memakai conf operasional
        # (confidence_threshold) dan inilah yang menggambarkan perilaku
        # sistem saat berjalan. Keduanya kini memakai split yang sama, jadi
        # tidak ada lagi angka val yang terlanjur dikira angka test.
        report["note"] = (
            f"overall_metrics dan per_class_metrics dihitung pada split "
            f"'{self.split}' dengan conf>={self.map_conf} (mAP standar). "
            f"confusion_matrix dihitung pada split '{self.split}' dengan "
            f"conf>={self.conf_threshold} (ambang operasional)."
        )
        save_path = self.output_dir / "evaluation_report.json"
        text = json.dumps(report, indent=2, ensure_ascii=False)
        save_path.write_text(text, encoding="utf-8")
        # Salinan ber-akhiran split: file tanpa akhiran selalu milik run
        # terakhir, jadi hasil val dan test tetap bisa dibaca berdampingan.
        (self.output_dir / f"evaluation_report_{self.split}.json").write_text(
            text, encoding="utf-8")
        print(f"\n[INFO] Laporan: {save_path}")

        rows = [{"Metrik": k, "Nilai": v} for k, v in map_metrics.items()]
        if fps_metrics:
            if "fps" in fps_metrics:
                rows.append({"Metrik": "FPS", "Nilai": fps_metrics["fps"]})
            if "avg_inference_time_ms" in fps_metrics:
                rows.append({"Metrik": "Avg Inference (ms)",
                             "Nilai": fps_metrics["avg_inference_time_ms"]})
        if confusion:
            # .get() bukan [""] supaya bagian laporan yang tidak punya
            # confusion matrix tetap bisa ditulis.
            for label, key in (("TP (CM)", "tp"), ("FP (CM)", "fp"),
                               ("FN (CM)", "fn"),
                               ("Precision (CM)", "precision"),
                               ("Recall (CM)", "recall"),
                               ("F1 (CM)", "f1")):
                if key in confusion:
                    value = confusion[key]
                    rows.append({
                        "Metrik": label,
                        "Nilai": round(value, 4) if isinstance(value, float) else value,
                    })
        metrics_csv = self.output_dir / "evaluation_metrics.csv"
        pd.DataFrame(rows).to_csv(metrics_csv, index=False)

        per_class_csv = self.output_dir / "evaluation_per_class.csv"
        pd.DataFrame([
            {"Kelas": name, **{k: v for k, v in m.items() if k != "note"}}
            for name, m in per_class.items()
        ]).to_csv(per_class_csv, index=False)

        for src in (metrics_csv, per_class_csv):
            shutil.copy2(src, src.with_name(f"{src.stem}_{self.split}{src.suffix}"))
        print(f"[INFO] CSV  : {metrics_csv}")
        return save_path

    # ------------------------------------------------------------------
    # Orquestrasi
    # ------------------------------------------------------------------

    def full_evaluation(self) -> dict:
        print("\n" + "=" * 60)
        print("EVALUASI MODEL LENGKAP")
        print("=" * 60)

        map_metrics, per_class, _, run_dir = self.evaluate_map(plots=True)
        fps_metrics = self.evaluate_fps()

        confusion = {}
        if self.eval_cfg.get("confusion_matrix", True):
            _, confusion = self.generate_confusion_matrix()

        curves: List[Path] = []
        if self.eval_cfg.get("pr_curve", True) or self.eval_cfg.get("f1_curve", True):
            curves = self.collect_curves(run_dir)

        report = self.save_evaluation_report(map_metrics, per_class, fps_metrics,
                                            confusion, curves)

        print("\n" + "=" * 60)
        print(f"RINGKASAN [{self.split}, conf>={self.map_conf}]: "
              f"mAP50={map_metrics['mAP50']:.4f}  "
              f"mAP50-95={map_metrics['mAP50-95']:.4f}  "
              f"P={map_metrics['Precision']:.4f}  R={map_metrics['Recall']:.4f}  "
              f"F1={map_metrics['F1']:.4f}  "
              f"FPS={fps_metrics.get('fps', 0.0):.1f}")
        if confusion:
            print(f"           di ambang conf>={confusion.get('conf_threshold')}: "
                  f"TP={confusion['tp']}  FP={confusion['fp']}  "
                  f"FN={confusion['fn']}  P={confusion['precision']:.4f}  "
                  f"R={confusion['recall']:.4f}  F1={confusion['f1']:.4f}")
        print("=" * 60)
        return {
            "overall": map_metrics,
            "per_class": per_class,
            "fps": fps_metrics,
            "confusion": confusion,
            "curves": [p.name for p in curves],
            "report": str(report),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluasi Model Deteksi Kendaraan")
    parser.add_argument("--config", default=None, help="path config.yaml")
    parser.add_argument("--model", default=None, help="override path bobot model")
    parser.add_argument("--task", default="all",
                        choices=["map", "fps", "confusion", "all"],
                        help="tugas evaluasi yang dijalankan")
    parser.add_argument("--fps-images", type=int, default=50,
                        help="jumlah frame untuk ukur FPS")
    parser.add_argument("--max-images", type=int, default=None,
                        help="batasi jumlah gambar (untuk uji cepat)")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                        help="split yang dievaluasi - mAP/P/R/F1 sekaligus "
                             "confusion matrix (default: val). Gunakan "
                             "'test' untuk pengujian akhir.")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    try:
        evaluator = ModelEvaluator(config, model_path=args.model,
                                   split=args.split)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if args.task == "map":
        evaluator.evaluate_map()
    elif args.task == "fps":
        evaluator.evaluate_fps(num_images=args.fps_images)
    elif args.task == "confusion":
        evaluator.generate_confusion_matrix(max_images=args.max_images)
    else:
        evaluator.full_evaluation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
