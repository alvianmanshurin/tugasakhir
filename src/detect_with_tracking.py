"""
CLI deteksi kendaraan + ROI + pelacakan + hitung dual-line (CPU-friendly).

File ini SENGAJA hanya pembungkus ``pipeline.VehiclePipeline``. Versi
sebelumnya menyalin seluruh logika deteksi, tracking, penghitungan, dan
database ke sini, sehingga program ini dan GUI punya dua tracker dengan dua
ruang ID berbeda - sehingga ID yang tampil di video tidak pernah cocok
dengan yang tersimpan di database.

Logika pipeline hanya ada di ``src/pipeline.py``.

Contoh:
    python src/detect_with_tracking.py --source video.mp4 --output out.mp4
    python src/detect_with_tracking.py --source 0 --show
    python src/detect_with_tracking.py --source video.mp4 --db
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import VehiclePipeline
from utils.paths import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deteksi + ROI + tracking + hitung dual-line kendaraan (YOLO11n, CPU)"
    )
    parser.add_argument("--source", default="0",
                        help="path video, URL, atau '0' untuk webcam")
    parser.add_argument("--model", default=None,
                        help="override path bobot (default dari config.yaml)")
    parser.add_argument("--config", default=None,
                        help="override path config.yaml")
    parser.add_argument("--output", default=None,
                        help="simpan video hasil ke path ini")
    parser.add_argument("--show", action="store_true",
                        help="tampilkan preview (tekan q atau ESC untuk berhenti)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="batas jumlah frame (untuk uji cepat)")
    parser.add_argument("--db", action="store_true",
                        help="simpan hasil ke database SQLite")
    parser.add_argument("--device", default=None, help="override device, mis. cpu / 0")
    parser.add_argument("--conf", type=float, default=None,
                        help="override ambang confidence high (track baru)")
    parser.add_argument("--low-conf", type=float, default=None,
                        help="override ambang low (rescue track); harus <= --conf")
    parser.add_argument("--no-roi", action="store_true",
                        help="nonaktifkan filter ROI")
    parser.add_argument("--no-count", action="store_true",
                        help="tampilkan deteksi/tracking tanpa menghitung")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config = load_config(Path(args.config) if args.config else None)

    # Override harus diterapkan ke config SEBELUM VehiclePipeline dibangun,
    # karena ambang confidence/ROI dibaca di __init__.
    if args.conf is not None:
        config["model"]["confidence_threshold"] = args.conf
    if args.low_conf is not None:
        config["model"]["low_conf_threshold"] = args.low_conf
    if args.no_roi:
        config["roi"]["enabled"] = False
        config["roi"]["draw_roi"] = False

    pipeline = VehiclePipeline(config=config, model_path=args.model, device=args.device)

    if args.no_count:
        # Matikan penghitungan tanpa mematikan deteksi/tracking: set posisi
        # garis agar tidak ada lintasan yang bisa memenuhi syarat hitung.
        pipeline.counter.min_track_length = 10 ** 9
        pipeline.counter.min_displacement = 10 ** 9

    print("=== DETEKSI + ROI + TRACKING + HITUNG ===")
    print(f"Model       : {pipeline.model_path}")
    print(f"Device      : {pipeline.device}  imgsz={pipeline.imgsz}")
    print(f"Ambang      : low {pipeline.low_conf_threshold} / high {pipeline.conf_threshold}")
    print(f"ROI         : {'ON' if pipeline.roi_filter.config.enabled else 'OFF'}"
          f"  (cakupan {pipeline.roi_filter.coverage_ratio((1080, 1920, 3)):.1%} @1920x1080)")
    print(f"Tracker     : min_hits={pipeline.tracker.min_hits} "
          f"max_age={pipeline.tracker.max_age} "
          f"max_distance={pipeline.tracker.max_distance}px")
    print(f"Garis hitung: {pipeline.counter.line1_position} / "
          f"{pipeline.counter.line2_position} arah={pipeline.counter.direction}")
    print(f"Sumber      : {args.source}")
    if args.output:
        print(f"Output      : {args.output}")
    print(f"Database    : {'AKTIF' if args.db else 'tidak aktif'}")
    print("-" * 60)

    try:
        summary = pipeline.process_video(
            args.source,
            output_path=args.output,
            show=args.show,
            max_frames=args.max_frames,
            save_db=args.db,
        )
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return 1

    print("-" * 60)
    print("RINGKASAN")
    print(f"  Frame diproses  : {summary['total_frames']}")
    print(f"  FPS rerata      : {summary['avg_fps']:.1f}  (kecepatan pemrosesan nyata)")
    print(f"  Deteksi model   : {summary['total_detections']} (output YOLO, termasuk "
          f"yang dibuang ROI)")
    print(f"  Baris track     : {summary['total_tracked_rows']} (1 per track per frame "
          f"- inilah yang masuk DB)")
    if summary.get("total_after_roi_rows"):
        print(f"  Setelah ROI     : {summary['total_after_roi_rows']} "
              f"(deteksi yang lolos filter ROI)")
    print(f"  KENDARAAN       : {summary['total_counted']} (unik per track - inilah "
          f"angka yang benar)")
    for cls, n in sorted(summary["by_class"].items()):
        print(f"    - {cls:6s}: {n}")
    for direction, label in (("down", "MASUK"), ("up", "KELUAR")):
        per_cls = summary["by_direction"].get(direction) or {}
        if per_cls:
            print(f"  {label:7s}: {dict(per_cls)}")
    if args.db:
        print("\n  Database       : tersimpan (total_counted = kendaraan unik)")
    return 0


# Alias backwards-compatible: kode lama mengimpor VehicleDetectionPipeline.
VehicleDetectionPipeline = VehiclePipeline


if __name__ == "__main__":
    raise SystemExit(main())
