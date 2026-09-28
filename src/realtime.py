"""
Deteksi real-time dari webcam / video.

Modul ini SEBELUMNYA punya implementasi sendiri: tracker-less, me-resize
frame distort ke ukuran input, memakai signature ``VehicleCounter`` yang
sudah dihapus, dan menghitung "total" dari jumlah box per frame. Semua itu
salah:

- Resize distort (tidak letterbox) membuat rasio aspek kendaraan berubah,
  jadi box meleset dan akurasi mAP tidak sesuai hasil evaluasi.
- Tanpa tracker, satu kendaraan yang terbaca di 30 frame dihitung 30 kali.
- ``max_lost_frames`` sudah tidak ada di ``VehicleCounter``; pemanggil lama
  akan ``TypeError``.

Sekarang modul ini hanya lapisan tipis: semua logika ada di
``VehiclePipeline`` supaya track_id, ROI, dan counter konsisten dengan CLI
dan GUI. Dijalankan tanpa argumen, webcam dibuka (0).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import VehiclePipeline
from utils.paths import load_config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deteksi + tracking + counting real-time (webcam/video)")
    parser.add_argument("--source", default="0",
                        help="0 = webcam (default), path video, atau URL RTSP")
    parser.add_argument("--config", default=None, help="Path config YAML")
    parser.add_argument("--output", default=None, help="Simpan video hasil")
    parser.add_argument("--no-show", action="store_true",
                        help="Jangan tampilkan preview (untuk server/headless)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Batas frame (untuk uji cepat)")
    parser.add_argument("--save-db", action="store_true",
                        help="Simpan hasil ke database")
    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    print("=== DETEKSI REAL-TIME ===")
    print("Tekan 'q' pada jendela preview untuk berhenti.\n")

    pipeline = VehiclePipeline(config)

    try:
        summary = pipeline.process_video(
            source=args.source,
            output_path=args.output,
            show=not args.no_show,
            max_frames=args.max_frames,
            save_db=args.save_db,
        )
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n[INFO] Dihentikan pengguna.")
        return 0

    print("\nRingkasan:")
    for key in ("frames", "total_detections", "total_tracked_rows",
                "total_counted", "fps", "elapsed_seconds"):
        if key in summary:
            print(f"  {key:20s} {summary[key]}")
    print(f"  {'counts':20s} {summary.get('counts', {})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
