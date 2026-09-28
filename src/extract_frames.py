"""
Ekstraksi Frame Video untuk Dataset Deteksi Kendaraan
Ekstrak frame dari video KIRI/TENGAH di gerbang masuk Kampus ITERA

Perubahan terhadap versi lama:

* ``load_config("config/config.yaml")`` diselesaikan relatif ke cwd, jadi
  menjalankan skrip dari folder lain berakhir dengan FileNotFoundError.
  Sekarang memakai ``utils.paths.load_config`` yang path-nya absolut.
* Output ``data/raw`` dan ``data/annotated/...`` juga relatif ke cwd, sehingga
  ekstraksi bisa menulis frame ke folder di luar project.
* ``split_to_trainval()`` adalah splitter kedua yang membagi gambar SATUAN
  ACAK per file, bukan per grup. Dua frame berdekatan dari video yang sama
  bisa masuk train dan val sekaligus, dan metrik validasi jadi tergelembung -
  masalah yang sudah diselesaikan ``dataset_prepare.split_dataset()``. Splitter
  kedua ini dihapus; ``--action split`` sekarang meneruskan ke implementasi
  group-aware yang sama.
* Split lama juga menyentuh file label KOSONG untuk semua gambar, termasuk
  gambar yang kendaraannya belum dianotasi. Anotasi kosong berarti "tidak ada
  kendaraan", jadi training belajar mengabaikan kendaraan yang ada.
* Semua kesalahan argumen keluar dengan exit code 0.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import yaml

from utils.paths import DATA_DIR, load_config

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv")


class VideoFrameExtractor:
    """
    Ekstraktor frame video untuk pembuatan dataset.

    Fitur:
    - List semua video yang tersedia
    - Dapatkan informasi video (FPS, resolusi, durasi)
    - Ekstrak frame dengan interval tertentu
    - Split frame menjadi train/val (group-aware, via dataset_prepare)
    """

    def __init__(self, config):
        """
        Inisialisasi ekstraktor frame.

        Args:
            config: dict konfigurasi dari config.yaml
        """
        self.config = config
        dataset_cfg = config.get("dataset", {}) or {}
        self.video_source = dataset_cfg.get("video_source", str(DATA_DIR / "raw"))
        # Absolut terhadap root project, bukan cwd.
        self.output_dir = DATA_DIR / "raw"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_videos(self):
        """
        List semua video yang tersedia di direktori sumber.

        Returns:
            Daftar path video yang ditemukan
        """
        video_dir = Path(self.video_source)
        if not video_dir.is_absolute():
            video_dir = DATA_DIR / video_dir
        if not video_dir.exists():
            print(f"[ERROR] Direktori video tidak ditemukan: {video_dir}")
            return []

        # Cari semua ekstensi video, bukan hanya .MOV dan .mp4.
        videos = sorted({p for ext in VIDEO_EXTS
                         for p in video_dir.glob(f"*{ext}")})
        print(f"\n[INFO] Ditemukan {len(videos)} video di: {video_dir}")
        for v in videos:
            size_mb = v.stat().st_size / (1024 * 1024)
            print(f"  - {v.name} ({size_mb:.1f} MB)")

        return videos

    def get_video_info(self, video_path):
        """
        Mendapatkan informasi video.

        Args:
            video_path: path ke file video

        Returns:
            Dict berisi fps, total_frames, width, height, duration
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0

        cap.release()
        return info

    def extract_frames(self, video_path, output_dir=None, interval=30,
                       max_frames=None, resize=None):
        """
        Ekstrak frame dari video.

        Proses:
        1. Buka video
        2. Baca frame satu per satu
        3. Simpan frame setiap N interval
        4. Resize jika diperlukan
        5. Hentikan jika mencapai batas max_frames

        Args:
            video_path: path ke file video
            output_dir: direktori output (opsional)
            interval: ekstrak setiap N frame
            max_frames: jumlah frame maksimum
            resize: tuple (width, height) untuk resize

        Returns:
            Daftar path file frame yang diekstrak
        """
        video_path = Path(video_path)
        if not video_path.is_file():
            print(f"[ERROR] Video tidak ditemukan: {video_path}")
            return []

        if interval < 1:
            print("[ERROR] --interval minimal 1 (0 akan membagi dengan nol)")
            return []

        if output_dir:
            out_dir = Path(output_dir)
        else:
            # Buat subdirektori berdasarkan nama video
            out_dir = self.output_dir / video_path.stem

        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"EKSTRAKSI FRAME")
        print(f"{'=' * 60}")
        print(f"Video: {video_path.name}")

        # Dapatkan info video
        info = self.get_video_info(video_path)
        if info:
            print(f"Resolusi: {info['width']}x{info['height']}")
            print(f"FPS: {info['fps']:.1f}")
            print(f"Durasi: {info['duration']:.1f}s")
            print(f"Total frame: {info['total_frames']}")

        # Jarak antar frame hasil ekstrak menentukan seberapa mirip dua gambar
        # berturut-turut. Kalau jaraknya terlalu dekat, dataset jadi didominasi
        # frame nyaris identik.
        if info and info["fps"] > 0:
            print(f"Jarak antar frame: {interval / info['fps']:.1f} detik")
            if interval / info["fps"] < 1.0:
                print("[PERINGATAN] Jarak < 1 detik menghasilkan banyak frame "
                      "hampir identik (duplikat).")

        print(f"Ekstrak setiap: {interval} frame")
        print(f"Output: {out_dir}")

        # Buka video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Tidak dapat membuka video: {video_path}")
            return []

        frame_count = 0
        extracted = 0
        extracted_files = []
        start_time = time.time()

        # Hitung file yang sudah ada untuk melanjutkan penomoran
        existing = len(list(out_dir.glob("*.jpg")))

        # Loop ekstraksi frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Ekstrak frame setiap interval tertentu
            if frame_count % interval == 0:
                # Resize jika ditentukan
                if resize:
                    frame = cv2.resize(frame, resize)

                # Simpan frame
                filename = f"{video_path.stem}_{existing + extracted:05d}.jpg"
                save_path = out_dir / filename
                if not cv2.imwrite(str(save_path), frame):
                    print(f"[ERROR] Gagal menulis: {save_path}")
                    cap.release()
                    return extracted_files
                extracted_files.append(save_path)
                extracted += 1

                # Tampilkan progress
                if extracted % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_extract = extracted / elapsed if elapsed > 0 else 0
                    print(f"  Diekstrak: {extracted} frame | FPS: {fps_extract:.1f}")

                # Cek batas frame
                if max_frames and extracted >= max_frames:
                    print(f"  Mencapai batas frame: {max_frames}")
                    break

            frame_count += 1

        cap.release()
        elapsed = time.time() - start_time

        print(f"\n[OK] Berhasil ekstrak {extracted} frame ke: {out_dir}")
        print(f"[OK] Waktu: {elapsed:.1f}s")

        return extracted_files

    def extract_all_videos(self, interval=30, max_frames_per_video=500, resize=None):
        """
        Ekstrak frame dari semua video.

        Args:
            interval: ekstrak setiap N frame
            max_frames_per_video: frame maksimum per video
            resize: tuple (width, height) untuk resize
        Returns:
            Total frame yang diekstrak, atau ``None`` bila tidak ada video
            (supaya ``main()`` bisa mengembalikan exit code gagal).
        """
        videos = self.list_videos()

        if not videos:
            print("[ERROR] Tidak ada video ditemukan")
            return None

        print(f"\n[INFO] Memproses {len(videos)} video...")
        print(f"[INFO] Interval: setiap {interval} frame")
        print(f"[INFO] Frame maks per video: {max_frames_per_video}")

        total_extracted = 0

        for video in videos:
            print(f"\n{'=' * 60}")
            print(f"Memproses: {video.name}")
            print(f"{'=' * 60}")

            files = self.extract_frames(
                video,
                interval=interval,
                max_frames=max_frames_per_video,
                resize=resize,
            )
            total_extracted += len(files)

        print(f"\n{'=' * 60}")
        print(f"EKSTRAKSI SELESAI")
        print(f"{'=' * 60}")
        print(f"Total video: {len(videos)}")
        print(f"Total frame diekstrak: {total_extracted}")
        print(f"Direktori output: {self.output_dir}")

        # Buat ringkasan
        self._create_summary(videos, total_extracted)
        return total_extracted

    def _create_summary(self, videos, total_frames):
        """
        Membuat ringkasan hasil ekstraksi.

        Args:
            videos: daftar video yang diproses
            total_frames: total frame yang diekstrak
        """
        summary = {
            "timestamp": datetime.now().isoformat(),
            "videos": [v.name for v in videos],
            "total_frames": total_frames,
            "output_dir": str(self.output_dir),
        }

        summary_path = self.output_dir / "extraction_summary.yaml"
        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.dump(summary, f, default_flow_style=False, allow_unicode=True)

        print(f"\n[INFO] Ringkasan tersimpan: {summary_path}")

    def split_to_trainval(self, source_dir=None, train_ratio=0.8, seed=42,
                          clean=False):
        """
        Split frame hasil ekstrak menjadi train/val.

        Diteruskan ke ``DatasetPreparer.split_dataset()`` supaya hanya ada SATU
        implementasi split di project, dan split-nya group-aware (blok 30
        frame). Versi lama membagi gambar satuan acak dan menyentuh file
        label kosong untuk semua gambar, termasuk yang belum dianotasi.

        Args:
            source_dir: direktori sumber frame (butuh images/ + labels/),
                        default ``data/annotated``
            train_ratio: rasio data train (0-1)
            seed: seed shuffle
            clean: hapus isi folder train/val tujuan
        """
        from dataset_prepare import DatasetPreparator

        source = Path(source_dir) if source_dir else (DATA_DIR / "annotated")
        if not source.is_absolute():
            source = DATA_DIR / source
        if not (source / "images").is_dir():
            print(f"[ERROR] Folder gambar tidak ditemukan: {source / 'images'}")
            print("Struktur yang dibutuhkan: <sumber>/images dan <sumber>/labels")
            return 1

        preparator = DatasetPreparator(self.config)
        try:
            preparator.split_dataset(source, train_ratio=train_ratio,
                                     seed=seed, clean=clean)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] {exc}")
            return 1
        return 0


def main() -> int:
    """Fungsi utama untuk menjalankan script ekstraksi frame dari command line."""
    parser = argparse.ArgumentParser(
        description="Ekstrak frame dari video untuk dataset deteksi kendaraan"
    )
    parser.add_argument("--action", type=str, default="extract",
                        choices=["list", "info", "extract", "extract-all", "split"],
                        help="Aksi yang akan dilakukan")
    parser.add_argument("--config", type=str, default=None,
                        help="path config.yaml (default: config/config.yaml)")
    parser.add_argument("--video", type=str,
                        help="Path file video")
    parser.add_argument("--interval", type=int, default=30,
                        help="Ekstrak setiap N frame (default: 30)")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Frame maksimum per video")
    parser.add_argument("--output", type=str,
                        help="Direktori output")
    parser.add_argument("--resize", type=int, nargs=2, default=None,
                        metavar=("WIDTH", "HEIGHT"), help="Resize frame")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="Rasio split train")
    parser.add_argument("--seed", type=int, default=42, help="Seed shuffle split")
    parser.add_argument("--clean", action="store_true",
                        help="Hapus isi folder train/val sebelum split")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    extractor = VideoFrameExtractor(config)

    # Jalankan aksi sesuai pilihan
    if args.action == "list":
        return 0 if extractor.list_videos() else 1

    if args.action == "info":
        if not args.video:
            print("[ERROR] --video wajib untuk action info")
            return 1
        info = extractor.get_video_info(args.video)
        if not info:
            print(f"[ERROR] Video tidak dapat dibuka: {args.video}")
            return 1
        print(f"\nInfo Video:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        return 0

    if args.action == "extract":
        if not args.video:
            print("[ERROR] --video wajib untuk action extract")
            return 1
        files = extractor.extract_frames(
            args.video,
            output_dir=args.output,
            interval=args.interval,
            max_frames=args.max_frames,
            resize=tuple(args.resize) if args.resize else None,
        )
        return 0 if files else 1

    if args.action == "extract-all":
        total = extractor.extract_all_videos(
            interval=args.interval,
            max_frames_per_video=args.max_frames,
            resize=tuple(args.resize) if args.resize else None,
        )
        return 0 if total else 1

    # action == "split"
    return extractor.split_to_trainval(args.output, args.ratio,
                                        args.seed, args.clean)


if __name__ == "__main__":
    raise SystemExit(main())
