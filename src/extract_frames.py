"""
Ekstraksi Frame Video untuk Dataset Deteksi Kendaraan
Ekstrak frame dari video KIRI/TENGAH di gerbang masuk Kampus ITERA
"""

import os
import cv2
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime


def load_config(config_path="config/config.yaml"):
    """Memuat file konfigurasi YAML dari path yang diberikan."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class VideoFrameExtractor:
    """
    Ekstraktor frame video untuk pembuatan dataset.
    
    Fitur:
    - List semua video yang tersedia
    - Dapatkan informasi video (FPS, resolusi, durasi)
    - Ekstrak frame dengan interval tertentu
    - Split frame menjadi train/val
    """

    def __init__(self, config):
        """
        Inisialisasi ekstraktor frame.
        
        Args:
            config: dict konfigurasi dari config.yaml
        """
        self.config = config
        self.video_source = config["dataset"]["video_source"]
        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_videos(self):
        """
        List semua video yang tersedia di direktori sumber.
        
        Returns:
            Daftar path video yang ditemukan
        """
        video_dir = Path(self.video_source)
        if not video_dir.exists():
            print(f"[ERROR] Direktori video tidak ditemukan: {video_dir}")
            return []

        # Cari file video (.MOV dan .mp4)
        videos = list(video_dir.glob("*.MOV")) + list(video_dir.glob("*.mp4"))
        print(f"\n[INFO] Ditemukan {len(videos)} video di: {video_dir}")
        for v in videos:
            size_mb = v.stat().st_size / (1024*1024)
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
        if output_dir:
            out_dir = Path(output_dir)
        else:
            # Buat subdirektori berdasarkan nama video
            out_dir = self.output_dir / video_path.stem

        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"EKSTRAKSI FRAME")
        print(f"{'='*60}")
        print(f"Video: {video_path.name}")

        # Dapatkan info video
        info = self.get_video_info(video_path)
        if info:
            print(f"Resolusi: {info['width']}x{info['height']}")
            print(f"FPS: {info['fps']:.1f}")
            print(f"Durasi: {info['duration']:.1f}s")
            print(f"Total frame: {info['total_frames']}")

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
                cv2.imwrite(str(save_path), frame)
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
        """
        videos = self.list_videos()

        if not videos:
            print("[ERROR] Tidak ada video ditemukan")
            return

        print(f"\n[INFO] Memproses {len(videos)} video...")
        print(f"[INFO] Interval: setiap {interval} frame")
        print(f"[INFO] Frame maks per video: {max_frames_per_video}")

        total_extracted = 0

        for video in videos:
            print(f"\n{'='*60}")
            print(f"Memproses: {video.name}")
            print(f"{'='*60}")

            files = self.extract_frames(
                video,
                interval=interval,
                max_frames=max_frames_per_video,
                resize=resize,
            )
            total_extracted += len(files)

        print(f"\n{'='*60}")
        print(f"EKSTRAKSI SELESAI")
        print(f"{'='*60}")
        print(f"Total video: {len(videos)}")
        print(f"Total frame diekstrak: {total_extracted}")
        print(f"Direktori output: {self.output_dir}")

        # Buat ringkasan
        self._create_summary(videos, total_extracted)

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
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        print(f"\n[INFO] Ringkasan tersimpan: {summary_path}")

    def split_to_trainval(self, source_dir=None, train_ratio=0.8):
        """
        Split frame hasil ekstrak menjadi train/val.
        
        Args:
            source_dir: direktori sumber frame
            train_ratio: rasio data train (0-1)
        """
        import random
        import shutil

        source = Path(source_dir) if source_dir else self.output_dir
        images = list(source.glob("*.jpg"))

        if not images:
            print(f"[ERROR] Tidak ada gambar ditemukan di: {source}")
            return

        print(f"\n[SPLIT] Total gambar: {len(images)}")
        print(f"[SPLIT] Rasio train: {train_ratio}")

        # Acak gambar
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)

        train_files = images[:split_idx]
        val_files = images[split_idx:]

        # Buat direktori
        train_img = Path("data/annotated/images/train")
        val_img = Path("data/annotated/images/val")
        train_lbl = Path("data/annotated/labels/train")
        val_lbl = Path("data/annotated/labels/val")

        for d in [train_img, val_img, train_lbl, val_lbl]:
            d.mkdir(parents=True, exist_ok=True)

        # Salin file
        for f in train_files:
            shutil.copy2(f, train_img / f.name)
            (train_lbl / (f.stem + ".txt")).touch()

        for f in val_files:
            shutil.copy2(f, val_img / f.name)
            (val_lbl / (f.stem + ".txt")).touch()

        print(f"[OK] Train: {len(train_files)} gambar -> {train_img}")
        print(f"[OK] Val: {len(val_files)} gambar -> {val_img}")


def main():
    """Fungsi utama untuk menjalankan script ekstraksi frame dari command line."""
    parser = argparse.ArgumentParser(
        description="Ekstrak frame dari video untuk dataset deteksi kendaraan"
    )
    parser.add_argument("--action", type=str, default="extract",
                       choices=["list", "info", "extract", "extract-all", "split"],
                       help="Aksi yang akan dilakukan")
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
    args = parser.parse_args()

    config = load_config()
    extractor = VideoFrameExtractor(config)

    # Jalankan aksi sesuai pilihan
    if args.action == "list":
        extractor.list_videos()

    elif args.action == "info":
        if args.video:
            info = extractor.get_video_info(args.video)
            if info:
                print(f"\nInfo Video:")
                for k, v in info.items():
                    print(f"  {k}: {v}")
        else:
            print("[ERROR] Silakan tentukan --video")

    elif args.action == "extract":
        if args.video:
            extractor.extract_frames(
                args.video,
                output_dir=args.output,
                interval=args.interval,
                max_frames=args.max_frames,
                resize=tuple(args.resize) if args.resize else None,
            )
        else:
            print("[ERROR] Silakan tentukan --video")

    elif args.action == "extract-all":
        extractor.extract_all_videos(
            interval=args.interval,
            max_frames_per_video=args.max_frames,
            resize=tuple(args.resize) if args.resize else None,
        )

    elif args.action == "split":
        extractor.split_to_trainval(args.output, args.ratio)


if __name__ == "__main__":
    main()
