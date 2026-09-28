"""
Pengumpulan dataset dari webcam atau video.

Versi lama file ini tidak punya CLI sama sekali (``main()`` hanya print
satu baris), jadi perintahnya di README tidak pernah bisa dijalankan. Nama
file dari webcam juga memakai ``%H%M%S`` sehingga mengambil 100 frame dalam
1 detik menghasilkan 100 nama yang bentrok dan menimpa file satu sama
lain.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Nama frame versi lama hanya sampai detik, jadi 100 frame dalam 1 detik
# saling menimpa. Nama sekarang selalu berakhir dengan indeks.
LEGACY_STAMP_NAME = re.compile(r"^\d{8}_\d{6}$")


def _check_interval(interval) -> int:
    interval = int(interval)
    if interval < 1:
        raise ValueError("interval minimal 1 (interval 0 akan membuat "
                         "pembagian dengan nol)")
    return interval


def collect_from_webcam(output_dir, num_frames=100, interval=1,
                        source=0, prefix="webcam") -> int:
    """
    Ambil ``num_frames`` gambar dari webcam.

    ``interval`` = jumlah frame webcam yang dilewati di antara dua
    penyimpanan. Tanpa interval, 100 frame dari kamera 30 FPS hanya
    membedakan kendaraan sejauh 3.3 detik - praktis semua gambarnya
    identik dan hanya menambah data redundan.
    """
    interval = _check_interval(interval)
    if num_frames < 1:
        raise ValueError("num_frames minimal 1")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Tidak dapat membuka webcam")

    print(f"Mengumpulkan {num_frames} gambar dari webcam "
          f"(setiap {interval} frame)...")

    saved = 0
    read = 0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        while saved < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            read += 1
            if (read - 1) % interval:
                continue
            # Indeks dijamin membuat nama unik walau timestamp sama.
            path = output_dir / f"{prefix}_{stamp}_{saved:05d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
            if saved % 10 == 0:
                print(f"  Progress: {saved}/{num_frames}")
    finally:
        cap.release()

    print(f"Selesai! {saved} gambar tersimpan di: {output_dir}")
    return saved


def collect_from_video(source, output_dir, interval=30, prefix=None,
                       start=0, duration=None) -> int:
    """
    Ekstrak frame dari video setiap ``interval`` frame.

    Frame yang berdekatan nyaris identik, jadi menyimpan tiap frame
    menghasilkan banyak data duplikat yang membuat val set tidak
    independen. ``interval`` yang wajar (mis. 30 = sekali per detik pada
    30 FPS) penting untuk menjaga kualitas dataset.
    """
    source = str(source)
    interval = _check_interval(interval)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    prefix = prefix or Path(source).stem
    print(f"Video: {source}  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  {fps:.1f} FPS  "
          f"{total} frame")
    if interval < fps:
        print(f"[PERINGATAN] interval {interval} < {fps:.0f} FPS, jadi jarak "
              f"antar gambar < 1 detik. Frame berdekatan/highlight bisa "
              f"membocor ke split lain.")

    saved = 0
    idx = 0
    end = None if duration is None else start + int(duration * fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if pos < start:
                continue
            if end is not None and pos >= end:
                break
            if idx % interval == 0:
                cv2.imwrite(str(output_dir / f"{prefix}_{idx:06d}.jpg"), frame)
                saved += 1
                if saved % 20 == 0:
                    print(f"  {saved} gambar (frame {int(pos)})")
            idx += 1
    finally:
        cap.release()

    print(f"Selesai! {saved} gambar tersimpan di: {output_dir}")
    return saved


def list_existing(directory) -> int:
    d = Path(directory)
    images = sorted(p for p in d.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS) \
        if d.is_dir() else []
    if not images:
        print(f"Belum ada gambar di {d}")
        return 0
    total_bytes = sum(p.stat().st_size for p in images)
    print(f"{len(images)} gambar di {d} ({total_bytes / 1e6:.1f} MB)")
    print(f"  pertama: {images[0].name}")
    print(f"  terakhir: {images[-1].name}")

    # Hanya nama versi lama (timestamp sampai detik) yang rawan menimpa
    # file. Nama berindeks seperti merged_000010.jpg itu wajar dan tidak
    # boleh dilaporkan sebagai tabrakan.
    legacy = [p.name for p in images if LEGACY_STAMP_NAME.match(p.stem)]
    if legacy:
        print(f"\n[PERINGATAN] {len(legacy)} file memakai penamaan lama "
              f"(timestamp hanya sampai detik) dan rawan saling menimpa.")
        print("  Contoh: " + ", ".join(legacy[:3]))
    return len(images)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pengumpulan dataset dari webcam dan video")
    parser.add_argument("--action", default="list",
                        choices=["list", "webcam", "video"],
                        help="list: ringkasan folder, webcam: capture, "
                             "video: ekstrak frame")
    parser.add_argument("--output-dir", default="data/raw",
                        help="Folder tujuan gambar")
    parser.add_argument("--source", default="0",
                        help="webcam index (default 0) atau path video")
    parser.add_argument("--num-frames", type=int, default=100,
                        help="Jumlah gambar dari webcam")
    parser.add_argument("--interval", type=int, default=1,
                        help="Lewati N-1 frame di antara dua penyimpanan")
    parser.add_argument("--start", type=int, default=0,
                        help="Frame awal (untuk video)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Durasi dalam detik (untuk video)")
    args = parser.parse_args()

    try:
        if args.action == "list":
            list_existing(args.output_dir)
        elif args.action == "webcam":
            src = 0 if args.source in ("0", 0) else args.source
            collect_from_webcam(args.output_dir, args.num_frames,
                                args.interval, source=src)
        else:
            if args.source in ("0", 0):
                print("[ERROR] --source wajib diisi untuk action video")
                return 1
            collect_from_video(args.source, args.output_dir,
                               args.interval, start=args.start,
                               duration=args.duration)
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
