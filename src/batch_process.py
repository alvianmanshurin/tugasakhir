"""
Batch processing gambar (deteksi + annotasi) untuk sekumpulan file.

Versi lama file ini tidak punya CLI (``main()`` hanya print), memakai
``models/yolov8n_vehicle/weights/best.pt`` yang tidak pernah ada, dan
menghitung label kelas dari konstanta yang bisa tidak cocok dengan model.

Sekarang memakai ``VehiclePipeline`` supaya:
- model diambil dari config / ``runs/detect/models/...``,
- threshold sama dengan evaluasi (0.25/0.5),
- letterbox, bukan resize distort,
- ROI ikut diterapkan kalau diaktifkan.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from pipeline import VehiclePipeline
from utils.paths import load_config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: Path):
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def batch_process(input_dir, output_dir, config=None, model_path=None,
                  recursive=False, prefix="detected_", save_labels=False):
    """
    Proses semua gambar di ``input_dir`` dan simpan gambar ber-annotasi.

    Memakai ``VehiclePipeline.process_image()`` - tracking dilewati karena
    tiap gambar berdiri sendiri, dan ``ObjectTracker`` hanya mengonfirmasi
    track setelah beberapa frame berturut-turut sehingga lewat
    ``process_frame()`` setiap gambar akan menghasilkan 0 box.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Bukan direktori: {input_dir}")

    if recursive:
        images = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    else:
        images = list_images(input_dir)

    if not images:
        raise FileNotFoundError(f"Tidak ada gambar di {input_dir}")

    pipeline = VehiclePipeline(config, model_path=model_path)
    print(f"[INFO] {len(images)} gambar -> {output_dir}")

    label_dir = output_dir / "labels"
    if save_labels:
        label_dir.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[PERINGATAN] Gagal baca: {img_path}")
            continue

        annotated, dets, info = pipeline.process_image(img)
        total_boxes += len(dets)

        out_path = output_dir / f"{prefix}{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

        if save_labels and dets:
            h, w = img.shape[:2]
            lines = []
            for t in dets:
                x1, y1, x2, y2 = t["bbox"]
                xc = ((x1 + x2) / 2) / w
                yc = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{t['class_id']} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            (label_dir / f"{img_path.stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")

        if idx % 20 == 0 or idx == len(images):
            print(f"  [{idx}/{len(images)}] {total_boxes} box total")

    print(f"[OK] Selesai. {total_boxes} box. Hasil: {output_dir}")
    return {"images": len(images), "boxes": total_boxes, "output_dir": str(output_dir)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch processing gambar dengan VehiclePipeline")
    parser.add_argument("--input-dir", required=True, help="Direktori gambar")
    parser.add_argument("--output-dir", required=True, help="Direktori output")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--recursive", action="store_true",
                        help="Cari gambar di subfolder juga")
    parser.add_argument("--prefix", default="detected_",
                        help="Prefix nama file output")
    parser.add_argument("--save-labels", action="store_true",
                        help="Sekalian simpan label YOLO untuk verifikasi manual")
    args = parser.parse_args()

    try:
        batch_process(
            args.input_dir, args.output_dir,
            config=load_config(args.config) if args.config else load_config(),
            model_path=args.model, recursive=args.recursive,
            prefix=args.prefix, save_labels=args.save_labels,
        )
    except (NotADirectoryError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
