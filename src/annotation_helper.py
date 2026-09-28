"""
Helper anotasi: template LabelMe dan konversi ke format YOLO.

Versi lama file ini tidak punya CLI, dan ``create_labelme_template()``
menulis ``imageHeight/imageWidth = 0``. File seperti itu tidak bisa
dikonversi: pembaginya nol, dan hasil konversi di
``dataset_prepare.convert_labelme_to_yolo()`` ditolak sebagai "ukuran
gambar tidak valid". Templatenya sendiri tidak berguna.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from utils.paths import get_class_names

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_size(image_path) -> tuple:
    """Baca ukuran gambar, atau (0, 0) kalau tidak bisa."""
    img = cv2.imread(str(image_path))
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    return int(w), int(h)


def create_labelme_template(image_path, output_path=None) -> Path:
    """
    Buat template LabelMe untuk satu gambar, lengkap dengan ukuran asli.

    LabelMe butuh ``imageWidth``/``imageHeight`` yang benar supaya template
    bisa langsung dipakai; nilai 0 membuat file tidak bisa dikonversi.
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    w, h = _read_size(image_path)
    if w == 0 or h == 0:
        raise ValueError(f"Gambar tidak bisa dibaca: {image_path}")

    output_path = Path(output_path) if output_path else image_path.with_suffix(
        ".json")

    template = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(template, indent=2), encoding="utf-8")
    print(f"Template dibuat: {output_path} ({w}x{h})")
    return output_path


def create_templates(image_dir, output_dir=None, limit=None) -> list:
    """Buat template untuk semua gambar di ``image_dir``."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Bukan direktori: {image_dir}")

    images = sorted(p for p in image_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit:
        images = images[:limit]
    if not images:
        raise FileNotFoundError(f"Tidak ada gambar di {image_dir}")

    output_dir = Path(output_dir) if output_dir else image_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []
    for img in images:
        created.append(create_labelme_template(img, output_dir / f"{img.stem}.json"))

    print(f"\n[OK] {len(created)} template dibuat di {output_dir}")
    print("Langkah berikutnya: buka dengan LabelMe, lalu konversi dengan")
    print("  python src/dataset_prepare.py --action labelme --input <folder json>")
    return created


def convert_labelme_to_yolo(json_path, output_path, class_mapping) -> int:
    """
    Konversi satu file LabelMe ke YOLO. Returns jumlah baris.

    Versi lama tidak melakukan clamping, sehingga bounding box yang
    menjauh dari tepi gambar menghasilkan nilai di luar 0-1 dan label
    ditolak Ultralytics saat training.
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    image_w = int(data.get("imageWidth") or 0)
    image_h = int(data.get("imageHeight") or 0)
    if image_w <= 0 or image_h <= 0:
        raise ValueError(f"{json_path.name}: imageWidth/imageHeight "
                         "tidak valid - buka ulang di LabelMe lalu simpan")

    lines = []
    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in class_mapping:
            continue
        points = shape.get("points") or []
        if len(points) < 2:
            continue

        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        x1, x2 = max(0.0, min(min(xs), image_w)), min(image_w, max(xs))
        y1, y2 = max(0.0, min(min(ys), image_h)), min(image_h, max(ys))
        if x2 <= x1 or y2 <= y1:
            continue

        xc = ((x1 + x2) / 2) / image_w
        yc = ((y1 + y2) / 2) / image_h
        w = (x2 - x1) / image_w
        h = (y2 - y1) / image_h
        lines.append(f"{class_mapping[label]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""),
                           encoding="utf-8")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Helper anotasi: template LabelMe dan konversi YOLO")
    parser.add_argument("--action", default="template",
                        choices=["template", "templates", "convert"],
                        help="template: 1 gambar, templates: semua gambar, "
                             "convert: LabelMe -> YOLO")
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--image", default=None, help="Path satu gambar")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--json", default=None, help="File LabelMe (convert)")
    parser.add_argument("--output", default=None, help="Output .txt (convert)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        if args.action == "template":
            if not args.image:
                print("[ERROR] --image wajib untuk action template")
                return 1
            create_labelme_template(args.image, args.output)
        elif args.action == "templates":
            if not args.image_dir:
                print("[ERROR] --image-dir wajib untuk action templates")
                return 1
            create_templates(args.image_dir, args.output_dir, args.limit)
        else:
            if not args.json:
                print("[ERROR] --json wajib untuk action convert")
                return 1
            mapping = {name: int(cid)
                       for cid, name in get_class_names().items()}
            out = args.output or Path(args.json).with_suffix(".txt")
            n = convert_labelme_to_yolo(args.json, out, mapping)
            print(f"[OK] {n} baris ditulis ke {out}")
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
