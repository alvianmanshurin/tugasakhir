"""
Auto-annotation gambar dengan model pretrained YOLO11.

PERINGATAN PENTING: pemetaan kelas di bawah hanya valid untuk model COCO
(80 kelas). Kalau diberi model yang SUDAH dilatih untuk proyek ini
(``best.pt``, 4 kelas), pemetaan ini SALAH - model itu sudah mengembalikan
0=motor, 1=mobil, 2=bus, 3=truk secara langsung, sehingga kelas 2 (bus)
akan dipetakan lagi menjadi 2 (bus) tetapi kelas 3 (truk) menjadi 0 (motor).
Karena itu ``--model`` di sini hanya boleh model COCO; model proyek
ditolak eksplisit.

Fitur:
- Tidak pernah menimpa label yang sudah ada tanpa ``--overwrite``.
- Box di luar rentang 0-1 di-clamp, box degenerate dibuang.
- Mendukung camouflage: bus/truk besar sering salah diklasifikasikan
  sebagai car oleh COCO, jadi bisa dipetakan manual lewat ``--swap``.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO

from utils.paths import get_class_names, load_config

# COCO -> proyek. Hanya berlaku untuk model COCO 80 kelas.
COCO_TO_PROJECT = {
    3: 0,   # motorcycle -> motor
    2: 1,   # car        -> mobil
    5: 2,   # bus        -> bus
    7: 3,   # truck      -> truk
}

# Nama COCO yang ditampilkan di log
COCO_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def check_model_is_coco(model) -> tuple:
    """
    Pastikan model adalah model COCO, bukan model proyek.

    Dicek dari NAMA kelas, bukan jumlah kelas: model proyek punya 4 kelas
    dengan nama motor/mobil/bus/truk, sedangkan model COCO punya 80 kelas
    bernama person/bicycle/car/... Jadi jumlah kelas saja tidak cukup
    untuk membedakan keduanya.

    Returns:
        (n_classes, model_names) dari model.

    Raises:
        ValueError kalau model ternyata sudah dilatih untuk kelas proyek.
    """
    names = model.names
    n = len(names)

    project_names = {str(v).lower() for v in get_class_names().values()}
    model_names = {str(v).lower() for v in names.values()} if isinstance(
        names, dict) else {str(v).lower() for v in names}

    # Model proyek dikenali dari nama kelasnya, bukan dari jumlah kelas.
    if project_names and project_names == model_names:
        raise ValueError(
            f"Model ini memakai kelas proyek {sorted(project_names)} - "
            "jadi model yang SUDAH dilatih untuk proyek ini, bukan model COCO.\n"
            "Model proyek sudah mengembalikan ID kelas yang benar "
            "(0=motor, 1=mobil, 2=bus, 3=truk), jadi pemetaan COCO akan "
            "merusak label (mis. truck -> motor).\n"
            "Gunakan model COCO untuk auto-annotation, mis: --model yolo11n.pt"
        )

    if n <= 4:
        raise ValueError(
            f"Model cuma punya {n} kelas ({sorted(model_names)}), "
            "kemungkinan bukan model COCO.\n"
            "Auto-annotation butuh model COCO (mis. yolo11n.pt, 80 kelas) "
            "yang bisa mengenali motor/mobil/bus/truk."
        )
    return n, names


def clamp_box(x1, y1, x2, y2, w, h):
    """
    Clamp bbox ke gambar, kembalikan (x_c, y_c, w, h) ternormalisasi.

    Returns None kalau box jadi degenerate setelah clamp (w atau h = 0),
    karena box seperti itu tidak berguna saat training.
    """
    x1 = max(0.0, min(float(x1), w))
    x2 = max(0.0, min(float(x2), w))
    y1 = max(0.0, min(float(y1), h))
    y2 = max(0.0, min(float(y2), h))

    if x2 <= x1 or y2 <= y1:
        return None

    return (
        ((x1 + x2) / 2) / w,
        ((y1 + y2) / 2) / h,
        (x2 - x1) / w,
        (y2 - y1) / h,
    )


def auto_annotate(
    image_dir,
    label_dir,
    model_name: str = "yolo11n.pt",
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    img_size: int = 640,
    device: str = "cpu",
    overwrite: bool = False,
    swap: Optional[Dict[int, int]] = None,
    min_box_fraction: float = 0.0005,
) -> dict:
    """
    Anotasi gambar memakai model pretrained YOLO11.

    Args:
        image_dir: direktori gambar
        label_dir: direktori output label YOLO
        model_name: WAJIB model COCO (mis. yolo11n.pt)
        conf_threshold: ambang confidence
        iou_threshold: ambang IoU untuk NMS
        img_size: imgsz untuk inferensi
        device: "cpu" atau "0"
        overwrite: timpa label yang sudah ada (default: lindungi)
        swap: koreksi manual, mis. {5: 3} untuk memaksa bus menjadi truk
        min_box_fraction: buang box yang luasnya < nilai ini (0.0005 = 0.05% gambar)

    Returns:
        dict ringkasan
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_files = list_images(image_dir)
    if not image_files:
        raise FileNotFoundError(f"Tidak ada gambar di {image_dir}")

    print(f"[INFO] {len(image_files)} gambar di {image_dir}")
    print(f"[INFO] Model: {model_name}  conf={conf_threshold}  imgsz={img_size}")

    model = YOLO(model_name)
    check_model_is_coco(model)
    project_names = get_class_names()  # id -> nama

    swap = {int(k): int(v) for k, v in (swap or {}).items()}
    if swap:
        print(f"[INFO] Override manual kelas COCO: {swap}")

    total_boxes = 0
    empty = 0
    protected = 0
    degenerate = 0
    class_counts: Counter = Counter()

    for i, img_path in enumerate(image_files, 1):
        lbl_file = label_dir / (img_path.stem + ".txt")

        if lbl_file.exists() and not overwrite:
            protected += 1
            continue

        results = model.predict(
            source=str(img_path), imgsz=img_size, conf=conf_threshold,
            iou=iou_threshold, device=device, verbose=False,
        )

        lines = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            img_h, img_w = r.orig_shape

            for box in r.boxes:
                coco_cls = int(box.cls[0])
                proj_cls = swap.get(coco_cls, COCO_TO_PROJECT.get(coco_cls))
                if proj_cls is None:
                    continue  # kelas COCO lain (person, dll) -> abaikan

                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                norm = clamp_box(x1, y1, x2, y2, img_w, img_h)
                if norm is None:
                    degenerate += 1
                    continue
                xc, yc, w, h = norm
                if w * h < min_box_fraction:
                    degenerate += 1
                    continue

                lines.append(f"{proj_cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                class_counts[proj_cls] += 1
                total_boxes += 1

        lbl_file.write_text("\n".join(lines) + ("\n" if lines else ""),
                            encoding="utf-8")
        if not lines:
            empty += 1

        if i % 50 == 0 or i == len(image_files):
            print(f"  [{i}/{len(image_files)}] {total_boxes} box")

    print("\n" + "=" * 60)
    print("AUTO-ANNOTATION SELESAI")
    print("=" * 60)
    print(f"  Gambar ditemukan : {len(image_files)}")
    print(f"  Box ditulis      : {total_boxes}")
    print(f"  Label kosong     : {empty} (tidak ada kendaraan)")
    print(f"  Label dilindungi : {protected} (sudah ada, --overwrite untuk menimpa)")
    print(f"  Box dibuang      : {degenerate} (degenerate / terlalu kecil)")
    print("\n  Per kelas:")
    for cls_id in sorted(project_names):
        name = project_names[cls_id]
        print(f"    {cls_id} ({name}): {class_counts[cls_id]}")
    print(f"\n  Label tersimpan di: {label_dir}")
    if not overwrite:
        print("\n  CATATAN: label lama tidak ditimpa. Jalankan ulang dengan")
        print("  --overwrite untuk menulis ulang semuanya, atau periksa manual")
        print("  label yang dilindungi.")

    return {
        "images": len(image_files), "boxes": total_boxes, "empty": empty,
        "protected": protected, "discarded": degenerate,
        "by_class": dict(class_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-annotate gambar dengan model pretrained YOLO11 (COCO)")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--model", default="yolo11n.pt",
                        help="WAJIB model COCO, mis. yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true",
                        help="timpa label yang sudah ada")
    parser.add_argument("--swap", default=None,
                        help="koreksi kelas manual, mis. '5:3' untuk bus->truk")
    args = parser.parse_args()

    try:
        valid_classes = set(get_class_names())
        swap = {}
        if args.swap:
            for part in args.swap.split(","):
                if ":" not in part:
                    print(f"[ERROR] Format --swap harus 'coco:proyek', dapat: {part}")
                    return 1
                src, dst = part.split(":", 1)
                try:
                    src_id, dst_id = int(src), int(dst)
                except ValueError:
                    print(f"[ERROR] --swap harus angka 'coco:proyek', dapat: {part}")
                    return 1
                if dst_id not in valid_classes:
                    print(f"[ERROR] Kelas proyek {dst_id} tidak valid "
                          f"(pilihan: {sorted(valid_classes)})")
                    return 1
                swap[src_id] = dst_id

        auto_annotate(
            image_dir=args.image_dir, label_dir=args.label_dir,
            model_name=args.model, conf_threshold=args.conf,
            iou_threshold=args.iou, img_size=args.img_size, device=args.device,
            overwrite=args.overwrite, swap=swap,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
