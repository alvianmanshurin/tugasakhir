"""
Alur dataset otomatis - menghubungkan extract, auto-annotate, split,
validate, yaml, dan train dalam satu perintah.

    python main.py workflow            # stage + annotate + validate + yaml
    python main.py workflow --rebuild  # + split --clean (menimpa train/val)
    python main.py workflow --extract  # + ekstrak frame dari video dulu
    python main.py workflow --train    # + training di akhir
    python main.py workflow --stage annotate
    python main.py workflow --list

Urutan tahap dan kapan dijalankan:

    extract    hanya dengan --extract   ekstrak frame dari video config
    stage      selalu (awal)            salin frame pool -> data/staging/images
    annotate   selalu                   auto_annotate COCO -> data/staging/labels
    split      hanya dengan --rebuild   split group-aware -> train/val (--clean)
    validate   selalu                   cek label & kelas (read-only)
    yaml       selalu                   tulis ulang data/dataset.yaml
    train      hanya dengan --train     training YOLO

Kenapa ada ``data/staging``: folder ``data/annotated`` berisi label hasil
review MANUAL - alur ini tidak boleh menimpanya diam-diam. Tahap split
(membersihkan train/val) hanya jalan kalau diminta eksplisit lewat
``--rebuild``, dan staging bisa diulang tanpa menyentuh dataset final.
"""

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.paths import ROOT, load_config  # noqa: E402

STAGING_DIR = ROOT / "data" / "staging"
STAGING_IMAGES = STAGING_DIR / "images"
STAGING_LABELS = STAGING_DIR / "labels"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

STAGE_ORDER = ["extract", "stage", "annotate", "split", "validate",
               "yaml", "train"]

STAGE_INFO: Dict[str, str] = {
    "extract":   "ekstrak frame dari video (butuh --extract)",
    "stage":     "salin frame pool -> data/staging/images (idempoten)",
    "annotate":  "auto-annotate model COCO -> data/staging/labels",
    "split":     "split group-aware -> train/val (hanya --rebuild, MENIMPA)",
    "validate":  "validasi label & kelas (read-only)",
    "yaml":      "tulis ulang data/dataset.yaml",
    "train":     "training YOLO (butuh --train)",
}


# ----------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------

def _banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _list_images(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_pool(config: dict) -> Path:
    """
    Direktori sumber frame: ``data/raw/merged`` (pool gabungan 4 video)
    bila ada, fallback ke ``detection.source`` atau subdirektorinya.
    """
    candidates = [ROOT / "data" / "raw" / "merged"]
    source = config.get("detection", {}).get("source")
    if source:
        candidates.append(Path(source))
        candidates.extend(sorted(Path(source).iterdir())
                           if Path(source).is_dir() else [])

    for cand in candidates:
        if cand.is_dir() and _list_images(cand):
            return cand
    raise FileNotFoundError(
        "Pool frame tidak ditemukan (dicari: data/raw/merged, "
        f"{source}). Jalankan 'python main.py workflow --extract' "
        "atau 'python main.py extract --action extract-all' dulu.")


# ----------------------------------------------------------------------
# Tahapan
# ----------------------------------------------------------------------

def stage_extract(config: dict, opts) -> None:
    from extract_frames import VideoFrameExtractor

    extractor = VideoFrameExtractor(config)
    total = extractor.extract_all_videos(interval=opts.interval)
    if not total:
        raise RuntimeError("Ekstraksi frame gagal / tidak ada video.")


def stage_stage(config: dict, opts) -> None:
    pool = find_pool(config)
    images = _list_images(pool)
    if not images:
        raise FileNotFoundError(f"Tidak ada gambar di pool: {pool}")

    STAGING_IMAGES.mkdir(parents=True, exist_ok=True)
    STAGING_LABELS.mkdir(parents=True, exist_ok=True)

    copied = skipped = 0
    for img in images:
        dst = STAGING_IMAGES / img.name
        if dst.exists():
            skipped += 1
            continue
        shutil.copy2(img, dst)
        copied += 1
    print(f"[OK] Pool      : {pool}")
    print(f"[OK] Disalin   : {copied} gambar baru -> {STAGING_IMAGES}")
    print(f"[OK] Ada       : {skipped} gambar sudah di staging")


def stage_annotate(config: dict, opts) -> None:
    from auto_annotate import auto_annotate

    if not _list_images(STAGING_IMAGES):
        raise FileNotFoundError(
            f"Tidak ada gambar di {STAGING_IMAGES}. Jalankan tahap 'stage' "
            "dulu: python main.py workflow --stage stage")

    model = opts.model or config.get("model", {}).get("architecture") \
        or "yolo11n.pt"
    if not Path(model).is_absolute():
        model = str(ROOT / model)
    if not Path(model).is_file():
        raise FileNotFoundError(f"Model COCO tidak ditemukan: {model}")

    device = str(config.get("model", {}).get("device", "cpu"))
    auto_annotate(
        image_dir=STAGING_IMAGES,
        label_dir=STAGING_LABELS,
        model_name=model,
        conf_threshold=opts.conf,
        img_size=opts.imgsz,
        device=device,
        overwrite=opts.overwrite,
    )


def stage_split(config: dict, opts) -> None:
    from dataset_prepare import DatasetPreparator

    if not _list_images(STAGING_IMAGES):
        raise FileNotFoundError(
            f"Tidak ada gambar di {STAGING_IMAGES}. Jalankan tahap "
            "'stage' + 'annotate' dulu.")
    if not _list_images(STAGING_LABELS):
        raise FileNotFoundError(
            f"Tidak ada label di {STAGING_LABELS}. Jalankan tahap "
            "'annotate' dulu - split tanpa label akan membuat seluruh "
            "dataset jadi label kosong.")

    preparator = DatasetPreparator(config)
    preparator.split_dataset(STAGING_DIR, train_ratio=opts.ratio,
                             seed=opts.seed, clean=True)


def stage_validate(config: dict, opts) -> None:
    from dataset_prepare import DatasetPreparator

    DatasetPreparator(config).validate_dataset()


def stage_yaml(config: dict, opts) -> None:
    from dataset_prepare import DatasetPreparator

    out = ROOT / "data" / "dataset.yaml"
    DatasetPreparator(config).create_dataset_yaml(str(out))
    print(f"[OK] {out}")


def stage_train(config: dict, opts) -> None:
    import train

    train.train_model(config, batch=opts.batch, epochs=opts.epochs)


STAGE_FUNCS: Dict[str, Callable] = {
    "extract": stage_extract,
    "stage": stage_stage,
    "annotate": stage_annotate,
    "split": stage_split,
    "validate": stage_validate,
    "yaml": stage_yaml,
    "train": stage_train,
}


# ----------------------------------------------------------------------
# Rencana & eksekusi
# ----------------------------------------------------------------------

def build_plan(opts) -> List[str]:
    if opts.stage:
        return [opts.stage]
    plan: List[str] = []
    if opts.extract:
        plan.append("extract")
    plan += ["stage", "annotate"]
    if opts.rebuild:
        plan.append("split")
    plan += ["validate", "yaml"]
    if opts.train:
        plan.append("train")
    return plan


def print_plan(plan: List[str]) -> None:
    print("[INFO] Rencana tahap:")
    for i, name in enumerate(plan, 1):
        print(f"  {i}. {name:<9} - {STAGE_INFO[name]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Alur dataset otomatis: stage -> annotate -> split -> "
                    "validate -> yaml -> train")
    parser.add_argument("--config", default=None,
                        help="path config.yaml (default: config/config.yaml)")
    parser.add_argument("--stage", choices=STAGE_ORDER,
                        help="jalankan SATU tahap ini saja")
    parser.add_argument("--extract", action="store_true",
                        help="tambah tahap ekstrak frame dari video")
    parser.add_argument("--rebuild", action="store_true",
                        help="tambah tahap split --clean (MENIMPA train/val "
                             "dan label review manual!)")
    parser.add_argument("--train", action="store_true",
                        help="tambah tahap training di akhir")
    parser.add_argument("--list", action="store_true",
                        help="tampilkan rencana tahap lalu keluar")

    ann = parser.add_argument_group("annotate")
    ann.add_argument("--model", default=None,
                     help="path model COCO (default: model.architecture)")
    ann.add_argument("--conf", type=float, default=0.35,
                     help="confidence auto-annotate (default: 0.35)")
    ann.add_argument("--imgsz", type=int, default=640,
                     help="imgsz auto-annotate (default: 640)")
    ann.add_argument("--overwrite", action="store_true",
                     help="timpa label staging yang sudah ada")

    ext = parser.add_argument_group("extract")
    ext.add_argument("--interval", type=int, default=30,
                     help="ekstrak setiap N frame (default: 30)")

    spl = parser.add_argument_group("split")
    spl.add_argument("--ratio", type=float, default=0.8,
                     help="rasio train (default: 0.8)")
    spl.add_argument("--seed", type=int, default=42,
                     help="seed shuffle split (default: 42)")

    trn = parser.add_argument_group("train")
    trn.add_argument("--epochs", type=int, default=None,
                     help="jumlah epoch (mengikuti config bila kosong)")
    trn.add_argument("--batch", type=int, default=None,
                     help="batch size (mengikuti config bila kosong)")
    opts = parser.parse_args()

    plan = build_plan(opts)

    if opts.list:
        print("Tahap workflow (default: stage, annotate, validate, yaml)")
        print("-" * 60)
        for name in STAGE_ORDER:
            flag = "AKTIF" if name in plan else "-"
            print(f"  [{flag:^5}] {name:<9} - {STAGE_INFO[name]}")
        print("-" * 60)
        print("Flag: --extract (tambah extract), --rebuild (tambah split),")
        print("      --train (tambah train), --stage <tahap> (satu tahap).")
        return 0

    print_plan(plan)
    config = load_config(Path(opts.config) if opts.config else None)

    for name in plan:
        _banner(f"TAHAP: {name.upper()}")
        started = time.perf_counter()
        try:
            STAGE_FUNCS[name](config, opts)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"\n[ERROR] Tahap '{name}' gagal: {exc}")
            return 1
        elapsed = time.perf_counter() - started
        print(f"[OK] Tahap '{name}' selesai ({elapsed:.1f} detik)")

    _banner("WORKFLOW SELESAI")
    print(f"  Tahap dijalankan : {', '.join(plan)}")
    print(f"  Staging          : {STAGING_DIR}")
    print("  Tahap berikutnya : review label staging, lalu "
          "'python main.py workflow --rebuild' untuk split final.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
