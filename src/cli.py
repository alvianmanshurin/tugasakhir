"""
CLI terpadu untuk seluruh proyek.

Semua modul di ``src/`` tetap bisa dijalankan langsung seperti dulu
(``python src/train.py``). Dispatcher ini hanya menambah SATU pintu masuk
yang memanggil ``main()`` masing-masing modul:

    python main.py <command> [args...]
    python -m src <command> [args...]

Argumen setelah command diteruskan apa adanya ke argparse modul tersebut,
jadi ``python main.py train --quick`` sama persis dengan
``python src/train.py --quick``.

Modul di-import saat command dipanggil (bukan saat startup), supaya
``python main.py --help`` tetap instan tanpa memuat ultralytics/torch.
"""

import importlib
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# command -> (modul di src/, keterangan)
COMMANDS: Dict[str, Tuple[str, str]] = {
    "annotate":     ("auto_annotate",
                     "Auto-label gambar dengan model COCO"),
    "annot-helper": ("annotation_helper",
                     "Template LabelMe / konversi LabelMe ke YOLO"),
    "batch":        ("batch_process",
                     "Batch deteksi folder, simpan label YOLO"),
    "collect":      ("dataset_collect",
                     "Kumpulkan data (webcam / video)"),
    "cctv":         ("cctv_connect",
                     "Scan, uji, dan jalankan deteksi CCTV RTSP"),
    "compare":      ("comparison",
                     "Bandingkan beberapa laporan evaluasi"),
    "dataset":      ("dataset_prepare",
                     "Validasi / split / yaml dataset"),
    "detect":       ("detect",
                     "Deteksi kendaraan pada gambar / folder"),
    "detect-track": ("detect_with_tracking",
                     "Deteksi + tracking + hitung kendaraan"),
    "evaluate":     ("evaluate",
                     "Evaluasi model (mAP, FPS, confusion matrix)"),
    "export":       ("export_model",
                     "Export model (ONNX, TFLite, TorchScript, ...)"),
    "extract":      ("extract_frames",
                     "Ekstrak frame dari video dataset"),
    "gui":          ("gui_app",
                     "GUI aplikasi deteksi kendaraan"),
    "monitor":      ("monitor",
                     "Alias realtime.py"),
    "pipeline":     ("pipeline",
                     "Pipeline deteksi + tracking + counting (video/CCTV)"),
    "query-db":     ("query_db",
                     "Query database hasil deteksi"),
    "realtime":     ("realtime",
                     "Deteksi real-time (webcam)"),
    "train":        ("train",
                     "Training model YOLO"),
    "workflow":     ("workflow",
                     "Alur dataset otomatis: stage -> annotate -> "
                     "split -> validate -> yaml -> train"),
}

# Alias lama -> command resmi
ALIASES: Dict[str, str] = {
    "help": "help",
    "auto-annotate": "annotate",
    "prepare": "dataset",
    "count": "pipeline",
}


def print_help() -> None:
    print("CLI terpadu - Vehicle Counting System (UPT K3L ITERA)")
    print("\nPemakaian:")
    print("  python main.py <command> [args...]")
    print("  python -m src <command> [args...]")
    print("\nCommand:")
    width = max(len(c) for c in COMMANDS)
    for cmd, (_, desc) in COMMANDS.items():
        print(f"  {cmd:<{width}}  {desc}")
    print("\nCommand lain:")
    print(f"  {'--version':<{width}}  versi paket")
    print(f"  {'help':<{width}}  tampilkan bantuan ini")
    print("\nArgumen setelah command diteruskan ke modulnya, misal:")
    print("  python main.py train --quick")
    print("  python main.py annotate --image-dir data/annotated/images/val "
          "--label-dir data/annotated/labels/val")
    print("  python main.py workflow --list")
    print("\nCara lama (masih didukung):")
    print("  python src/train.py --quick")


def _load_version() -> str:
    try:
        from src import __version__
        return __version__
    except Exception:
        return "?"


def main(argv: Optional[list] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] in ("--version", "-V"):
        print(f"vehicle-counting {_load_version()}")
        return 0

    if not argv or argv[0] in ("-h", "--help", "help"):
        print_help()
        return 0

    cmd = argv[0]
    cmd = ALIASES.get(cmd, cmd)
    entry = COMMANDS.get(cmd)
    if entry is None:
        print(f"[ERROR] Command tidak dikenal: {argv[0]}")
        print("        Jalankan 'python main.py --help' untuk daftar command.")
        return 1

    module_name, _ = entry
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # dependency rusak / modul gagal import
        print(f"[ERROR] Gagal memuat modul '{module_name}': {exc}")
        return 1

    fn = getattr(module, "main", None)
    if not callable(fn):
        print(f"[ERROR] Modul '{module_name}' tidak punya fungsi main().")
        return 1

    # Argumen modul dibaca dari sys.argv (semua modul memakai argparse
    # default), jadi teruskan apa adanya dengan penanda command aslinya.
    sys.argv = [f"main.py {cmd}"] + argv[1:]
    try:
        result = fn()
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        return code if isinstance(code, int) else 1
    except KeyboardInterrupt:
        print("\n[INFO] Dibatalkan pengguna.")
        return 130
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
