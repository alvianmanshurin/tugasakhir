"""
Utilitas path & konfigurasi bersama.

Semua entry point resolving path relatif ke file config, bukan ke current
working directory. Dengan begitu script bisa dijalankan dari direktori
manapun tanpa membuat database/config kedua yang kosong.
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# utils/ -> src/ -> root project
ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT / "src"
CONFIG_PATH = ROOT / "config" / "config.yaml"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"
DB_PATH = DATA_DIR / "detections.db"

_CACHE: Dict[str, Dict[str, Any]] = {}


def ensure_src_on_path() -> None:
    """Pastikan ``src`` bisa di-import dari entry point mana pun."""
    src = str(SRC_DIR)
    if src not in sys.path:
        sys.path.insert(0, src)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Muat config.yaml.

    Semua path di dalam config dinormalisasi menjadi path absolut terhadap
    root project, sehingga aman dipakai dari direktori kerja mana pun.
    """
    path = Path(config_path) if config_path else CONFIG_PATH
    path = path if path.is_absolute() else (ROOT / path)

    if not path.is_file():
        raise FileNotFoundError(f"Config tidak ditemukan: {path}")

    cache_key = str(path)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Isi config tidak valid (harus mapping): {path}")

    _resolve_paths(config)
    _CACHE[cache_key] = config
    return config


def _resolve_paths(config: Dict[str, Any]) -> None:
    """Ubah path relatif di config menjadi absolut terhadap ROOT."""
    model_cfg = config.setdefault("model", {})
    if model_cfg.get("best_weights"):
        model_cfg["best_weights"] = _abs(model_cfg["best_weights"])

    dataset_cfg = config.setdefault("dataset", {})
    for key in ("yaml_path", "train_images", "val_images", "test_images",
                "train_labels", "val_labels", "test_labels"):
        if dataset_cfg.get(key):
            dataset_cfg[key] = _abs(dataset_cfg[key])

    detection_cfg = config.setdefault("detection", {})
    for key in ("source", "output_dir"):
        if detection_cfg.get(key):
            detection_cfg[key] = _abs(detection_cfg[key])

    evaluation_cfg = config.setdefault("evaluation", {})
    if evaluation_cfg.get("output_dir"):
        evaluation_cfg["output_dir"] = _abs(evaluation_cfg["output_dir"])


def _abs(value: str) -> str:
    p = Path(value)
    return str(p if p.is_absolute() else (ROOT / p))


def get_model_path(config: Optional[Dict[str, Any]] = None,
                   explicit: Optional[str] = None) -> str:
    """
    Path bobot YOLOv11 hasil training.

    Tidak ada fallback ke ``yolo11n.pt`` (COCO 80 kelas): memakai model
    COCO diam-diam akan melabeli person sebagai "motor" dan car sebagai
    "bus" karena ID kelas 0-3 bertabrakan. Model dilatih wajib ada.
    """
    if explicit:
        path = _abs(explicit) if not Path(explicit).is_absolute() else explicit
        if not Path(path).is_file():
            raise FileNotFoundError(f"Model tidak ditemukan: {path}")
        return path

    cfg = config if config is not None else load_config()
    model_cfg = cfg.get("model", {})

    candidates = []
    if model_cfg.get("best_weights"):
        candidates.append(Path(model_cfg["best_weights"]))
    if model_cfg.get("project_dir"):
        candidates.append(Path(model_cfg["project_dir"]) / "weights" / "best.pt")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    searched = ", ".join(str(c) for c in candidates) or "(tidak ada best_weights di config)"
    raise FileNotFoundError(
        "Model hasil training tidak ditemukan.\n"
        f"  Dicari: {searched}\n"
        "  Jalankan: python src/train.py\n"
        "  Atau set 'model.best_weights' di config/config.yaml"
    )


def get_class_names(config: Optional[Dict[str, Any]] = None) -> Dict[int, str]:
    """
    Mapping class id -> nama.

    Diambil dari config ``dataset.names``; fallback ke
    ``config/predefined_classes.txt``.
    """
    cfg = config if config is not None else load_config()
    names = cfg.get("dataset", {}).get("names")

    if names:
        return {int(k): str(v) for k, v in names.items()}

    classes_file = ROOT / "config" / "predefined_classes.txt"
    if classes_file.is_file():
        with open(classes_file, "r", encoding="utf-8") as f:
            listed = [line.strip() for line in f if line.strip()]
        if listed:
            return {i: name for i, name in enumerate(listed)}

    raise ValueError(
        "Nama kelas tidak ditemukan. Set 'dataset.names' di config/config.yaml "
        "atau isi config/predefined_classes.txt"
    )


def resolve_roi_boundary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Koordinat ROI apa adanya (belum di-resolve ke frame tertentu)."""
    return dict(config.get("roi", {}).get("boundary", {}))
