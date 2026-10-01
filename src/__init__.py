"""PENGEMBANGAN SUB SISTEM PERHITUNGAN JUMLAH KENDARAAN BERMOTOR BERBASIS PENGOLAHAN CITRA
Studi Kasus: UPT K3L ITERA

Import ``src`` (mis. ``import src.train``) me-bootstrap ``src/`` ke
``sys.path`` supaya import flat (``from utils.paths import ...``) konsisten
baik saat modul dijalankan langsung (``python src/train.py``) maupun lewat
CLI terpadu (``python main.py <command>`` / ``python -m src``).
"""

import sys
from pathlib import Path

__version__ = "1.0.0"
__author__ = "Penelitian Tugas Akhir"

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
