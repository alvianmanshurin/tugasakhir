"""
Script monitoring real-time - ALIAS dari ``realtime.py``.

Versi lama file ini punya implementasi sendiri yang:
- memuat ``models/yolov8n_vehicle/weights/best.pt`` (path tidak pernah ada;
  model sebenarnya ada di ``runs/detect/models/vehicle_detection/weights/``),
- memakai conf 0.35 yang berbeda dari threshold di config (0.5),
- tidak punya tracker/counter, jadi tidak ada hitungan kendaraan.

Menjalankan file ini sekarang sama dengan ``python src/realtime.py`` agar
tidak ada dua hasil berbeda dari "real-time detection".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from realtime import main

if __name__ == "__main__":
    print("[INFO] monitor.py = alias realtime.py")
    raise SystemExit(main())
