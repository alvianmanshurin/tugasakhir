# FLOW DIAGRAM - Vehicle Detection System
# Sistem Perhitungan Jumlah Kendaraan Bermotor
# Studi Kasus: UPT K3L ITERA

> **Catatan status diagram ini.** Alur di bawah masih menggambarkan kondisi
> awal proyek. Angka metrik di dalamnya (mAP50 71.6%, 64.7%, dan seterusnya)
> berasal dari evaluation report lama yang tidak bisa direproduksi ulang
> karena `evaluate.py` saat itu salah membaca `save_dir` dan nama file kurva.
> **Jangan mengutip angka itu.**
>
> Untuk alur dan metrik yang benar, lihat `WORKFLOW.md`. Ringkasan terbaru:
> split 564 train / 71 val / 70 test, dan hasil `python src/evaluate.py --task all`
> adalah mAP50 0.3850, mAP50-95 0.2993, Precision 0.6027, Recall 0.4121,
> F1 0.4850, FPS 32.9.

---

## 1. Main Workflow (Updated)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VEHICLE DETECTION SYSTEM                             │
│                    UPT K3L ITERA - TUGAS AKHIR                              │
│                    Status: [PIPELINE SELESAI, MODEL PERLU DIPERBAIKAN]     │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │   VIDEO      │
    │   SOURCE     │
    │  (4 videos)  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   EXTRACT    │
    │   FRAMES     │  ← extract_frames.py
    │  (705 imgs)  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   SPLIT      │
    │ TRAIN/VAL/   │  ← 564 train, 71 val, 70 test
    │    TEST      │     (split lama: acak per gambar, bukan group-aware)
    │(564/71/70)   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  AUTO-       │  ← auto_annotate.py (YOLOv11 COCO)
    │  ANNOTATE    │     1097 boxes detected
    │  (5 min)     │     387 train, 110 val labeled
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   TRAIN      │
    │   MODEL      │  ← YOLOv11n, 50 epoch, CPU
    │  (30-40 min) │     output: runs/detect/models/vehicle_detection/
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  EVALUATE    │  ← evaluate.py --task all
    │ mAP50: 38.5% │     Precision: 60.3%, Recall: 41.2%
    │ (terukur)    │    FPS: 32.9
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────────────────────┐
    │            DETECTION PIPELINE            │
    │  ┌──────────────────────────────────┐    │
    │  │  YOLOv11 Detection (imgsz=416)    │    │
    │  └──────────────────────────────────┘    │
    │  ┌──────────────────────────────────┐    │
    │  │  ROI Filter (20m road boundary)  │    │
    │  └──────────────────────────────────┘    │
    │  ┌──────────────────────────────────┐    │
    │  │  Object Tracking (ByteTrack)     │    │
    │  └──────────────────────────────────┘    │
    │  ┌──────────────────────────────────┐    │
    │  │  Line Crossing Counter           │    │
    │  └──────────────────────────────────┘    │
    └──────────────────────────────────────────┘
```

---

## 2. Detection + ROI + Tracking Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DETECTION + ROI + TRACKING PIPELINE                      │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │                  INPUT SOURCE                       │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
    │  │  Image  │ │ Video   │ │ Webcam  │ │ Folder  │    │
    │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │
    └───────┼───────────┼───────────┼───────────┼─────────┘
            │           │           │           │
            └───────────┴─────┬─────┴───────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  LOAD MODEL     │
                    │  best.pt        │
                    │  (YOLOv11n)      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  YOLO INFERENCE │
                    │  (imgsz=416)    │
                    │  (conf=0.5)     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  PARSE RESULTS  │
                    │  - bbox         │
                    │  - class_id     │
                    │  - confidence   │
                    └────────┬────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │      ROI FILTER         │
                │  ┌───────────────────┐  │
                │  │ Trapezoid Polygon │  │
                │  │ (150,120)─────────│──│──(490,120)  20m
                │  │     │             │  │             (jauh)
                │  │     │    ROAD     │  │
                │  │     │   AREA      │  │
                │  │     │             │  │
                │  │ (0,416)───────────│──│──(640,416)  0m
                │  └───────────────────┘  │             (dekat)
                │                         │
                │  Filter:                │
                │  - Inside polygon?      │
                │  - Distance <= 20m?     │
                │  - Min bbox height?     │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │    OBJECT TRACKER       │
                │  (ByteTrack-inspired)   │
                │                         │
                │  - Center distance      │
                │  - Class matching       │
                │  - Track lifecycle:     │
                │    new → tentative →    │
                │    confirmed → lost     │
                │                         │
                │  Output:                │
                │  - track_id             │
                │  - velocity             │
                │  - age, hits            │
                └────────────┬────────────┘
                             │
                             ▼
                 ┌─────────────────────────┐
                 │  DUAL-LINE COUNTER      │
                 │                         │
                 │  ═══════════════════════│  ← Line 1 (atas, biru)
                 │                         │
                 │  ZONE COUNTING          │
                 │                         │
                 │  ═══════════════════════│  ← Line 2 (bawah, kuning)
                 │                         │
                 │  Count when track       │
                 │  crosses BOTH lines     │
                 │  (above line1 AND       │
                 │   below line2)          │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │    DRAW RESULTS         │
                 │  - Bounding boxes       │
                 │  - Track IDs            │
                 │  - Class labels         │
                 │  - Velocity arrows      │
                 │  - HUD (FPS, counts)    │
                 │  - ROI overlay          │
                 │  - Dual counting lines  │
                └────────────┬────────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │       OUTPUT            │
                │  - Annotated frame      │
                │  - Statistics dict      │
                │  - Video file           │
                └─────────────────────────┘
```

---

## 3. ROI Distance Estimation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ROI DISTANCE MAPPING                                     │
└─────────────────────────────────────────────────────────────────────────────┘

    Camera Position (0m)
    ┌─────────────────────────────────────┐
    │             ╱╲                      │
    │            ╱  ╲                     │
    │           ╱    ╲                    │
    │          ╱      ╲                   │
    │         ╱  5m    ╲                  │
    │        ╱──────────╲                 │
    │       ╱            ╲                │
    │      ╱    10m       ╲               │
    │     ╱────────────────╲              │
    │    ╱                  ╲             │
    │   ╱      15m           ╲            │
    │  ╱──────────────────────╲           │
    │ ╱                        ╲          │
    │╱          20m             ╲         │
    ╱────────────────────────────╲        │
    │                             │       │
    │←─────── Road Width ────────→│       │
    │                             │       │
    └─────────────────────────────────────┘

    Pixel Y-coordinate → Distance Mapping:
    ─────────────────────────────────────
    y = 416 (bottom)  → 0m   (near camera)
    y = 312            → 5m
    y = 208            → 10m
    y = 164            → 15m
    y = 120 (top)      → 20m  (far)

    Formula:
    distance = (roi_bottom - y) / (roi_bottom - roi_top) * max_distance
```

---

## 4. Object Tracking Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OBJECT TRACKING STATE MACHINE                            │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐
    │   NEW       │  ← First detection
    │  (hits=1)   │
    └──────┬──────┘
           │ Matched again
           ▼
    ┌─────────────┐
    │ TENTATIVE   │  ← hits < 3
    │  (hits=2)   │
    └──────┬──────┘
           │ hits >= 3
           ▼
    ┌─────────────┐
    │ CONFIRMED   │  ← Active tracking
    │  (valid)    │     shows ID, counts
    └──────┬──────┘
           │ Not matched
           ▼
    ┌─────────────┐
    │   LOST      │  ← time_since_update > 0
    │  (aging)    │
    └──────┬──────┘
           │ time_since_update > 30
           ▼
    ┌─────────────┐
    │  REMOVED    │  ← Track deleted
    │  (dead)     │
    └─────────────┘

    Matching Criteria:
    ────────────────────
    - Center distance < 80px
    - Same class_id (or penalty +500)
    - time_since_update <= 5
```

---

## 5. Training Results

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING METRICS (Epoch 39)                              │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │                    PROGRESS BAR                     │
    │  Epoch: 39/50 [████████████████████████░░░░] 78%    │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │              TRAINING CURVES                        │
    │                                                     │
    │  Loss    ┤╲                                         │
    │  (box)   ┤ ╲╲                                       │
    │          ┤  ╲╲╲╲                                    │
    │          ┤      ╲╲╲╲╲╲╲                             │
    │          ┤              ╲╲╲╲╲╲╲╲╲╲                  │
    │          └──────────────────────────→ Epoch         │
    │                                                     │
    │  mAP50   ┤              ╱╲                          │
    │          ┤           ╱╲╱  ╲╱╲                       │
    │          ┤        ╱╲╱          ╲╱╲                  │
    │          ┤     ╱╲╱                  ╲╱╲             │
    │          ┤  ╱╲╱                          ╲          │
    │          └──────────────────────────→ Epoch         │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │  METRICS LAMA - TIDAK DAPAT DIREPRODUKSI           │
    │  Lihat outputs/evaluation/evaluation_report.json    │
    │  untuk angka yang benar.                            │
    │  ┌─────────────────────────────────────────────┐    │
    │  │  mAP50:        71.6%  (best at epoch 39)    │    │
    │  │  mAP50-95:     54.1%                        │    │
    │  │  Precision:    59.5%                        │    │
    │  │  Recall:       79.3%                        │    │
    │  │  Box Loss:     0.834                        │    │
    │  │  Cls Loss:     0.815                        │    │
    │  │  DFL Loss:     0.829                        │    │
    │  └─────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────┘
```

---

## 6. Evaluation Results (angka lama - ganti dengan report terbaru)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│       EVALUATION RESULTS (ANGKA LAMA - SUDAH TIDAK VALID)                   │
│       sumber: evaluation report lama, evaluate.py lama salah baca save_dir  │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │           CONFUSION MATRIX (Normalized)             │
    │                                                     │
    │              Predicted                              │
    │           motor  mobil   bus   truk                 │
    │  motor  [ 0.72   0.05   0.02  0.01 ]  ← 80% correct │
    │  mobil  [ 0.04   0.88   0.01  0.02 ]  ← 88% correct │
    │  bus    [ 0.05   0.03   0.50  0.08 ]  ← 50% correct │
    │  truk   [ 0.03   0.06   0.04  0.42 ]  ← 42% correct │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │           PER-CLASS METRICS                         │
    │                                                     │
    │  Class     mAP50    Recall    Precision    Support  │
    │  ─────────────────────────────────────────────────  │
    │  motor     77.0%    81.7%     63.8%        82       │
    │  mobil     85.4%    93.6%     67.7%       141       │
    │  bus       52.8%    66.7%     57.1%         6       │
    │  truk      43.7%    62.5%     60.0%        24       │
    │  ─────────────────────────────────────────────────  │
    │  ALL       64.7%    76.1%     62.2%       253       │
    │                                                     │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │  PER-CLASS TERUKUR (evaluate.py --task all)         │
    │                                                     │
    │  Class     AP50     AP50-95   Precision  Recall  F1  │
    │  ─────────────────────────────────────────────────  │
    │  mobil     0.6922   0.5597    0.9184     0.7031  0.7965│
    │  motor     0.4529   0.3054    0.6923     0.5455  0.6102│
    │  truk      0.3950   0.3322    0.8000     0.4000  0.5333│
    │  bus       0.0000   0.0000    0.0000     0.0000  0.0000│
    │  ─────────────────────────────────────────────────  │
    │  ALL       0.3850   0.2993    0.6027     0.4121  0.4850│
    │                                                     │
    │  Catatan: bus 0.0 karena hanya 10 instance bus di   │
    │  train (1.19%). Prioritas: tambah anotasi bus,     │
    │  lalu split group-aware, lalu training ulang.       │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │           SPEED BENCHMARK                           │
    │                                                     │
    │  Preprocess:   0.5 ms                               │
    │  Inference:   31.4 ms                               │
    │  Postprocess:  0.4 ms                               │
    │  ─────────────────────                              │
    │  Total:       32.3 ms/frame  →  ~31 FPS             │
    │                                                     │
    │  Hardware: Intel i3-1115G4 @ 3.00GHz (CPU only)     │
    └─────────────────────────────────────────────────────┘
```

---

## 7. Command Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMMAND REFERENCE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

SETUP
─────────────────────────────────────────────────────────────
pip install -r requirements.txt          # Install dependencies
pip install labelImg                     # Install LabelImg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

DATASET
─────────────────────────────────────────────────────────────
python src/extract_frames.py --action extract-all      # Extract frames
python src/extract_frames.py --action split            # Split train/val
python src/dataset_prepare.py --action validate        # Validate

AUTO-ANNOTATION
─────────────────────────────────────────────────────────────
python src/auto_annotate.py \
  --image-dir data/annotated/images/train \
  --label-dir data/annotated/labels/train \
  --conf 0.35

TRAINING
─────────────────────────────────────────────────────────────
python -c "from ultralytics import YOLO; model = YOLO('yolov11n.pt'); ..."

EVALUATION
─────────────────────────────────────────────────────────────
python -c "from ultralytics import YOLO; model = YOLO('best.pt'); model.val(...)"

DETECTION + TRACKING
─────────────────────────────────────────────────────────────
python src/detect_with_tracking.py --source 0 --show                  # Webcam
python src/detect_with_tracking.py --source video.mp4 --show          # Video
python src/detect_with_tracking.py --source 0 --show --no-roi         # No ROI
python src/detect_with_tracking.py --source 0 --show --roi-max-dist 15 # 15m max
python src/detect_with_tracking.py --source 0 --show --no-track       # No tracking

OLD DETECTION (without ROI/tracking)
─────────────────────────────────────────────────────────────
python src/realtime.py --source 0 --show                 # Webcam (basic)
python src/detect.py --source image.jpg                  # Single image
```

---

## 8. Project Structure (Updated)

```
tugasakhir/
│
├─── config/
│    ├── config.yaml              ← Config (ROI + Tracking included)
│    └── predefined_classes.txt   ← Class list for LabelImg
│
├─── data/
│    ├── dataset.yaml             ← YOLO dataset config
│    │
│    ├── raw/                     ← Video frames
│    │   ├── KIRI-7/              (174 frames)
│    │   ├── KIRI-9/              (181 frames)
│    │   ├── TENGAH-7/            (179 frames)
│    │   ├── TENGAH-9/            (171 frames)
│    │   └── merged/              (705 frames)
│    │
    │    └── annotated/               ← Labeled dataset
    │        ├── images/
    │        │   ├── train/           (564 images)
    │        │   ├── val/             (71 images)
    │        │   └── test/            (70 images)
    │        └── labels/             (1:1 dengan images, label kosong = background)
│
├─── src/
│    ├── auto_annotate.py         ★ NEW - Auto-annotation
│    ├── detect_with_tracking.py  ★ NEW - Detection + ROI + Tracking
│    │
│    ├── train.py                 ← Training script
│    ├── detect.py                ← Basic detection
│    ├── evaluate.py              ← Evaluation
│    ├── realtime.py              ← Basic webcam
│    ├── gui_app.py               ← GUI
│    ├── pipeline.py              ← Pipeline
│    ├── extract_frames.py        ← Frame extraction
│    ├── comparison.py            ← YOLO vs Manual
│    ├── monitor.py               ← Training monitor
│    │
│    └── utils/
│        ├── counter.py           ← Line crossing counter
│        ├── roi_filter.py        ★ NEW - ROI 20m filter
│        ├── tracker.py           ★ NEW - Object tracker
│        ├── visualizer.py        ← Drawing utilities
│        └── metrics.py           ← Metrics calculation
│
├─── models/
│    └── vehicle_detection/        ← (di dalam runs/detect/models/)
│
├─── runs/detect/models/vehicle_detection/
│    ├── weights/
│    │   ├── best.pt              ★ Best model (PyTorch)
│    │   ├── last.pt              ← Last checkpoint
│    │   └── best.torchscript     ← Exported TorchScript
│    ├── results.csv
│    ├── confusion_matrix.png
│    └── BoxF1_curve.png
│
├─── outputs/
│    ├── inference_val/           ← Inference results
│    └── evaluation/
│        └── evaluation_report.json
│
├── WORKFLOW.md                   ← UPDATED
├── FLOW_DIAGRAM.md              ← UPDATED (this file)
├── README.md
└── requirements.txt
```

---

## 9. Timeline (Updated)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROJECT TIMELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Minggu 1: Dataset Preparation
    ═══════════════════════════════════════════════════════════
    [████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░]
    - Extract frames dari video (705 frames)
    - Split train/val/test (564/71/70, split lama acak per gambar)
    - Auto-annotate dengan YOLOv11 COCO (1097 boxes)
    ✓ SELESAI

    Minggu 2: Training & Evaluation
    ═══════════════════════════════════════════════════════════
    [████████████████████████████████████████░░░░░░░░░░░░░░░░]
    - Training YOLOv11n (50 epoch, ~30 menit, CPU)
    - Evaluasi model (mAP50 terukur 38.5%, Precision 60.3%, Recall 41.2%)
    - Buat ROI filter (boundary trapezoid 1920x1080)
    - Buat object tracker (ByteTrack)
    ✓ SELESAI

    Minggu 3: Pipeline & Testing
    ═══════════════════════════════════════════════════════════
    [████████████████████████████████████████████████████████████]
    - Pipeline terunifikasi (deteksi → ROI → tracking → counting)
    - Database + GUI + CCTV connector
    - Tes otomatis 256 assertion
    ✓ SELESAI

    Minggu 4: Perbaikan Kualitas Model
    ═══════════════════════════════════════════════════════════
    [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████]
    - Tambah anotasi bus (sekarang hanya 10 instance di train)
    - Regenerate split group-aware (blok 30 frame)
    - Training ulang + evaluasi ulang
    - Kalibrasi ROI dengan rekaman gerbang asli

    Minggu 5: Deployment
    ═══════════════════════════════════════════════════════════
    [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
    - Deploy di gate UPT K3L ITERA
    - Validasi di lapangan
    - Dokumentasi akhir
    - Presentasi
```

---

## 10. Hardware Specification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HARDWARE SPECIFICATION                              │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │              LAPTOP SPECIFICATION                   │
    │  ┌─────────────────────────────────────────────┐    │
    │  │  CPU: Intel Core i3-1115G4 @ 3.00GHz        │    │
    │  │  Cores: 2 physical, 4 logical               │    │
    │  │  RAM: 8 GB                                  │    │
    │  │  GPU: Intel UHD Graphics (Integrated)       │    │
    │  │  OS: Windows 11 Home                        │    │
    │  └─────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │              OPTIMIZED SETTINGS                     │
    │  ┌─────────────────────────────────────────────┐    │
    │  │  Model: YOLOv11n (Nano, 2.6M params)         │    │
    │  │  Image Size: 416x416                        │    │
    │  │  Batch Size: 4                              │    │
    │  │  Device: CPU                                │    │
    │  │  Epochs: 50                                 │    │
    │  │  Training Time: ~30 menit                   │    │
    │  │  Inference FPS: 32.9 (deteksi saja)          │    │
    │  │  Pipeline FPS: 16-22 (ROI+tracking)         │    │
    │  │  ROI: trapezoid 1920x1080 (~13.8% area)     │    │
    │  │  Tracker: ByteTrack-inspired                │    │
    │  └─────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────┘
```
