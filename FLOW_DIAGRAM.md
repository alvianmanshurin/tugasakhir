# FLOW DIAGRAM - Vehicle Detection System
# Sistem Perhitungan Jumlah Kendaraan Bermotor
# Studi Kasus: UPT K3L ITERA

> **Catatan status diagram ini.** Alur di bawah masih menggambarkan kondisi
> awal proyek. Angka metrik tertua di dalamnya (mAP50 71.6%, 64.7%, dst.)
> berasal dari evaluation report yang tidak bisa direproduksi ulang karena
> `evaluate.py` saat itu salah membaca `save_dir` dan nama file kurva.
> **Jangan mengutip angka itu.**
>
> Untuk alur dan metrik yang benar, lihat `WORKFLOW.md`. Ringkasan terbaru
> (03 Okt 2026): split 2045 train / 480 val / 292 test, dan hasil
> `python src/evaluate.py --task all --split test` adalah mAP50 **0.6965**,
> mAP50-95 **0.4670**, Precision 0.6562, Recall 0.7387, F1 0.6668
> (`conf>=0.001`); pada ambang operasional `conf>=0.5` confusion matrix
> test memberi P 0.8950 / R 0.6368.

---

## 1. Main Workflow (Updated)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VEHICLE DETECTION SYSTEM                             │
│                    UPT K3L ITERA - TUGAS AKHIR                              │
│                  Status: [PIPELINE SELESAI, MODEL ULANG - TINGGAL BUS]      │
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
    │  Angka benar: outputs/evaluation/                   │
    │  evaluation_report_test.json (run 03 Okt 2026).     │
    │  mAP50 0.6965, mAP50-95 0.4670, F1 0.6668.         │
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

## 6. Evaluation Results (run 03 Okt 2026)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  EVALUATION RESULTS - YOLO11n, 20 epoch, dataset 2817 frame                 │
│  sumber: outputs/evaluation/evaluation_report_test.json                     │
│  conf>=0.001 (mAP standar), imgsz 416, split test (292 gambar, 669 instans) │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │  OVERALL (test set, conf>=0.001)                    │
    │                                                     │
    │  mAP50        0.6965   target 0.75   belum          │
    │  mAP50-95     0.4670   target 0.50   belum          │
    │  Precision    0.6562   target 0.70   belum          │
    │  Recall       0.7387   target 0.70   TERCAPAI       │
    │  F1           0.6668   target 0.70   belum          │
    │  FPS          28.4     target > 5    TERCAPAI       │
    │                                                     │
    │  val (480 gambar): 0.6972 / 0.4530 / 0.6162 /       │
    │                    0.7631 / 0.6656                  │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │  CONFUSION MATRIX - conf>=0.5 (ambang operasional)  │
    │  baris = label sebenarnya, kolom = prediksi         │
    │                                                     │
    │            motor  mobil   bus   truk  (bg)          │
    │  motor   [ 0.61   0.00   0.00  0.00   0.39 ]        │
    │  mobil   [ 0.00   0.68   0.00  0.00   0.32 ]        │
    │  bus     [ 0.00   0.00   0.45  0.00   0.55 ]        │
    │  truk    [ 0.00   0.00   0.00  0.43   0.57 ]        │
    │  (bg)    [ 0.30   0.40   0.22  0.08  ----- ] ← FP   │
    │                                                     │
    │  TP 426   FP 50   FN 243   P 0.8950   R 0.6368      │
    │  Tidak ada salah kelas antar jenis kendaraan -      │
    │  semua galat adalah objek terlewat (FN) dan objek    │
    │  latar yang terpanggil kendaraan (FP).              │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │  PER-CLASS (test, conf>=0.001)                      │
    │                                                     │
    │  Class    AP50    AP50-95  Precis.  Recall   F1     │
    │  ────────────────────────────────────────────────   │
    │  mobil    0.8986  0.6369   0.8566   0.7532  0.8016  │
    │  motor    0.8495  0.4650   0.7950   0.7840  0.7895  │
    │  truk     0.6329  0.4593   0.6683   0.5992  0.6319  │
    │  bus      0.4050  0.3070   0.3047   0.8182  0.4441  │
    │  ────────────────────────────────────────────────   │
    │  ALL      0.6965  0.4670   0.6562   0.7387  0.6668  │
    │                                                     │
    │  Catatan: bus tetap terlemah - recall 0.82 dengan   │
    │  precision 0.30. Penyebab: hanya 99 instans bus     │
    │  (~2% dari 2817 frame). Prioritas: tambah anotasi   │
    │  bus, lalu training ulang.                          │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │           SPEED BENCHMARK                           │
    │                                                     │
    │  End-to-end (50 gambar 1920x1080 -> 416):           │
    │      35.2 ms/frame  ->  28.4 FPS                    │
    │      (3 pengukuran pada hari yang sama: 28-33 FPS)  │
    │  ─────────────────────                              │
    │  Loop val Ultralytics (per gambar):                 │
    │      0.5 + 18.3 + 0.4 ms  ~  19 ms                  │
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
│    │   ├── KIRI-7/              (695 frames)
│    │   ├── KIRI-9/              (723 frames)
│    │   ├── TENGAH-7/            (716 frames)
│    │   ├── TENGAH-9/            (683 frames)
│    │   └── merged/              (2817 frames)
│    │
    │    └── annotated/               ← Labeled dataset
    │        ├── images/
    │        │   ├── train/           (2045 images)
    │        │   ├── val/             (480 images)
    │        │   └── test/            (292 images)
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
│        ├── evaluation_report.json       ← run terakhir (= test)
│        ├── evaluation_report_val.json
│        ├── evaluation_report_test.json
│        ├── evaluation_metrics.csv       (+ _val.csv / _test.csv)
│        ├── evaluation_per_class.csv     (+ _val.csv / _test.csv)
│        ├── confusion_matrix_val.png
│        ├── confusion_matrix_test.png
│        └── PR/F1/P/R_curve.png
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
    - Extract frames dari video (2817 frames, interval 15)
    - Split train/val/test (2045/480/292, group-aware, leakage 0.0%)
    - Auto-annotate dengan YOLO11 COCO (4835 boxes)
    ✓ SELESAI

    Minggu 2: Training & Evaluation
    ═══════════════════════════════════════════════════════════
    [████████████████████████████████████████░░░░░░░░░░░░░░░░]
    - Training YOLOv11n (20 epoch, 4 jam 10 menit, CPU)
    - Evaluasi model (mAP50 69.7% test, Precision 65.6%, Recall 73.9%)
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
    [████████████████████████████████████████████████████░░░░]
    - Regenerate split group-aware (blok 30 frame)          ✓ 02 Okt
    - Kalibrasi ROI dengan rekaman gerbang asli             ✓ 03 Okt
    - Training ulang 20 epoch + evaluasi dua split          ✓ 03 Okt
    - Tambah anotasi bus (99 instance, target 150-200)      ⏳

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
     │  │  ROI: trapezoid 1920x1080 (~16.7% area)     │    │
    │  │  Tracker: ByteTrack-inspired                │    │
    │  └─────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────┘
```
