# FLOW DIAGRAM - Vehicle Detection System
# Sistem Perhitungan Jumlah Kendaraan Bermotor
# Studi Kasus: UPT K3L ITERA

---

## 1. Main Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VEHICLE DETECTION SYSTEM                             │
│                    UPT K3L ITERA - TUGAS AKHIR                              │
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
    │  TRAIN/VAL   │  ← 80% train, 20% val
    │ (564/141)    │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  ANNOTATE    │
    │  (LabelImg)  │  ← Manual annotation
    │  YOLO format │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   TRAIN      │
    │   MODEL      │  ← train.py (YOLOv8n)
    │  (50 epochs) │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  EVALUATE    │  ← evaluate.py
    │  (mAP, FPS)  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │   DEPLOY     │
    │  (Detection) │  ← detect.py, realtime.py
    └──────────────┘
```

---

## 2. Dataset Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATASET PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │                  VIDEO SOURCE                        │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
    │  │ KIRI-7  │ │ KIRI-9  │ │TENGAH-7 │ │TENGAH-9 │  │
    │  │ .MOV    │ │ .MOV    │ │ .MOV    │ │ .MOV    │  │
    │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │
    │       │           │           │           │         │
    └───────┼───────────┼───────────┼───────────┼─────────┘
            │           │           │           │
            └───────────┴─────┬─────┴───────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  EXTRACT FRAMES │
                    │  interval=60    │
                    │  max=300/video  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  MERGE FRAMES   │
                    │  (705 total)    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  SPLIT 80/20    │
                    └───┬─────────┬───┘
                        │         │
                ┌───────┘         └───────┐
                ▼                         ▼
        ┌──────────────┐          ┌──────────────┐
        │  TRAIN SET   │          │   VAL SET    │
        │  (564 imgs)  │          │  (141 imgs)  │
        └──────┬───────┘          └──────┬───────┘
               │                         │
               ▼                         ▼
        ┌──────────────┐          ┌──────────────┐
        │  ANNOTATE    │          │  ANNOTATE    │
        │  (LabelImg)  │          │  (LabelImg)  │
        └──────┬───────┘          └──────┬───────┘
               │                         │
               ▼                         ▼
        ┌──────────────┐          ┌──────────────┐
        │ labels/train │          │  labels/val  │
        │   (.txt)     │          │    (.txt)    │
        └──────────────┘          └──────────────┘
```

---

## 3. Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │  DATASET.YAML   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  YOLOv8n.pt     │  ← Pretrained model
    │  (Nano, 3.2M)   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────────────────────────────┐
    │           TRAINING PROCESS              │
    │  ┌─────────────────────────────────┐    │
    │  │  epochs: 50                     │    │
    │  │  batch_size: 4                  │    │
    │  │  image_size: 416                │    │
    │  │  device: CPU                    │    │
    │  │  optimizer: SGD                 │    │
    │  └─────────────────────────────────┘    │
    └────────┬────────────────────────────────┘
             │
             ▼
    ┌─────────────────────────────────────────┐
    │           OUTPUT FILES                  │
    │  ┌──────────────┐ ┌──────────────┐     │
    │  │  best.pt     │ │  last.pt     │     │
    │  │  (best mAP)  │ │  (final)     │     │
    │  └──────────────┘ └──────────────┘     │
    │  ┌──────────────┐ ┌──────────────┐     │
    │  │ results.csv  │ │ confusion_   │     │
    │  │              │ │ matrix.png   │     │
    │  └──────────────┘ └──────────────┘     │
    └─────────────────────────────────────────┘
```

---

## 4. Detection Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DETECTION PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │                  INPUT SOURCE                        │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐  │
    │  │  Image  │ │ Video   │ │ Webcam  │ │ Folder  │  │
    │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘  │
    └───────┼───────────┼───────────┼───────────┼─────────┘
            │           │           │           │
            └───────────┴─────┬─────┴───────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  LOAD MODEL     │
                    │  best.pt        │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  YOLO INFERENCE │
                    │  (imgsz=416)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
      ┌──────────────┐ ┌──────────┐ ┌──────────────┐
      │  DETECTIONS  │ │  COUNTS  │ │  BOUNDING    │
      │  (boxes,     │ │  motor:  │ │  BOXES       │
      │   classes,   │ │  mobil:  │ │  (drawn on   │
      │   conf)      │ │  bus:    │ │   image)     │
      │              │ │  truk:   │ │              │
      └──────┬───────┘ └────┬─────┘ └──────┬───────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │  OUTPUT         │
                    │  - Image/Video  │
                    │  - JSON/CSV     │
                    │  - Console      │
                    └─────────────────┘
```

---

## 5. Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐     ┌─────────────────┐
    │  TRAINED MODEL  │     │  VALIDATION SET │
    │  (best.pt)      │     │  (141 images)   │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             └───────────┬───────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  MODEL.VAL()    │
                └────────┬────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  PRECISION   │ │   RECALL     │ │  F1-SCORE    │
│  (TP/TP+FP) │ │  (TP/TP+FN)  │ │  (2*P*R/P+R)│
└──────────────┘ └──────────────┘ └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     mAP50       │
                │  mAP50-95       │
                └────────┬────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   CONFUSION  │ │   PR CURVE   │ │  F1 CURVE    │
│   MATRIX     │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  FPS BENCHMARK  │
                │  (Inference     │
                │   Speed)        │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  EVALUATION     │
                │  REPORT         │
                │  (JSON/CSV)     │
                └─────────────────┘
```

---

## 6. Project Structure Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROJECT STRUCTURE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

tugasakhir/
│
├─── config/
│    ├── config.yaml              ← Konfigurasi proyek
│    └── predefined_classes.txt   ← Daftar kelas (LabelImg)
│
├─── data/
│    ├── dataset.yaml             ← Konfigurasi dataset YOLO
│    │
│    ├── raw/                     ← Video frames (original)
│    │   ├── KIRI-7/
│    │   ├── KIRI-9/
│    │   ├── TENGAH-7/
│    │   └── TENGAH-9/
│    │
│    └── annotated/               ← Dataset untuk training
│        ├── images/
│        │   ├── train/           ← 564 images
│        │   └── val/             ← 141 images
│        └── labels/
│            ├── train/           ← .txt files (LabelImg)
│            └── val/             ← .txt files (LabelImg)
│
├─── src/
│    ├── train.py                 ← Training script
│    ├── detect.py                ← Detection script
│    ├── evaluate.py              ← Evaluation script
│    ├── realtime.py              ← Webcam processing
│    ├── gui_app.py               ← GUI application
│    ├── pipeline.py              ← Full pipeline
│    ├── extract_frames.py        ← Video frame extraction
│    ├── comparison.py            ← YOLO vs Manual
│    ├── monitor.py               ← Training monitor
│    └── utils/
│        ├── counter.py           ← Vehicle counting logic
│        ├── visualizer.py        ← Drawing utilities
│        └── metrics.py           ← Metrics calculation
│
├─── models/
│    └── vehicle_detection/
│        └── weights/
│            ├── best.pt          ← Best model
│            └── last.pt          ← Last checkpoint
│
├─── outputs/
│    ├── detections/              ← Detection results
│    └── evaluation/              ← Evaluation reports
│
├─── quick_start.py               ← Setup script
├─── setup_labelimg.py            ← LabelImg setup
├─── requirements.txt             ← Dependencies
└─── README.md                    ← Documentation
```

---

## 7. Command Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMMAND REFERENCE                                   │
└─────────────────────────────────────────────────────────────────────────────┘

SETUP
─────────────────────────────────────────────────────────────
python quick_start.py                    # Setup awal
pip install -r requirements.txt          # Install dependencies

DATASET
─────────────────────────────────────────────────────────────
python src/extract_frames.py --action list              # List videos
python src/extract_frames.py --action extract-all      # Extract all
python src/extract_frames.py --action split            # Split train/val
python src/dataset_prepare.py --action validate        # Validate

ANNOTATION
─────────────────────────────────────────────────────────────
pip install labelImg                     # Install LabelImg
labelImg                                 # Jalankan LabelImg
python setup_labelimg.py --action guide  # Lihat panduan

TRAINING
─────────────────────────────────────────────────────────────
python src/train.py --check              # Cek spesifikasi
python src/train.py --quick              # Quick (10 epochs)
python src/train.py                      # Full (50 epochs)
python src/monitor.py --action watch     # Monitor progress

DETECTION
─────────────────────────────────────────────────────────────
python src/detect.py --source image.jpg  # Deteksi 1 gambar
python src/detect.py --source data/raw   # Batch deteksi
python src/realtime.py --source 0 --show # Webcam

EVALUATION
─────────────────────────────────────────────────────────────
python src/evaluate.py                   # Full evaluation
python src/evaluate.py --task fps        # FPS only

PIPELINE
─────────────────────────────────────────────────────────────
python src/pipeline.py --action status   # Cek status
python src/pipeline.py --action full     # Full pipeline
python src/pipeline.py --action from-train  # Dari training
```

---

## 8. Hardware Requirements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HARDWARE SPECIFICATION                              │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │              LAPTOP SPECIFICATION                   │
    │  ┌─────────────────────────────────────────────┐    │
    │  │  CPU: Intel Core i3-1115G4 @ 3.00GHz       │    │
    │  │  Cores: 2 physical, 4 logical              │    │
    │  │  RAM: 8 GB                                  │    │
    │  │  GPU: Intel UHD Graphics (Integrated)      │    │
    │  │  OS: Windows 11 Home                        │    │
    │  └─────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────┐
    │              OPTIMIZED SETTINGS                     │
    │  ┌─────────────────────────────────────────────┐    │
    │  │  Model: YOLOv8n (Nano, 3.2M params)        │    │
    │  │  Image Size: 416x416                        │    │
    │  │  Batch Size: 4                              │    │
    │  │  Device: CPU                                │    │
    │  │  Epochs: 50                                 │    │
    │  │  Training Time: ~25-30 minutes              │    │
    │  │  Inference FPS: ~5-8 FPS                    │    │
    │  └─────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────┘
```

---

## 9. Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROJECT TIMELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

    Minggu 1-2: Dataset
    ═══════════════════════════════════════════════════════════
    [████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
    - Extract frames dari video (705 frames)
    - Split train/val (564/141)
    - Anotasi dengan LabelImg (~4 jam)

    Minggu 3: Training
    ═══════════════════════════════════════════════════════════
    [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████░░░░]
    - Training model (~30 menit)
    - Evaluasi dan optimasi
    - Testing deteksi

    Minggu 4: Deployment
    ═══════════════════════════════════════════════════════════
    [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████]
    - Real-time testing
    - Dokumentasi
    - Presentasi
```
