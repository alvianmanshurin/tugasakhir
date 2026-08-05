# WORKFLOW GUIDE - Vehicle Detection System
# Panduan Lengkap dari Awal sampai Akhir

## Status Saat Ini

```
[OK] Video frames extracted: 705 frames
[OK] Split to train/val: 564 train, 141 val
[OK] Auto-annotate with YOLOv8 pretrained (1097 boxes)
[OK] Train YOLOv8n model (39 epochs, best mAP50: 71.6%)
[OK] Evaluate model (mAP50: 64.7%, Precision: 62.2%, Recall: 76.1%)
[OK] ROI Filter (20m road boundary)
[OK] Object Tracking (ByteTrack-inspired)
[OK] Detection + Tracking Pipeline
[ ] Real-time testing with webcam
[ ] Deployment at gate ITERA
```

---

## LANGKAH 1: Auto-Anotasi Dataset

### 1.1 Auto-Annotate dengan YOLOv8 Pretrained

Script `src/auto_annotate.py` menggunakan model YOLOv8 yang sudah dilatih di COCO dataset untuk otomatis melabeli kendaraan.

```bash
# Auto-annotate training set
python src/auto_annotate.py \
  --image-dir data/annotated/images/train \
  --label-dir data/annotated/labels/train \
  --conf 0.35

# Auto-annotate validation set
python src/auto_annotate.py \
  --image-dir data/annotated/images/val \
  --label-dir data/annotated/labels/val \
  --conf 0.35
```

### 1.2 Review dengan LabelImg (Opsional)

Jika ingin merevisi hasil auto-annotation:

```bash
pip install labelImg
labelImg data/annotated/images/train config/predefined_classes.txt
```

**Shortcut:**
- `W` : Buat bounding box baru
- `D` : Gambar berikutnya
- `A` : Gambar sebelumnya
- `Ctrl+S` : Save

### 1.3 Class Mapping

| COCO ID | COCO Name | Project ID | Project Name |
|---------|-----------|------------|--------------|
| 3       | motorcycle| 0          | motor        |
| 2       | car       | 1          | mobil        |
| 5       | bus       | 2          | bus          |
| 7       | truck     | 3          | truk         |

---

## LANGKAH 2: Training Model

### 2.1 Training dengan Ultralytics

```bash
$env:PYTHONPATH = "D:\PLib"
python -c "
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='data/dataset.yaml',
    epochs=50,
    imgsz=416,
    batch=4,
    device='cpu',
    workers=2,
    optimizer='SGD',
    lr0=0.01,
    patience=15,
    save_period=10,
    project='models',
    name='yolov8n_vehicle',
    exist_ok=True
)
"
```

### 2.2 Training Configuration

| Parameter | Value | Keterangan |
|-----------|-------|------------|
| Model     | YOLOv8n | Nano (3.2M params) |
| Image Size | 416x416 | Dikurangi untuk CPU speed |
| Batch Size | 4    | Hemat RAM 8GB |
| Epochs    | 50    | Early stop di 15 epoch |
| Optimizer | SGD   | Hemat memori |
| Device    | CPU   | Intel i3-1115G4 |

### 2.3 Output Training

```
models/yolov8n_vehicle/
├── weights/
│   ├── best.pt          ← Best model (mAP50 terbaik)
│   ├── last.pt          ← Last checkpoint
│   ├── epoch0.pt        ← Checkpoint epoch 0
│   ├── epoch10.pt       ← Checkpoint epoch 10
│   └── epoch20.pt       ← Checkpoint epoch 20
├── results.csv          ← Training metrics
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── P_curve.png
```

---

## LANGKAH 3: Evaluasi

### 3.1 Jalankan Evaluasi

```bash
$env:PYTHONPATH = "D:\PLib"
python -c "
from ultralytics import YOLO

model = YOLO('models/yolov8n_vehicle/weights/best.pt')
results = model.val(
    data='data/dataset.yaml',
    imgsz=416,
    batch=4,
    plots=True,
    save_json=True,
    project='outputs/evaluation',
    name='yolov8n_best',
    exist_ok=True
)
"
```

### 3.2 Hasil Evaluasi

| Metric | Nilai |
|--------|-------|
| **Overall mAP50** | 64.7% |
| **Overall mAP50-95** | 49.1% |
| **Precision** | 62.2% |
| **Recall** | 76.1% |
| **Speed (CPU)** | 31.4ms/frame (~32 FPS) |

### 3.3 Per Kelas

| Kelas | mAP50 | Recall | Precision |
|-------|-------|--------|-----------|
| motor | 77.0% | 81.7%  | 63.8%     |
| mobil | 85.4% | 93.6%  | 67.7%     |
| bus   | 52.8% | 66.7%  | 57.1%     |
| truk  | 43.7% | 62.5%  | 60.0%     |

### 3.4 Output Files

```
outputs/evaluation/yolov8n_best/
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxF1_curve.png
├── BoxPR_curve.png
├── BoxP_curve.png
├── BoxR_curve.png
├── val_batch0_pred.jpg
└── predictions.json
```

---

## LANGKAH 4: Deteksi + ROI + Tracking

### 4.1 Pipeline Lengkap

```bash
# Webcam dengan ROI + Tracking
python src/detect_with_tracking.py --source 0 --show

# Video file
python src/detect_with_tracking.py --source "path/video.mp4" --show --output output.mp4

# Matikan ROI
python src/detect_with_tracking.py --source 0 --show --no-roi

# Atur jarak ROI
python src/detect_with_tracking.py --source 0 --show --roi-max-dist 15
```

### 4.2 ROI Configuration

```yaml
# config/config.yaml
roi:
  enabled: true
  max_distance_m: 20.0
  boundary:
    top_left: [150, 120]      # 20m (ujung jalan)
    top_right: [490, 120]     # 20m (ujung jalan)
    bottom_left: [0, 416]     # 0m (dekat kamera)
    bottom_right: [640, 416]  # 0m (dekat kamera)
  min_bbox_height: 20
  draw_roi: true
```

### 4.3 Tracking Configuration

```yaml
# config/config.yaml
tracking:
  enabled: true
  max_age: 30               # Hapus track setelah 30 frame
  min_hits: 3               # Valid setelah 3 match
  max_distance: 80.0        # Jarak matching (px)
  track_buffer: 50
  show_track_id: true
  show_velocity: false
```

### 4.4 Fitur Pipeline

| Fitur | Keterangan |
|-------|------------|
| **ROI 20m** | Hanya objek dalam 20m yang dideteksi |
| **Object Tracking** | Setiap kendaraan punya ID unik |
| **Distance Estimation** | Jarak objek ditampilkan |
| **Line Crossing Counter** | Hitung kendaraan lewat garis |
| **Velocity Display** | Arah gerak kendaraan |
| **HUD** | FPS, jumlah per kelas, active tracks |

---

## Struktur Folder Final

```
tugasakhir/
├── config/
│   ├── config.yaml              ← Konfigurasi (termasuk ROI & tracking)
│   └── predefined_classes.txt   ← Daftar kelas
│
├── data/
│   ├── dataset.yaml
│   ├── raw/
│   │   ├── KIRI-7/              ← 174 frames
│   │   ├── KIRI-9/              ← 181 frames
│   │   ├── TENGAH-7/            ← 179 frames
│   │   ├── TENGAH-9/            ← 171 frames
│   │   └── merged/              ← 705 merged frames
│   └── annotated/
│       ├── images/
│       │   ├── train/           ← 564 images
│       │   └── val/             ← 141 images
│       └── labels/
│           ├── train/           ← 387 labeled (auto-annotate)
│           └── val/             ← 110 labeled (auto-annotate)
│
├── src/
│   ├── auto_annotate.py         ← Auto-annotation script
│   ├── detect_with_tracking.py  ← Detection + ROI + Tracking pipeline
│   ├── train.py
│   ├── detect.py
│   ├── evaluate.py
│   ├── realtime.py
│   ├── gui_app.py
│   ├── pipeline.py
│   ├── extract_frames.py
│   ├── comparison.py
│   └── utils/
│       ├── counter.py           ← Vehicle counting
│       ├── roi_filter.py        ← ROI 20m filter
│       ├── tracker.py           ← Object tracking
│       ├── visualizer.py
│       └── metrics.py
│
├── models/
│   └── yolov8n_vehicle/
│       ├── weights/
│       │   ├── best.pt          ← Best model (11.7 MB)
│       │   └── last.pt
│       └── results.csv
│
├── outputs/
│   ├── detections/
│   └── evaluation/
│       └── yolov8n_best/        ← Evaluation plots
│
├── WORKFLOW.md                  ← Dokumen ini
├── FLOW_DIAGRAM.md              ← Diagram alur
├── README.md
└── requirements.txt
```

---

## Estimasi Waktu

| Aktivitas | Waktu |
|-----------|-------|
| Auto-annotate 705 gambar | ~5 menit |
| Training 50 epochs | ~30-40 menit |
| Evaluasi | ~10 detik |
| ROI + Tracking setup | Sudah selesai |

---

## Tips Penting

1. **Auto-annotation** cukup akurat untuk starting point (mAP50: 64.7%)
2. **ROI boundary** harus disesuaikan dengan sudut kamera
3. **Tracking** membantu mengurangi false positive
4. **Model bisa di-improve** dengan manual annotation correction
5. **GPU acceleration** akan meningkatkan FPS secara signifikan
