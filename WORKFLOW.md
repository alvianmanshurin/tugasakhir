# WORKFLOW GUIDE - Vehicle Detection System
# Panduan Lengkap dari Awal sampai Akhir

## Status Saat Ini

```
[OK] Video frames extracted: 705 frames
[OK] Split to train/val: 564 train, 141 val
[OK] Auto-annotate with YOLOv11 pretrained (1097 boxes)
[OK] Train YOLOv11n model (50 epochs, best mAP50: 72.6%)
[OK] Evaluate model (mAP50: 72.6%, Precision: 62.5%, Recall: 74.5%)
[OK] ROI Filter (area jalan)
[OK] Object Tracking (ByteTrack-inspired)
[OK] Detection + Tracking Pipeline
[OK] Export TorchScript model
[ ] Real-time testing with webcam
[ ] Deployment at gate ITERA
```

---

## LANGKAH 1: Auto-Anotasi Dataset

### 1.1 Auto-Annotate dengan YOLOv11 Pretrained

Script `src/auto_annotate.py` menggunakan model YOLOv11 yang sudah dilatih di COCO dataset untuk otomatis melabeli kendaraan.

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
python src/train.py
```

Atau langsung dengan Python:

```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='D:/KULIAH/Tugas Akhir/tugasakhir/data/dataset.yaml',
    epochs=50, imgsz=416, batch=4,
    device='cpu', workers=2, optimizer='SGD',
    lr0=0.01, patience=15, save_period=10,
    project='models', name='vehicle_detection',
    exist_ok=True
)
```

### 2.2 Training Configuration

| Parameter | Value | Keterangan |
|-----------|-------|------------|
| Model     | YOLOv11n | Nano (2.6M params) |
| Image Size | 416x416 | Dikurangi untuk CPU speed |
| Batch Size | 4    | Hemat RAM 8GB |
| Epochs    | 50    | Early stop di 15 epoch |
| Optimizer | SGD   | Hemat memori |
| Device    | CPU   | Intel i3-1115G4 |

### 2.3 Output Training

```
runs/detect/models/vehicle_detection/
├── weights/
│   ├── best.pt          ← Best model (mAP50 terbaik)
│   ├── last.pt          ← Last checkpoint
│   ├── best.torchscript ← Exported TorchScript model
│   ├── epoch0.pt        ← Checkpoint epoch 0
│   ├── epoch10.pt       ← Checkpoint epoch 10
│   └── epoch20.pt       ← Checkpoint epoch 20
├── results.csv          ← Training metrics
├── confusion_matrix.png
├── BoxF1_curve.png
├── BoxPR_curve.png
├── BoxP_curve.png
└── BoxR_curve.png
```

---

## LANGKAH 3: Evaluasi

### 3.1 Jalankan Evaluasi

```bash
python src/evaluate.py --task all
```

### 3.2 Hasil Evaluasi

| Metric | Nilai |
|--------|-------|
| **Overall mAP50** | 72.6% |
| **Overall mAP50-95** | 54.8% |
| **Precision** | 62.5% |
| **Recall** | 74.5% |
| **Speed (CPU)** | 27.2ms/frame (~25 FPS) |

### 3.3 Per Kelas

| Kelas | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| motor | 68.9% | 81.7% | 80.8% | 55.1% |
| mobil | 78.0% | 85.4% | 89.5% | 67.4% |
| bus | 41.8% | 60.2% | 54.2% | 42.0% |
| truk | 61.4% | 70.8% | 65.9% | 54.7% |

### 3.4 Output Files

```
runs/detect/models/vehicle_detection/
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxF1_curve.png
├── BoxPR_curve.png
├── BoxP_curve.png
├── BoxR_curve.png
├── val_batch0_pred.jpg
└── results.csv

outputs/evaluation/
└── evaluation_report.json
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
| **ROI** | Filter area jalan |
| **Object Tracking** | Setiap kendaraan punya ID unik |
| **Dual-Line Counter** | Hitung kendaraan yang melewati 2 garis (atas → bawah atau sebaliknya) |
| **Velocity Display** | Arah gerak kendaraan |
| **HUD** | FPS, jumlah per kelas, active tracks, jumlah terhitung |

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
│           ├── train/           ← Labels train
│           └── val/             ← Labels val
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
│       ├── roi_filter.py        ← ROI filter
│       ├── tracker.py           ← Object tracking
│       ├── visualizer.py
│       └── metrics.py
│
├── runs/detect/models/vehicle_detection/
│   ├── weights/
│   │   ├── best.pt              ← Best model (PyTorch)
│   │   ├── last.pt              ← Last checkpoint
│   │   └── best.torchscript     ← Exported TorchScript
│   ├── results.csv
│   ├── confusion_matrix.png
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── BoxP_curve.png
│   └── BoxR_curve.png
│
├── outputs/
│   ├── inference_val/           ← Inference results
│   └── evaluation/
│       └── evaluation_report.json
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

1. **Auto-annotation** cukup akurat untuk starting point (mAP50: 72.6%)
2. **ROI boundary** harus disesuaikan dengan sudut kamera
3. **Tracking** membantu mengurangi false positive
4. **Model bisa di-improve** dengan manual annotation correction
5. **GPU acceleration** akan meningkatkan FPS secara signifikan
