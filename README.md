# PENGEMBANGAN SUB SISTEM PERHITUNGAN JUMLAH KENDARAAN BERMOTOR BERBASIS PENGOLAHAN CITRA

## Studi Kasus: UPT K3L ITERA

Sistem deteksi dan penghitungan kendaraan secara otomatis menggunakan YOLOv11 untuk pemantauan lalu lintas di gerbang masuk Kampus ITERA.

---

## Spesifikasi Hardware

| Komponen | Spesifikasi |
|----------|-------------|
| **CPU** | Intel Core i3-1115G4 @ 3.00GHz (2 cores, 4 threads) |
| **RAM** | 8 GB |
| **GPU** | Intel UHD Graphics (Integrated) |
| **OS** | Windows 11 Home |

> Konfigurasi sudah dioptimasi untuk laptop ini.

---

## Konfigurasi yang Digunakan

### Model & Training

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Model | YOLOv11n (Nano) | 2.6M params, tercepat untuk CPU |
| Image Size | 416x416 | Lebih cepat dari 640 |
| Batch Size | 4 | Hemat RAM |
| Device | CPU | Tidak ada CUDA |
| Epochs | 50 | ~25-30 menit |
| Optimizer | SGD | Hemat memori dari Adam |

### Augmentasi (Optimized untuk Vehicle Counting)

| Kategori | Parameter | Nilai | Keterangan |
|----------|-----------|-------|------------|
| **Flip** | `fliplr` | 0.5 | Flip horizontal 50% |
| | `flipud` | 0.0 | Flip vertikal MATI |
| **Mosaic** | `mosaic` | 1.0 | Gabung 4 gambar |
| | `close_mosaic` | 10 | Nonaktif di epoch terakhir |
| **Color/HSV** | `hsv_h` | 0.015 | Hue shift |
| | `hsv_s` | 0.7 | Saturasi |
| | `hsv_v` | 0.4 | Brightness (untuk CCTV) |
| **Geometri** | `degrees` | 0.0 | Rotasi MATI (kamera fixed) |
| | `translate` | 0.1 | Translasi 10% |
| | `scale` | 0.5 | Scale 50-150% |
| | `shear` | 5.0 | Shear ±5° (sudut kamera) |
| | `perspective` | 0.001 | Distorsi kamera ringan |
| **Kualitas** | `blur` | 0.01 | Blur 1% (kamera goyang) |
| | `erasing` | **0.0** | **DIMATIKAN** (counting) |
| | `grayscale` | 0.1 | Grayscale 10% |
| **Matikan** | `mixup` | 0.0 | Ganggu bounding box |
| | `copy_paste` | 0.0 | Duplikasi kendaraan |
| | `crop_fraction` | 1.0 | Full crop |

> **Catatan:** `erasing` dimatikan untuk menjaga integritas bounding box pada objek kendaraan besar (bus/truk).

---

## Struktur Proyek

```
tugasakhir/
├── config/
│   ├── config.yaml              # Konfigurasi proyek + augmentasi
│   └── predefined_classes.txt   # Daftar kelas untuk LabelImg
├── data/
│   ├── dataset.yaml             # Konfigurasi dataset YOLO
│   ├── raw/                     # Gambar mentah
│   └── annotated/               # Dataset train/val
│       ├── images/
│       │   ├── train/           # 80% data training
│       │   └── val/             # 20% data validasi
│       └── labels/
│           ├── train/           # Label training (YOLO format)
│           └── val/             # Label validasi
├── src/
│   ├── train.py                 # Training (CPU optimized + augmentasi)
│   ├── detect.py                # Deteksi ringan
│   ├── evaluate.py              # Evaluasi model
│   ├── realtime.py              # Real-time webcam/video
│   ├── gui_app.py               # GUI application
│   ├── comparison.py            # YOLO vs Manual
│   ├── batch_process.py         # Batch processing
│   ├── dataset_prepare.py       # Persiapan dataset
│   ├── dataset_collect.py       # Pengumpulan dataset
│   ├── annotation_helper.py     # Bantuan anotasi
│   ├── export_model.py          # Export model
│   └── monitor.py               # Training monitor
├── models/                      # Model tersimpan
│   └── vehicle_detection/
│       └── weights/
│           ├── best.pt          # Model terbaik
│           └── last.pt          # Model terakhir
├── outputs/                     # Hasil deteksi
├── quick_start.py               # Setup cepat
├── setup_labelimg.py            # Setup LabelImg
├── requirements.txt             # Dependensi
└── README.md
```

---

## Dataset

### Kelas Kendaraan

| ID | Kelas | Deskripsi |
|----|-------|-----------|
| 0 | `motor` | Sepeda motor (semua jenis) |
| 1 | `mobil` | Mobil penumpang |
| 2 | `bus` | Bus |
| 3 | `truk` | Truk |

### Format Anotasi (YOLO)

```
<class_id> <center_x> <center_y> <width> <height>
```

Contoh:
```
0 0.512 0.345 0.089 0.156
1 0.234 0.678 0.123 0.234
```

### Struktur Dataset

```
data/annotated/
├── images/
│   ├── train/     # 80% gambar training
│   └── val/       # 20% gambar validasi
└── labels/
    ├── train/     # Label training (.txt)
    └── val/       # Label validasi (.txt)
```

---

## Persiapan Lingkungan

### Quick Start (Recommended)

```bash
python quick_start.py
```

### Manual Installation

```bash
# 1. Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create directories
python quick_start.py  # Pilih step 2
```

---

## Panduan Penggunaan

### 1. Koleksi Dataset

```bash
# Tampilkan panduan
python src/dataset_collect.py --action guide

# Capture dari webcam
python src/dataset_collect.py --action webcam --interval 2

# Extract dari video
python src/dataset_collect.py --action video --source video.mp4 --interval 30

# Split dataset
python src/dataset_collect.py --action split --source data/raw
```

### 2. Anotasi dengan LabelImg

```bash
# Install LabelImg
python setup_labelimg.py --action install

# Lihat panduan
python setup_labelimg.py --action guide

# Launch LabelImg
python setup_labelimg.py --action launch
```

### 3. Validasi Dataset

```bash
python src/dataset_prepare.py --action validate
```

### 4. Training Model

```bash
# Cek spesifikasi
python src/train.py --check

# Quick training (10 epochs, ~5 menit)
python src/train.py --quick

# Training penuh (50 epochs, ~25-30 menit)
python src/train.py

# Monitor training
python src/monitor.py --action watch
```

### 5. Deteksi Kendaraan

```bash
# Deteksi 1 gambar
python src/detect.py --source path/to/image.jpg

# Batch deteksi
python src/detect.py --source data/raw/

# Dengan confidence threshold
python src/detect.py --source image.jpg --conf 0.6
```

### 6. Real-time Webcam

```bash
python src/realtime.py --source 0 --show
```

### 7. Evaluasi Model

```bash
# Evaluasi lengkap
python src/evaluate.py

# FPS benchmark
python src/evaluate.py --task fps
```

### 8. Perbandingan YOLO vs Manual

```bash
# Buat template manual count
python src/comparison.py --action template --image-dir data/raw --output manual_counts.json

# Isi manual_counts.json dengan hasil hitung manual

# Bandingkan
python src/comparison.py --action compare --image-dir data/raw --manual-json manual_counts.json
```

### 9. GUI Application

```bash
python src/gui_app.py
```

### 10. Export Model

```bash
python src/export_model.py --action export --format onnx tflite
```

---

## Estimasi Waktu

| Aktivitas | Estimasi |
|-----------|----------|
| Quick Training (10 epochs) | ~5 menit |
| Full Training (50 epochs) | ~25-30 menit |
| Deteksi 1 gambar | ~100-200ms |
| Real-time (416x416) | ~5-8 FPS |
| Batch (100 gambar) | ~2-3 menit |

---

## Pipeline Sistem

```
Video/Webcam → Frame Extraction → Preprocessing → YOLOv11 Detection → ROI Filter
                                                                          ↓
                                              Counting ← Tracking ← Bounding Box
```

### Komponen Utama

| Komponen | Deskripsi |
|----------|-----------|
| **Input** | Video file (.MOV, .MP4) atau Webcam/RTSP |
| **Preprocessing** | Resize 416x416, augmentasi saat training |
| **Detection** | YOLOv11n dengan confidence threshold 0.5 |
| **ROI Filter** | Region of Interest (trapezoid) |
| **Tracking** | ByteTrack-inspired multi-object tracking |
| **Counting** | Dual-line crossing detection |

### ROI Configuration

```
┌─────────────────────────────────┐
│         Gerbang Masuk           │
│    ┌───────────────────┐        │
│    │    Line 1 (0.48)  │ ← Masuk│
│    │                   │        │
│    │    Line 2 (0.75)  │ ← Keluar│
│    └───────────────────┘        │
└─────────────────────────────────┘
```

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| `line1_position` | 0.48 | Garis atas (masuk) |
| `line2_position` | 0.75 | Garis bawah (keluar) |
| `direction` | both | Hitung arah masuk & keluar |
| `min_track_length` | 3 | Frame minimum sebelum dihitung |

---

## Metrik Evaluasi

| Metrik | Deskripsi | Target |
|--------|-----------|--------|
| **Precision** | Proporsi deteksi benar | > 0.7 |
| **Recall** | Proporsi objek terdeteksi | > 0.7 |
| **F1-Score** | Harmonic mean Precision dan Recall | > 0.7 |
| **mAP50** | Mean Average Precision (IoU=0.5) | > 0.75 |
| **mAP50-95** | Mean Average Precision (IoU 0.5-0.95) | > 0.5 |
| **FPS** | Frames Per Second | > 5 |

### Hasil Training (Sebelum Optimasi Augmentasi)

| Kelas | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| motor | 0.689 | 0.817 | 0.808 | 0.551 |
| mobil | 0.780 | 0.854 | 0.895 | 0.674 |
| bus | 0.418 | 0.602 | 0.542 | 0.420 |
| truk | 0.614 | 0.708 | 0.659 | 0.547 |

> **Catatan:** Hasil di atas menggunakan augmentasi default. Setelah optimasi augmentasi (erasing dimatikan, shear/blur/ditambahkan), diharapkan peningkatan pada kelas bus dan truk.

---

## Tips untuk Laptop Ini

1. Gunakan **YOLOv11n** (sudah default)
2. Image size **416** (bukan 640)
3. Batch size **4** (hemat RAM)
4. Tutup aplikasi lain saat training
5. Gunakan `--quick` untuk testing
6. Webcam berjalan di ~5-8 FPS
7. Gunakan GPU jika tersedia (upgrade driver)
8. **Augmentasi sudah dioptimasi** untuk vehicle counting (erasing dimatikan)

---

## Changelog

### v1.1.0 - Augmentasi Optimization (17 Sep 2026)

- **Fixed:** Matikan random erasing (`erasing: 0.0`) untuk menjaga integritas bounding box
- **Added:** Shear augmentation (`±5°`) untuk simulasi sudut pandang kamera
- **Added:** Blur augmentation (`1%`) untuk robustness kamera goyang
- **Added:** Grayscale augmentation (`10%`) untuk robustness minim cahaya
- **Added:** Perspective (`0.001`) untuk distorsi kamera ringan
- **Updated:** Semua augmentasi parameters di-pass dari config.yaml ke model.train()

### v1.0.0 - Initial Release

- YOLOv11n training (50 epochs, mAP50: 72.6%)
- Dual-line counting system
- GUI application with CCTV/RTSP support
- ROI filter + Object tracking pipeline
- TorchScript model export

---

## Referensi

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [LabelImg GitHub](https://github.com/heartexlabs/labelImg)
- Dataset: Gerbang masuk Kampus ITERA

---

## Lisensi

Proyek ini untuk keperluan penelitian tugas akhir.
