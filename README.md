# PENGEMBANGAN SUB SISTEM PERHITUNGAN JUMLAH KENDARAAN BERMOTOR BERBASIS PENGOLAHAN CITRA

## Studi Kasus: UPT K3L ITERA

Sistem deteksi dan penghitungan kendaraan secara otomatis menggunakan YOLOv8 untuk pemantauan lalu lintas di gerbang masuk Kampus ITERA.

---

## Spesifikasi Hardware yang Digunakan

| Komponen | Spesifikasi |
|----------|-------------|
| **CPU** | Intel Core i3-1115G4 @ 3.00GHz (2 cores, 4 threads) |
| **RAM** | 8 GB |
| **GPU** | Intel UHD Graphics (Integrated) |
| **OS** | Windows 11 Home |

> **Note:** Konfigurasi proyek sudah dioptimasi untuk laptop dengan spesifikasi di atas.

---

## Konfigurasi yang Digunakan

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Model | YOLOv8n (Nano) | 3.2M params, tercepat untuk CPU |
| Image Size | 416x416 | Dikurangi dari 640 untuk kecepatan |
| Batch Size | 4 | Hemat RAM (8GB) |
| Device | CPU | Tidak ada CUDA |
| Epochs | 50 | Estimasi ~25-30 menit |

---

## Fitur Utama

- **Deteksi Kendaraan**: Motor, mobil, bus, truk menggunakan YOLOv8n
- **Penghitungan Otomatis**: Kendaraan yang melintasi garis tertentu
- **Real-time Processing**: Video secara real-time (CPU)
- **Evaluasi Akurasi**: Precision, Recall, F1-Score, mAP, FPS

---

## Struktur Proyek

```
tugasakhir/
├── config/
│   └── config.yaml           # Konfigurasi (sudah optimasi CPU)
├── data/
│   ├── dataset.yaml          # Konfigurasi dataset YOLO
│   ├── raw/                  # Gambar mentah
│   └── annotated/            # Dataset train/val
├── src/
│   ├── train.py              # Training (CPU optimized)
│   ├── detect.py             # Deteksi ringan
│   ├── evaluate.py           # Evaluasi model
│   ├── realtime.py           # Real-time (ringan)
│   ├── comparison.py         # YOLO vs Manual
│   ├── batch_process.py      # Batch processing
│   ├── dataset_prepare.py    # Persiapan dataset
│   ├── annotation_helper.py  # Bantuan anotasi
│   ├── export_model.py       # Export model
│   └── utils/
│       ├── counter.py        # Penghitungan kendaraan
│       ├── visualizer.py     # Visualisasi
│       └── metrics.py        # Kalkulasi metrik
├── models/                   # Model tersimpan
├── outputs/                  # Hasil deteksi & evaluasi
├── quick_start.py            # Setup cepat
├── requirements.txt          # Dependensi (CPU only)
└── README.md
```

---

## Persiapan Lingkungan

### 1. Install PyTorch (CPU Only)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install Dependencies Lainnya

```bash
pip install -r requirements.txt
```

### 3. Atau Gunakan Quick Start

```bash
python quick_start.py
```

---

## Struktur Dataset (YOLO Format)

```
data/annotated/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   └── val/
│       ├── img100.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt
    │   └── ...
    └── val/
        ├── img100.txt
        └── ...
```

### Format Label YOLO

```
<class_id> <x_center> <y_center> <width> <height>
```

Class IDs:
- `0`: motor
- `1`: mobil
- `2`: bus
- `3`: truk

---

## Penggunaan

### 1. Cek Sistem

```bash
python src/train.py --check
```

### 2. Training Model

```bash
# Quick training (10 epochs, ~5 menit)
python src/train.py --quick

# Training penuh (50 epochs, ~25-30 menit)
python src/train.py

# Resume dari checkpoint
python src/train.py --resume models/vehicle_detection/weights/last.pt
```

### 3. Deteksi & Penghitungan

```bash
# Deteksi 1 gambar
python src/detect.py --source path/to/image.jpg

# Batch deteksi
python src/detect.py --source data/raw/

# Dengan confidence threshold
python src/detect.py --source image.jpg --conf 0.6
```

### 4. Evaluasi Model

```bash
# Evaluasi lengkap
python src/evaluate.py

# Evaluasi FPS saja
python src/evaluate.py --task fps
```

### 5. Real-time dari Webcam

```bash
python src/realtime.py --source 0 --show
```

### 6. Perbandingan YOLO vs Manual

```bash
# Buat template untuk penghitungan manual
python src/comparison.py --action template --image-dir data/raw --output manual_counts.json

# Bandingkan hasil
python src/comparison.py --action compare --image-dir data/raw --manual-json manual_counts.json
```

---

## Estimasi Waktu

| Aktivitas | Estimasi Waktu |
|-----------|---------------|
| Quick Training (10 epochs) | ~5 menit |
| Full Training (50 epochs) | ~25-30 menit |
| Deteksi 1 gambar | ~100-200ms |
| Real-time (416x416) | ~5-8 FPS |

---

## Tips untuk Laptop Ini

1. **Gunakan YOLOv8n** (sudah di-set default)
2. **Image size 416** (bukan 640)
3. **Batch size 4** (hemat RAM)
4. **Disable cache** saat training
5. **Gunakan --quick** untuk testing cepat
6. **Tutup aplikasi lain** saat training

---

## Metrik Evaluasi

| Metrik | Deskripsi |
|--------|-----------|
| **Precision** | Proporsi deteksi benar dari total deteksi |
| **Recall** | Proporsi deteksi benar dari total objek aktual |
| **F1-Score** | Harmonic mean Precision dan Recall |
| **mAP50** | Mean Average Precision pada IoU=0.5 |
| **mAP50-95** | Mean Average Precision pada IoU 0.5-0.95 |
| **FPS** | Frames Per Second (kecepatan inferensi) |

---

## Referensi

- [YOLOv8 Documentation - Ultralytics](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- Dataset: Pengambilan langsung di lokasi gerbang masuk Kampus ITERA

---

## Lisensi

Proyek ini untuk keperluan penelitian tugas akhir.
