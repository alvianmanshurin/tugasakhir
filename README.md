# PENGEMBANGAN SUB SISTEM PERHITUNGAN JUMLAH KENDARAAN BERMOTOR BERBASIS PENGOLAHAN CITRA

## Studi Kasus: UPT K3L ITERA

Sistem deteksi dan penghitungan kendaraan secara otomatis menggunakan YOLOv8 untuk pemantauan lalu lintas di gerbang masuk Kampus ITERA.

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

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Model | YOLOv8n (Nano) | 3.2M params, tercepat untuk CPU |
| Image Size | 416x416 | Lebih cepat dari 640 |
| Batch Size | 4 | Hemat RAM |
| Device | CPU | Tidak ada CUDA |
| Epochs | 50 | ~25-30 menit |

---

## Struktur Proyek

```
tugasakhir/
├── config/
│   ├── config.yaml              # Konfigurasi proyek
│   └── predefined_classes.txt   # Daftar kelas untuk LabelImg
├── data/
│   ├── dataset.yaml             # Konfigurasi dataset YOLO
│   ├── raw/                     # Gambar mentah
│   └── annotated/               # Dataset train/val
├── src/
│   ├── train.py                 # Training (CPU optimized)
│   ├── detect.py                # Deteksi ringan
│   ├── evaluate.py              # Evaluasi model
│   ├── realtime.py              # Real-time webcam
│   ├── gui_app.py               # GUI application
│   ├── comparison.py            # YOLO vs Manual
│   ├── batch_process.py         # Batch processing
│   ├── dataset_prepare.py       # Persiapan dataset
│   ├── dataset_collect.py       # Pengumpulan dataset
│   ├── annotation_helper.py     # Bantuan anotasi
│   ├── export_model.py          # Export model
│   └── monitor.py               # Training monitor
├── models/                      # Model tersimpan
├── outputs/                     # Hasil deteksi
├── quick_start.py               # Setup cepat
├── setup_labelimg.py            # Setup LabelImg
├── requirements.txt             # Dependensi
└── README.md
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

## Metrik Evaluasi

| Metrik | Deskripsi |
|--------|-----------|
| **Precision** | Proporsi deteksi benar |
| **Recall** | Proporsi objek terdeteksi |
| **F1-Score** | Harmonic mean Precision dan Recall |
| **mAP50** | Mean Average Precision (IoU=0.5) |
| **mAP50-95** | Mean Average Precision (IoU 0.5-0.95) |
| **FPS** | Frames Per Second |

---

## Tips untuk Laptop Ini

1. Gunakan **YOLOv8n** (sudah default)
2. Image size **416** (bukan 640)
3. Batch size **4** (hemat RAM)
4. Tutup aplikasi lain saat training
5. Gunakan `--quick` untuk testing
6. Webcam berjalan di ~5-8 FPS
7. Gunakan GPU jika tersedia (upgrade driver)

---

## Referensi

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [LabelImg GitHub](https://github.com/heartexlabs/labelImg)
- Dataset: https://drive.google.com/drive/folders/1_uWlyPXIavfucFLPfNK_IU2iH3sDI8xu?usp=sharing

---

## Lisensi

Proyek ini untuk keperluan penelitian tugas akhir.
