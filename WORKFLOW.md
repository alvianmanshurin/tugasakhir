# WORKFLOW GUIDE - Vehicle Detection System
# Panduan Lengkap dari Awal sampai Akhir

## Status Saat Ini

```
[OK] Video frames extracted: 705 frames
[OK] Split to train/val: 564 train, 141 val
[ ] Annotate with LabelImg
[ ] Train model
[ ] Evaluate
[ ] Deploy
```

---

## LANGKAH 1: Anotasi Dataset ( LabelImg )

### 1.1 Install LabelImg

```bash
pip install labelImg
```

### 1.2 Jalankan LabelImg

```bash
labelImg
```

### 1.3 Pengaturan Awal

1. **Buka folder train:**
   - Klik "Open Dir" → pilih `data/annotated/images/train`

2. **Atur save directory:**
   - Klik "Change Save Dir" → pilih `data/annotated/labels/train`

3. **Pastikan format YOLO:**
   - Di bagian bawah, pastikan tertulis "YOLO" (bukan PascalVOC)

4. **Load classes:**
   - Klik "Edit" → "Edit Label"
   - Hapus semua default classes
   - Tambahkan 4 classes:
     - motor
     - mobil
     - bus
     - truk

### 1.4 Cara Anotasi

1. **Tekan W** untuk membuat bounding box baru
2. **Gambar rectangle** di sekitar kendaraan
3. **Pilih label** (motor/mobil/bus/truk)
4. **Tekan D** untuk lanjut ke gambar berikutnya
5. **Tekan Ctrl+S** untuk save
6. **Ulangi** sampai semua gambar selesai

### 1.5 Tips Anotasi

- **Bounding box harus tepat** (tight) di sekitar kendaraan
- **Jangan potong** kendaraan
- **Anotasi semua** kendaraan yang terlihat, termasuk yang kecil
- **Gunakan zoom** untuk kendaraan yang jauh
- **Simpan secara berkala** (Ctrl+S)

### 1.6 Anotasi Val Set

Setelah train selesai, ulangi untuk val:
1. Open Dir → `data/annotated/images/val`
2. Change Save Dir → `data/annotated/labels/val`

---

## LANGKAH 2: Validasi Dataset

```bash
python src/dataset_prepare.py --action validate
```

Expected output:
```
[TRAIN]
  Images: 564
  Labels: 564
  Classes: {'motor': 300, 'mobil': 250, ...}

[VAL]
  Images: 141
  Labels: 141
  Classes: {'motor': 80, 'mobil': 60, ...}

[OK] Dataset validation passed!
```

---

## LANGKAH 3: Training Model

### 3.1 Quick Training (Testing)

```bash
python src/train.py --quick
```

Waktu: ~5 menit (10 epochs)

### 3.2 Full Training

```bash
python src/train.py
```

Waktu: ~25-30 menit (50 epochs)

### 3.3 Monitor Training

```bash
python src/monitor.py --action watch
```

---

## LANGKAH 4: Evaluasi

```bash
python src/evaluate.py
```

Output:
- mAP50, mAP50-95
- Precision, Recall, F1-Score
- FPS (kecepatan)
- Confusion Matrix

---

## LANGKAH 5: Deteksi

### 5.1 Single Image

```bash
python src/detect.py --source path/to/image.jpg
```

### 5.2 Batch

```bash
python src/detect.py --source data/raw/
```

### 5.3 Webcam

```bash
python src/realtime.py --source 0 --show
```

---

## Struktur Folder Setelah Anotasi

```
data/annotated/
├── images/
│   ├── train/
│   │   ├── merged_00000.jpg
│   │   ├── merged_00001.jpg
│   │   └── ... (564 files)
│   └── val/
│       ├── merged_00564.jpg
│       └── ... (141 files)
└── labels/
    ├── train/
    │   ├── merged_00000.txt  ← ini dibuat oleh LabelImg
    │   └── ...
    └── val/
        ├── merged_00564.txt
        └── ...
```

---

## Format Label YOLO

Setiap file `.txt` berisi:
```
<class_id> <x_center> <y_center> <width> <height>
```

Contoh (1 motor, 1 mobil):
```
0 0.5 0.3 0.1 0.2
1 0.7 0.6 0.15 0.25
```

Class mapping:
- 0 = motor
- 1 = mobil
- 2 = bus
- 3 = truk

---

## Estimasi Waktu

| Aktivitas | Waktu |
|-----------|-------|
| Anotasi 564 gambar | ~3-4 jam |
| Anotasi 141 gambar | ~1 jam |
| Quick Training | ~5 menit |
| Full Training | ~25-30 menit |
| Evaluasi | ~2-3 menit |

---

## Tips Penting

1. **Anotasi yang benar** = model yang akurat
2. **Minimal 100 gambar per kelas** untuk hasil baik
3. **Variasi** kondisi (siang, ramai, sepi)
4. **Jangan skip** kendaraan yang kecil/jauh
5. **Simpan berkala** saat anotasi
