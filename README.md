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
│   ├── config.yaml              # Konfigurasi terpusat (model, deteksi, ROI, tracking, counting)
│   └── predefined_classes.txt   # Daftar kelas untuk LabelImg
├── data/
│   ├── dataset.yaml             # Path absolut, ditulis dataset_prepare.py
│   ├── detections.db            # SQLite: sesi, deteksi, statistik, ringkasan
│   ├── raw/                     # Frame hasil ekstraksi video
│   └── annotated/
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/   # YOLO txt, sejajar dengan images/
├── src/
│   ├── pipeline.py              # VehiclePipeline: detect → ROI → track → count
│   ├── detect_with_tracking.py  # CLI video/webcam + ringkasan + --db
│   ├── train.py                 # Training YOLO11n CPU + augmentasi
│   ├── evaluate.py              # Evaluasi, laporan JSON/CSV, confusion matrix
│   ├── detect.py                # Deteksi satu gambar / folder
│   ├── batch_process.py         # Batch gambar tanpa tracking
│   ├── realtime.py              # Real-time webcam/video
│   ├── monitor.py               # Alias realtime.main
│   ├── cctv_connect.py          # CCTV/RTSP + reconnect + simpan config
│   ├── gui_app.py               # GUI Tkinter (thread-safe)
│   ├── dataset_prepare.py       # validate / split group-aware / labelme / yaml
│   ├── dataset_collect.py       # Capture webcam, ekstrak frame video
│   ├── extract_frames.py        # Ekstraksi frame + info video
│   ├── auto_annotate.py         # Auto-label dari model COCO
│   ├── annotation_helper.py     # Template LabelMe + konversi YOLO
│   ├── export_model.py          # Export ONNX/TorchScript/TFLite/CoreML/OpenVINO
│   ├── comparison.py            # Bandingkan laporan antar model
│   ├── query_db.py              # Baca & ekspor database
│   └── utils/
│       ├── paths.py             # Root project, resolver model, class mapping
│       ├── database.py          # Schema, migrasi, UPSERT
│       ├── counter.py           # Dual-line counting
│       ├── roi_filter.py        # Filter + scaling ROI
│       ├── tracker.py           # Object tracking (sumber track_id tunggal)
│       ├── visualizer.py
│       └── metrics.py
├── runs/detect/models/vehicle_detection/
│   └── weights/best.pt          # Model yang dipakai semua entry point
├── exports/                     # Hasil export_model.py
├── outputs/                     # Deteksi, video teranotasi, laporan evaluasi
├── FLOW_DIAGRAM.md              # Diagram alur
├── requirements.txt
└── README.md
```

Semua path di dalam `config.yaml` dinormalisasi menjadi absolut terhadap
root project oleh `utils/paths.py`, jadi skrip aman dijalankan dari direktori
mana pun.

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
│   ├── train/     # 2045 gambar (80% grup, stratified per kelas)
│   ├── val/       # 480 gambar
│   └── test/      # 292 gambar (10% grup utuh, tersebar 4 video)
└── labels/
    ├── train/     # 2045 label (548 kosong = background)
    ├── val/       # 480 label (99 kosong)
    └── test/      # 292 label (62 kosong)
```

Label kosong itu disengaja: gambar tanpa kendaraan tetap perlu file `.txt`
kosong, kalau tidak Ultralytics mengabaikannya dan background negatif hilang
dari training.

> Split yang sedang dipakai masih hasil split lama per-gambar acak, sehingga
> beberapa frame berdekatan bisa berada di split berbeda. `dataset_prepare.py
> --action split` sudah group-aware (blok 30 frame) dan perlu dijalankan ulang
> sebelum training berikutnya.

---

## Persiapan Lingkungan

### Manual Installation

```bash
# 1. Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install dependencies
pip install -r requirements.txt

# 3. Cek spesifikasi laptop dan kesiapan training
python src/train.py --check
```

Direktori yang dibutuhkan dibuat otomatis oleh skrip yang relevan
(`dataset_prepare.py`, `export_model.py`, `evaluate.py`). Kalau ingin membuatnya
manual:

```bash
python -c "from pathlib import Path; [p.mkdir(parents=True, exist_ok=True) for p in (Path('data/raw'), Path('data/annotated/images'), Path('data/annotated/labels'), Path('outputs'), Path('exports'), Path('config'))]"
```

> Versi yang dipakai sekarang: Python 3.11, numpy 1.26.4, opencv-python 4.8.1.78,
> torch 2.14.0+cpu, ultralytics 8.4.138. Jangan naikkan numpy ke 2.x tanpa
> menaikkan opencv-python, atau yang terjadi adalah error "compiled using NumPy
> 1.x cannot be run in NumPy 2.0.0" saat membaca gambar.

---

## Panduan Penggunaan

### 0. CLI Terpadu (satu pintu masuk)

Semua script bisa dijalankan lewat satu entry point. Argumen setelah
command diteruskan apa adanya ke scriptnya:

```bash
python main.py --help               # daftar command
python main.py train --quick        # = python src/train.py --quick
python main.py detect --source ...  # = python src/detect.py --source ...
python -m src workflow --list       # setara, lewat mode package
```

Cara lama `python src/<script>.py` tetap didukung. Command yang tersedia:
`annotate`, `annot-helper`, `batch`, `collect`, `cctv`, `compare`,
`dataset`, `detect`, `detect-track`, `evaluate`, `export`, `extract`,
`gui`, `monitor`, `pipeline`, `query-db`, `realtime`, `train`,
`workflow`.

#### Alur dataset otomatis (`workflow`)

Menghubungkan extract → merge → auto-annotate → split → validate → yaml → train:

```bash
# Tahap aman (TIDAK menyentuh train/val): stage + annotate + validate + yaml
python main.py workflow

# Full rebuild: ekstrak video interval 15 + merge pool + pilih test + split + training
python main.py workflow --extract --interval 15 --rebuild --train

# Lihat rencana tahap / jalankan satu tahap
python main.py workflow --list
python main.py workflow --stage annotate
```

Tahapan: `extract` (frame dari video, hanya video di
`dataset.video_files`), `merge` (bangun ulang `data/raw/merged` + tulis
`dataset.video_ranges` supaya rentang tidak basi), `stage` (salin pool →
`data/staging/images`, refresh bila pool berubah, idempoten), `annotate`
(model COCO → `data/staging/labels`), `split` (pilih test = grup utuh
tersebar antar video, lalu train/val stratified per kelas - hanya jalan
dengan `--rebuild` karena menimpa label review manual), `validate`
(read-only), `yaml`, `train`. Label final di `data/annotated` tidak pernah
disentuh tanpa `--rebuild`. Rasio: `--ratio` (train 0.8),
`--test-ratio` (test 0.1), `--seed` (42, deterministik).

### 1. Koleksi Dataset

```bash
# Ringkasan isi folder
python src/dataset_collect.py --action list --output-dir data/raw

# Capture dari webcam (--interval = jumlah frame yang dilewati)
python src/dataset_collect.py --action webcam --num-frames 200 --interval 15

# Ekstrak frame dari video (--interval 30 pada 30 FPS = 1 frame/detik)
python src/dataset_collect.py --action video --source video.mp4 --interval 30

# Alternatif: utilitas ekstraksi yang juga bisa mencetak info video
python src/extract_frames.py --action info --video video.mp4
python src/extract_frames.py --action extract-all --interval 30
```

`--interval` kecil menghasilkan frame yang nyaris identik. Pada 30 FPS,
`--interval 1` berarti 100 frame hanya mencakup 3.3 detik.

### 2. Anotasi

```bash
# Auto-label pakai model COCO (HANYA untuk model COCO, bukan model proyek)
python src/auto_annotate.py \
  --image-dir data/annotated/images/train \
  --label-dir data/annotated/labels/train \
  --conf 0.35

# Template LabelMe dengan ukuran gambar yang benar
# (tulis ke folder terpisah supaya folder gambar tetap bersih)
python src/annotation_helper.py --action templates \
  --image-dir data/raw --output-dir data/labelme

# Review manual (LabelMe / LabelImg, salah satu)
pip install labelme
labelme data/annotated/images/train
# atau
pip install labelImg
labelImg data/annotated/images/train config/predefined_classes.txt

# Konversi LabelMe -> YOLO
python src/dataset_prepare.py --action convert-labelme --source data/raw
```

### 3. Validasi & Persiapan Dataset

```bash
python src/dataset_prepare.py --action validate

# Laporan kebocoran split (read-only, tidak mengubah file)
python src/dataset_prepare.py --action split-report

# Split group-aware (blok 30 frame, dibatasi batas antar video)
python src/dataset_prepare.py --action split --source <dir> --clean

python src/dataset_prepare.py --action yaml
```

#### Kebocoran Split: Sudah Diperbaiki {#split-report}

`--action split-report` mengukur kebocoran tanpa menyentuh file. Split lama
(acak per gambar) bocor total:

```
SEBELUM : Grup terbagi >1 split = 25 (92.6%), frame bocor = 694 (98.4%)
SEKARANG: Grup terbagi >1 split = 0  (0.0%),  frame bocor = 0   (0.0%)
```

Split sekarang dibangun `python main.py workflow --rebuild`: grup utuh
(blok 30 frame dibatasi `video_ranges`), test = grup utuh tersebar antar 4
video (10%), train/val stratified per kelas. Jalankan perintah ini kapan
saja untuk memverifikasi angkanya tetap 0%.

### 4. Training Model

```bash
# Cek spesifikasi
python src/train.py --check

# Quick training (10 epoch, ~5 menit)
python src/train.py --quick

# Training penuh (50 epoch, ~25-30 menit)
python src/train.py

# Real-time monitor (bukan --action watch)
python src/monitor.py --source 0 --show
```

### 5. Deteksi Kendaraan

```bash
# Deteksi 1 gambar
python src/detect.py --source path/to/image.jpg

# Folder
python src/detect.py --source data/annotated/images/test/

# Threshold lain
python src/detect.py --source image.jpg --conf 0.6

# Batch + simpan label YOLO untuk verifikasi manual
python src/batch_process.py --input-dir data/annotated/images/test \
                            --output-dir outputs/detections --save-labels
```

### 6. Video, Webcam, CCTV, GUI

```bash
# Video + ROI + tracking + counting
python src/detect_with_tracking.py --source video.mp4 --output out.mp4

# Simpan ke database
python src/detect_with_tracking.py --source video.mp4 --db

# Real-time webcam
python src/realtime.py --source 0 --show

# CCTV / RTSP
python src/cctv_connect.py --url rtsp://user:pass@ip:554/stream

# GUI
python src/gui_app.py
python src/gui_app.py --list-sources     # daftar video yang tersedia
```

### 7. Evaluasi Model

```bash
python src/evaluate.py --task all          # metrik + FPS + confusion matrix
python src/evaluate.py --task fps
python src/evaluate.py --task confusion --max-images 200

# Confusion matrix pada split tertentu
python src/evaluate.py --split test --task confusion

# Bandingkan antar model
python src/comparison.py outputs/evaluation
python src/comparison.py --per-class outputs/evaluation/evaluation_report.json
```

> `--split` hanya mengubah confusion matrix. mAP/Precision/Recall dari
> Ultralytics selalu mengikuti split yang tertulis di `data/dataset.yaml`
> (val). Nama file hasil ikut split: `confusion_matrix_val.png` dan
> `confusion_matrix_test.png`.

### 8. Database

```bash
python src/query_db.py --action sessions
python src/query_db.py --action detail --session 1
python src/query_db.py --action detections --session 1 --limit 20
python src/query_db.py --action export --session 1
```

Jumlah kendaraan selalu `COUNT(DISTINCT vehicle_id)`, bukan `COUNT(*)`.
Tabel `detections` menyimpan satu baris per track per frame.

### 9. Export Model

```bash
python src/export_model.py --format onnx --output exports
python src/export_model.py --format torchscript --output exports
```

`--imgsz` default diambil dari config (416). ONNX dan TorchScript sudah
terverifikasi; TFLite/CoreML/OpenVINO butuh dependency tambahan dan paling
sering gagal di CPU-only.

---

## Estimasi Waktu

| Aktivitas | Estimasi |
|-----------|----------|
| Quick Training (10 epochs) | ~5 menit |
| Full Training (50 epochs) | ~25-30 menit |
| Deteksi 1 gambar | ~100-200ms |
| Real-time (416x416) | ~5-8 FPS |
| Batch (100 gambar) | ~2-3 menit |
| Video + ROI + tracking (416, CPU) | ~16-22 FPS terukur |

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
| `line1_position` | 0.48 | Garis atas (rasio tinggi frame) |
| `line2_position` | 0.75 | Garis bawah (rasio tinggi frame) |
| `direction` | both | Hitung arah masuk & keluar |
| `min_track_length` | 3 | Frame minimum sebelum dihitung |

Arah `down` = masuk, `up` = keluar. ROI filter memakai boundary trapezoid
dalam koordinat pixel `reference_resolution` (1920x1080) yang diskalakan ke
resolusi video sebenarnya. Boundary sekarang hanya mencakup sekitar 13.8%
area frame, jadi kalibrasi terhadap rekaman gerbang masih wajib.

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

### Hasil Evaluasi Terukur

> **Angka di bawah berasal dari model & split LAMA (705 frame, 02 Sep-01 Okt
> 2026).** Dataset sekarang 2817 frame dengan split bebas-leakage, jadi
> training + evaluasi harus diulang sebelum angka ini dipakai di laporan.

`python src/evaluate.py --task all` (run terakhir, tersimpan di
`outputs/evaluation/evaluation_report.json`):

| Metrik | Nilai | Target | Status |
|--------|-------|--------|--------|
| mAP50 | 0.3850 | 0.75 | belum tercapai |
| mAP50-95 | 0.2993 | 0.50 | belum tercapai |
| Precision | 0.6027 | 0.70 | belum tercapai |
| Recall | 0.4121 | 0.70 | belum tercapai |
| F1 | 0.4850 | 0.70 | belum tercapai |
| FPS | 32.9 | > 5 | tercapai |

Per kelas:

| Kelas | AP50 | AP50-95 | Precision | Recall | F1 |
|-------|------|---------|-----------|--------|-----|
| mobil | 0.6922 | 0.5597 | 0.9184 | 0.7031 | 0.7965 |
| motor | 0.4529 | 0.3054 | 0.6923 | 0.5455 | 0.6102 |
| truk | 0.3950 | 0.3322 | 0.8000 | 0.4000 | 0.5333 |
| bus | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Confusion matrix (IoU>=0.5, conf>=0.5) pada dua split:

| Split | TP | FP | FN | Precision | Recall |
|-------|----|----|----|-----------|--------|
| val (71 gambar) | 68 | 17 | 40 | 0.8000 | 0.6296 |
| test (70 gambar) | 84 | 11 | 61 | 0.8842 | 0.5793 |

Test set di atas baru diukur sekali pada 28 Sep 2026. Angka test **tidak
boleh dipakai untuk tuning apa pun** - hanya untuk laporan akhir.

Kelas `bus` tidak menghasilkan satu pun true positive pada evaluasi lama.
Penyebabnya sudah terukur, bukan dugaan:

- hanya **99 instans bus** di seluruh 2817 frame (~2% instance)
- split lama menyisakan **1 instans bus di val** dan 98.4% frame bocor antar
  split - keduanya sudah diperbaiki (stratified split: bus 70 train / 18 val /
  11 test, leakage 0.0%)
- masalah yang tersisa adalah **jumlahnya**, bukan pembagiannya

**Mengganti split saja tidak menyelesaikan kelas bus**; annotasi bus tambahan
(target 150-200 instans) adalah langkah yang benar-benar menentukan.

> Angka 72.6% yang pernah ada di dokumen ini tidak bisa direproduksi lagi
> (jalur `save_dir` dan nama kurva tidak cocok dengan Ultralytics yang
> terpasang). Jangan mengutipnya di laporan.

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

### v1.4.0 - Dataset Regeneration & Leak-Free Split (02 Okt 2026)

- **Added:** tahap `merge` di workflow - membangun ulang `data/raw/merged`
  dari folder per video sekaligus menulis `dataset.video_ranges` di config
  (rentang tidak pernah basi setelah ekstrak ulang)
- **Added:** `extract` menghormati `dataset.video_files`, jadi video hasil
  proses (`*_output.mp4`) tidak ikut masuk pool dataset
- **Added:** `dataset --action select-test` + `workflow --test-ratio` -
  test set = grup utuh, round-robin antar video (default 10%)
- **Added:** split train/val **stratified per kelas** - kelas langka (bus)
  tidak lagi menumpuk di satu split (bus: 1 → 18 instans di val)
- **Added:** `rmtree_force()` - `split --clean` tahan folder beratribut
  ReadOnly (sebelumnya WinError 5 di tengah split)
- **Fixed:** ekstrak ulang menumpuk frame lama bercampur frame baru
  (penomoran melanjutkan file yang ada) - folder output kini dibersihkan
- **Fixed:** guard cek label di `workflow` memakai filter gambar, jadi `.txt`
  tidak pernah terdeteksi dan tahap split selalu gagal
- **Fixed:** stage tidak mendeteksi gambar basi - pool baru bisa berpadu
  dengan label lama; kini gambar basi diganti + labelnya dibuang, yatim
  dibersihkan
- **Data:** 2817 frame (interval 15, 4 video) → 4835 box auto-annotate;
  split train 2045 / val 480 / test 292, leakage **98.4% → 0.0%**

### v1.3.0 - CLI Unification & Dataset Workflow (01 Okt 2026)

- **Added:** `main.py` / `python -m src` - satu entry point untuk semua script; argumen diteruskan ke argparse masing-masing modul
- **Added:** `src/cli.py` - dispatcher dengan lazy import (help instan, tanpa memuat ultralytics/torch)
- **Added:** `src/workflow.py` - alur dataset otomatis: stage → annotate → split → validate → yaml → train, dengan staging terpisah (`data/staging`) supaya label review manual tidak tertimpa diam-diam
- **Fixed:** ringkasan "Per kelas" `auto_annotate` selalu menampilkan 0 (dict id↔nama terbalik)
- **Fixed:** `--swap` `auto_annotate` memberi pesan `[ERROR]` yang jelas, bukan traceback (validasi id kelas proyek)
- **Updated:** bootstrap `sys.path` konsisten di semua modul (`cctv_connect.py`, `src/__init__.py`)

### v1.2.0 - Pipeline Unification (28 Sep 2026)

- **Fixed:** `VehiclePipeline` menjadi satu-satunya sumber logika deteksi → ROI → track → count
- **Fixed:** `track_id` hanya berasal dari `utils/tracker.py`; ID dari modul lain bisa bentrok antar modul
- **Fixed:** GUI thread-safety (worker tidak lagi menyentuh widget Tk)
- **Fixed:** jumlah kendaraan = `counter.total_count`, bukan `COUNT(*)` baris database
- **Fixed:** GUI tidak lagi menambah penghitung naïf per frame
- **Fixed:** split dataset group-aware (blok 30 frame) mencegah frame berdekatan berbeda split
- **Fixed:** migrasi SQLite menambahkan UNIQUE constraint pada tabel ringkasan
- **Fixed:** CLI yang tidak punya argparse (`comparison`, `dataset_collect`, `annotation_helper`, `gui_app`)
- **Fixed:** `extract_frames.py` memakai config dan path absolut, splitter ganda dihapus
- **Fixed:** `query_db.py` memakai path database absolut dan exit code bermakna
- **Fixed:** `export_model.py` menghormati `--output` (dulu artefak selalu di folder weights)
- **Fixed:** `process_image()` tidak lagi melaporkan "baris track" padahal tanpa tracking
- **Added:** template LabelMe sekarang berisi ukuran gambar asli
- **Added:** ekstraksi frame memperingatkan jarak antar frame < 1 detik

### v1.1.0 - Augmentasi Optimization (17 Sep 2026)

- **Fixed:** Matikan random erasing (`erasing: 0.0`) untuk menjaga integritas bounding box
- **Added:** Shear augmentation (`±5°`) untuk simulasi sudut pandang kamera
- **Added:** Blur augmentation (`1%`) untuk robustness kamera goyang
- **Added:** Grayscale augmentation (`10%`) untuk robustness minim cahaya
- **Added:** Perspective (`0.001`) untuk distorsi kamera ringan
- **Updated:** Semua augmentasi parameters di-pass dari config.yaml ke model.train()

### v1.0.0 - Initial Release

- YOLOv11n training (50 epochs)
- Dual-line counting system
- GUI application with CCTV/RTSP support
- ROI filter + Object tracking pipeline
- TorchScript model export

---

## Referensi

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [LabelImg GitHub](https://github.com/heartexlabs/labelImg)
- Dataset: https://drive.google.com/drive/folders/1_uWlyPXIavfucFLPfNK_IU2iH3sDI8xu?usp=sharing

---

## Lisensi

Proyek ini untuk keperluan penelitian tugas akhir.
