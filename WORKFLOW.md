# WORKFLOW GUIDE - Vehicle Detection System
# Panduan Lengkap dari Awal sampai Akhir

## Status Saat Ini

```
[OK] Video frames extracted          : 705 frame unik (KIRI-7/9, TENGAH-7/9, merged)
[OK] Provenance frame merged_ dipetakan: 705/705 cocok ke 4 video sumber
[OK] Split lama terukur bocor          : 98.4% frame di grup yang terbagi >1 split
[OK] Auto-annotate YOLO11 COCO        : 1097 box
[OK] Train YOLOv11n (50 epoch, CPU)   : runs/detect/models/vehicle_detection/weights/best.pt
[OK] Evaluasi val (--task all)        : mAP50 38.5%, mAP50-95 29.9%, P 60.3%, R 41.2%
[OK] Evaluasi test (--split test)     : TP 84, FP 11, FN 61, P 88.4%, R 57.9%
[OK] ROI filter + boundary config     : trapezoid 1920x1080, min_bbox_height 20
[OK] Object tracking (min_hits 3)     : sumber tunggal track_id
[OK] Dual-line counter                : arah down (masuk) / up (keluar)
[OK] Database SQLite (WAL, UNIQUE)    : sessions, detections, frame_stats, ringkasan
[OK] Export ONNX & TorchScript        : artefak dipindah ke folder --output
[OK] Pipeline end-to-end diuji        : video sintetis + 70 gambar test
[ ] Anotasi bus tambahan              : hanya 15 dari 705 gambar memuat bus
[ ] Regenerate split (lihat BAGIAN 0)
[ ] Kalibrasi ROI dengan video operasional nyata (butuh rekaman gerbang)
[ ] Training ulang setelah perbaikan dataset
[ ] Verifikasi arah masuk/keluar di CCTV nyata (butuh video crossing)
[ ] Deployment at gate ITERA
```

> Catatan metrik: angka 72.6% di versi lama dokumen ini berasal dari
> evaluation report yang tidak bisa direproduksi lagi (jalur `save_dir` dan
> nama kurva tidak cocok dengan Ultralytics yang terpasang). Angka di atas
> adalah hasil run `python src/evaluate.py --task all` yang benar dan
> tersimpan di `outputs/evaluation/evaluation_report.json`.

---

## CLI Terpadu (semua perintah lewat satu pintu)

Setiap langkah di dokumen ini bisa dijalankan lewat `python main.py <command>`
(argumen sama persis dengan `python src/<script>.py`):

```bash
python main.py --help          # daftar command
python main.py extract --action extract-all
python main.py annotate --image-dir ... --label-dir ...
python main.py dataset --action split-report
python main.py train --quick
python main.py evaluate --task all
python main.py pipeline --source 0 --show
```

Alur dataset end-to-end (stage → annotate → split → validate → yaml → train):

```bash
python main.py workflow             # tahap aman (staging, tanpa timpa train/val)
python main.py workflow --rebuild   # + split --clean (MENIMPA label review manual)
python main.py workflow --list      # lihat rencana tahap
```

Cara lama `python src/<script>.py` tetap didukung.

---

## BAGIAN 0: Temuan Penting Tentang Split (baca sebelum training ulang)

### 0.1 Split yang aktif sekarang bocor total

`python src/dataset_prepare.py --action split-report` (read-only):

```
Jumlah grup      : 27 (group_size=30)
Grup terbagi ke >1 split : 25 (92.6% dari grup)
Frame di grup bocor     : 694 (98.4% dari frame)
```

Hampir semua blok 30 frame terbagi ke train, val, dan test sekaligus.
Artinya metrik validasi maupun test **terlalu tinggi** dan tidak
menunjukkan kemampuan di lapangan.

### 0.2 Asal tiap frame merged_ sudah diketahui

Dengan pencocokan MD5 ke `data/raw` (705/705 cocok), pool `merged_` ternyata
adalah 4 video yang digabung berurutan, dan nomor urutnya tidak tumpang tindih:

| Video sumber | Frame merged_ | Jumlah |
|--------------|---------------|--------|
| KIRI-7 | 00000 - 00173 | 174 |
| KIRI-9 | 00174 - 00354 | 181 |
| TENGAH-7 | 00355 - 00533 | 179 |
| TENGAH-9 | 00534 - 00704 | 171 |

Rentang ini sudah ditulis ke `config/config.yaml` sebagai `dataset.video_ranges`.

### 0.3 Blok 30 frame memotong batas antar video

Tanpa `video_ranges`, blok 30 memotong batas antar kamera di **3 titik**
(blok 5, 11, 17), sehingga frame dari dua kamera berbeda masuk grup yang sama.
Dengan `video_ranges` hal ini tidak mungkin terjadi; jumlah grup naik dari 24
jadi 27 dan tidak ada grup yang memuat dua kamera.

### 0.4 Pilihan split (perlu keputusan)

Simulasi read-only atas 705 gambar:

| Opsi | train | val | test | bus di train/val/test |
|------|-------|-----|------|----------------------|
| Blok 30 + batas video, 80/20 | 585 | 120 | - | 14 / 2 |
| Video-level: KIRI latih, TENGAH-7 val, TENGAH-9 test | 355 | 179 | 171 | 8 / 7 / 1 |
| Video-level + TENGAH-7 ikut latih | 534 | 179 | 171 | 15 / 7 / 1 |

Video-level mengukur generalisasi antar posisi kamera (lebih jujur untuk
studi kasus gerbang), tapi hanya punya 4 video sehingga tiap split didominasi
satu kamera dan variansnya tinggi.

### 0.5 Split TIDAK akan memperbaiki kelas bus

Setelah disimulasikan, kelas bus tetap tipis di semua opsi (maksimal 7
instans di val). Yang menentukan adalah menambah anotasi bus: sekarang hanya
**15 dari 705 gambar** yang memuat bus, total 16 instans.

Urutan yang masuk akal:
1. Anotasi lebih banyak gambar bus (target minimal 150-200 instans).
2. Regenerate split dengan `--action split` (group-aware + batas video).
3. Training ulang.
4. Evaluasi di test **satu kali saja** di akhir.

---

## LANGKAH 1: Auto-Anotasi Dataset

### 1.1 Auto-Annotate dengan YOLOv11 Pretrained

Script `src/auto_annotate.py` menggunakan model YOLOv11 yang sudah dilatih di COCO dataset untuk otomatis melabeli kendaraan.

```bash
# Cara otomatis (disarankan): stage + annotate sekaligus ke data/staging,
# tanpa menyentuh label review manual di data/annotated
python main.py workflow --stage stage
python main.py workflow --stage annotate

# Cara manual per split (seperti di bawah ini)
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
python src/train.py            # training penuh
python src/train.py --quick    # 10 epoch, untuk cek pipeline
```

`src/train.py` membaca `config/config.yaml` dan memakai
`project=models` sehingga output ada di
`runs/detect/models/vehicle_detection/`. Jangan memanggil `model.train()`
manual dengan path hardcode - itu sempat membuat dokumen dan kode
menunjuk ke folder berbeda.

`yolo11n.pt` dipakai HANYA sebagai arsitektur awal. Setelah training,
seluruh entry point (deteksi, realtime, CCTV, GUI, evaluasi, ekspor)
memakai `best.pt` hasil training lewat `utils.paths.get_model_path()`.

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

### 3.2 Hasil Evaluasi (run terakhir)

| Metric | Nilai | Catatan |
|--------|-------|---------|
| **mAP50** | 0.3850 | target skripsi 0.75, belum tercapai |
| **mAP50-95** | 0.2993 | target 0.5, belum tercapai |
| **Precision** | 0.6027 | |
| **Recall** | 0.4121 | titik lemah utama |
| **F1** | 0.4850 | |
| **FPS** | 32.9 | 416x416, CPU, tanpa tracking |

Confusion matrix (IoU>=0.5, conf>=0.5): TP 68, FP 17, FN 40
(precision 0.80, recall 0.63 pada ambang itu).

### 3.3 Per Kelas

| Kelas | AP50 | AP50-95 | Precision | Recall | F1 |
|-------|------|---------|-----------|--------|-----|
| mobil | 0.6922 | 0.5597 | 0.9184 | 0.7031 | 0.7965 |
| motor | 0.4529 | 0.3054 | 0.6923 | 0.5455 | 0.6102 |
| truk | 0.3950 | 0.3322 | 0.8000 | 0.4000 | 0.5333 |
| bus | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Kelas `bus` tidak menghasilkan satu pun true positive. Ini konsisten dengan
hasil run penuh sebelumnya. Recall overall yang rendah lebih dipengaruhi
kombinasi kelas langka dan jumlah contoh anotasi `bus` yang terlalu sedikit -
bukan oleh bug di pipeline. Prioritas yang masuk akal: tambah contoh anotasi
`bus`, lalu training ulang.

### 3.4 Output Files

Semua keluaran evaluasi dikumpulkan di `outputs/evaluation/`:

```
outputs/evaluation/
├── evaluation_report.json     ← metrik overall + per kelas + confusion matrix
├── evaluation_metrics.csv
├── evaluation_per_class.csv
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── PR_curve.png              ← disalin dari BoxPR_curve.png
├── F1_curve.png              ← disalin dari BoxF1_curve.png
├── P_curve.png               ← disalin dari BoxP_curve.png
└── R_curve.png               ← disalin dari BoxR_curve.png
```

Perbandingan antar model:

```bash
python src/comparison.py outputs/evaluation
python src/comparison.py --per-class outputs/evaluation/evaluation_report.json
```

---

## LANGKAH 4: Deteksi + ROI + Tracking

### 4.1 Pipeline Lengkap

```bash
# Webcam dengan ROI + tracking + counting
python src/detect_with_tracking.py --source 0 --show

# Video file, simpan video teranotasi
python src/detect_with_tracking.py --source "path/video.mp4" --output output.mp4

# Simpan ke database SQLite
python src/detect_with_tracking.py --source "path/video.mp4" --db

# Mode headless dengan config lain (mis. ROI dimatikan sementara)
python src/detect_with_tracking.py --config config/uji_roi_off.yaml --source video.mp4
```

Semua perintah di atas memakai `VehiclePipeline` yang sama dengan GUI,
CCTV, dan `realtime.py`. Angka "kendaraan" yang dicetak berasal dari
`counter.total_count` (track unik yang melewati dua garis), bukan dari jumlah
baris database. Baris database = 1 per track per frame, jadi satu kendaraan
yang terlihat 80 frame menghasilkan 80 baris.

### 4.2 ROI Configuration

```yaml
# config/config.yaml - nilai yang benar-benar dipakai saat ini
roi:
  enabled: true
  reference_resolution:      # koordinat boundary di bawah ini memakai skala ini
    width: 1920
    height: 1080
  boundary:
    top_left: [60, 497]
    top_right: [390, 484]
    bottom_left: [111, 1007]
    bottom_right: [979, 822]
  min_bbox_height: 20
  draw_roi: true
```

`reference_resolution` dipakai untuk menskalakan boundary ke resolusi video
yang sebenarnya. Nilai ini hanya menyisakan sekitar 13.8% dari area frame
1920x1080, jadi pada video operasional banyak kendaraan bisa terbuang.
Kalibrasi boundary terhadap rekaman gerbang adalah pekerjaan yang belum
selesai dan tidak bisa ditebak tanpa rekaman.

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
│   ├── dataset.yaml             ← path absolut, ditulis dataset_prepare
│   ├── detections.db            ← SQLite hasil counting
│   ├── raw/                     ← frame hasil ekstraksi video
│   └── annotated/
│       ├── images/
│       │   ├── train/           ← 564 gambar (177 label kosong = background)
│       │   ├── val/             ← 71 gambar (18 kosong)
│       │   └── test/            ← 70 gambar (13 kosong), dipakai evaluasi manual
│       └── labels/              ← YOLO txt, sejajar dengan images/
│
├── src/
│   ├── pipeline.py              ← VehiclePipeline: detect → ROI → track → count
│   ├── detect_with_tracking.py  ← CLI video/webcam + ringkasan
│   ├── train.py                 ← training YOLO11n CPU
│   ├── evaluate.py              ← evaluasi + laporan JSON/CSV + confusion matrix
│   ├── detect.py                ← deteksi satu gambar / folder
│   ├── batch_process.py         ← batch gambar tanpa tracking
│   ├── realtime.py              ← wrapper webcam real-time
│   ├── monitor.py               ← alias realtime.main
│   ├── cctv_connect.py          ← CCTV/RTSP, reconnect, simpan config
│   ├── gui_app.py               ← GUI Tkinter (thread-safe)
│   ├── dataset_prepare.py       ← validate / split group-aware / split-report / labelme / yaml
│   ├── dataset_collect.py       ← capture webcam, ekstrak frame dari video
│   ├── extract_frames.py        ← ekstraksi frame + info video
│   ├── auto_annotate.py         ← auto-label dari model COCO
│   ├── annotation_helper.py     ← template LabelMe + konversi YOLO
│   ├── export_model.py          ← ekspor ONNX/TorchScript/dll
│   ├── comparison.py            ← bandingkan laporan antar model
│   ├── query_db.py              ← baca & ekspor database
│   └── utils/
│       ├── paths.py             ← root, resolver model, class mapping
│       ├── database.py          ← schema, migrasi, UPSERT
│       ├── counter.py           ← dual-line counting
│       ├── roi_filter.py        ← filter & scaling ROI
│       ├── tracker.py           ← object tracking (sumber track_id)
│       ├── visualizer.py
│       └── metrics.py
│
├── runs/detect/models/vehicle_detection/
│   ├── weights/
│   │   ├── best.pt              ← dipakai semua entry point
│   │   ├── last.pt
│   │   └── epoch*.pt
│   ├── results.csv
│   ├── Box*_curve.png           ← disalin evaluate.py ke outputs/evaluation
│   └── confusion_matrix.png
│
├── exports/                     ← hasil export_model.py
├── outputs/
│   ├── detections/              ← gambar teranotasi
│   ├── evaluation/              ← laporan + kurva
│   └── *.mp4                    ← video teranotasi
│
├── WORKFLOW.md                  ← Dokumen ini
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

1. **Jangan percaya angka mAP lama.** Aksesori report 72.6% tidak bisa
   direproduksi; ukur ulang dengan `evaluate.py --task all` setelah model
   berubah.
2. **Kelas `bus` = 0 AP50.** Menambah contoh anotasi bus memberi dampak
   paling besar dibanding tuning inference.
3. **ROI boundary harus diukur dari rekaman gerbang**, bukan ditebak. Nilai
   sekarang hanya mencakup ~13.8% area frame.
4. **Jangan tambah track_id baru.** Semua modul memakai `utils/tracker.py`
   sebagai sumber tunggal; ID dari modul lain akan bentrok dengan
   `detections.vehicle_id` di database.
5. **Jumlah kendaraan = `counter.total_count`**, bukan `COUNT(*)` dari tabel
   `detections`. Tabel itu satu baris per track per frame.
6. **Split group-aware harus dipakai sebelum training berikutnya**, kalau
   tidak metrik validasi tetap tergelembung oleh frame berdekatan. UKur dulu
   dengan `python src/dataset_prepare.py --action split-report`. Rinciannya
   di BAGIAN 0.
7. **Jalankan skrip dari root project** atau pakai `--config` eksplisit.
   Semua path internal sudah absolut terhadap root, jadi aman, tapi output
   ad-hoc tetap mengikuti folder kerja.
