"""
Script Persiapan Dataset untuk Deteksi Kendaraan (YOLO11).

Perbaikan terhadap versi lama:
- ``config["dataset"]["names"]`` dibaca sebagai dict (YAML ``0: motor``),
  bukan diindeks dengan integer. Versi lama menulis
  ``if cls_id in self.class_names`` yang selalu False untuk dict berkey
  string, sehingga semua kelas dilaporkan "tidak dikenal".
- Split memakai ``group_size`` dari config untuk memisahkan frame dari video
  yang sama. Versi lama shuffle acak, jadi frame berdekatan dari satu video
  bisa masuk train dan val sekaligus - val jadi bocor (leakage) dan mAP
  terlihat lebih bagus dari kenyataan.
- Image extensions lengkap, dan label kosong dilaporkan sebagai "background"
  (gambar tanpa kendaraan) - ini valid untuk deteksi, bukan error.
"""

import argparse
import json
import os
import random
import shutil
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.paths import load_config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def rmtree_force(path: Path) -> None:
    """
    Hapus folder beserta isinya, tahan atribut ReadOnly (khas Windows).

    Folder dataset kadang terlanjur punya atribut ReadOnly (mis. hasil
    ekstrak arsip atau setelan lama). Dalam kasus itu shutil.rmtree
    menghapus semua isi dengan sukses lalu GAGAL di os.rmdir terakhir
    dengan WinError 5 "Access is denied" - split --clean berhenti di
    tengah dan train/val jadi setengah kosong.
    """
    for target in [path, *path.rglob("*")]:
        try:
            # Di Windows chmod(S_IWRITE) = melepas flag read-only.
            os.chmod(target, stat.S_IWRITE)
        except OSError:
            pass
    shutil.rmtree(path)


def list_images(directory: Path) -> List[Path]:
    """Semua file gambar di ``directory``, terurut."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def group_key(stem: str, group_size: int, video_ranges: Optional[Dict] = None) -> str:
    """
    Kunci grup untuk mencegah leakage antar split.

    Frame dari video yang sama sangat mirip antar frame berdekatan
    (kendaraan hampir tidak bergerak selama beberapa frame). Kalau frame
    berdekatan masuk train DAN val sekaligus, val bukan lagi data baru -
    model hanya perlu mengingat frame yang sudah dilihat, dan mAP
    validasi jadi terlalu tinggi.

    Dua bentuk nama ditangani:

    1. Ada prefix per video, mis. ``KIRI7_000123`` -> grup ``KIRI7``.
       Prefix sebelum underscore terakhir yang berupa angka.
    2. Nama polos bernomorurut dari satu stream gabungan, mis.
       ``merged_00010`` .. ``merged_00049`` -> grup berbasis blok nomor:
       ``10 // group_size = 0``, ``49 // group_size = 1``.

    Bentuk kedua penting untuk dataset di repo ini: seluruh frame bernama
    ``merged_NNNNN`` tanpa prefix video, jadi kalau hanya dipisah di
    underscore, SEMUA frame jatuh ke satu grup dan val jadi kosong.

    ``video_ranges`` (dict nama video -> [awal, akhir]) membagi nomor frame
    lebih dulu per video. Tanpa itu, blok ``group_size`` bisa memotong batas
    antar video: pada dataset di repo ini blok 30 memotong batas di 3 titik,
    sehingga dua kamera berbeda berakhir di grup yang sama.
    """
    if group_size <= 1:
        return stem

    parts = stem.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        return stem

    prefix, num = parts[0], int(parts[1])

    # Batas video: nomor frame dipetakan ke video asalnya lebih dulu.
    if video_ranges:
        for name, span in video_ranges.items():
            try:
                lo, hi = int(span[0]), int(span[1])
            except (TypeError, ValueError, IndexError):
                continue
            if lo <= num <= hi:
                return f"{name}#{num // group_size}"

    return f"{prefix}#{num // group_size}"

    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        prefix, num = parts[0], int(parts[1])
        return f"{prefix}#{num // group_size}"

    return stem


class DatasetPreparator:
    """
    Persiapan dan validasi dataset untuk training YOLO11.

    Fitur:
    - Validasi struktur dataset dan label
    - Split dataset menjadi train/val tanpa leakage antar video
    - Buat file dataset.yaml
    - Perbaiki label yang hilang
    - Konversi anotasi LabelMe ke format YOLO
    """

    def __init__(self, config: dict):
        self.config = config
        self.dataset_cfg = config["dataset"]
        # names dari YAML adalah dict {0: motor, ...}; pertahankan sebagai dict
        raw = self.dataset_cfg["names"] or {}
        self.class_names: Dict[int, str] = {int(k): str(v) for k, v in raw.items()}
        if not self.class_names:
            raise ValueError("dataset.names kosong di config")
        self.group_size = int(self.dataset_cfg.get("group_size", 1) or 1)
        # Batas antar video (lihat group_key). Kosong = tidak ada.
        # Span yang rusak diabaikan, bukan membuat seluruh tool gagal:
        # config.yaml ditulis tangan, satu typo tidak boleh menghentikan
        # validasi atau split.
        raw_ranges = self.dataset_cfg.get("video_ranges") or {}
        self.video_ranges: Dict[str, List[int]] = {}
        for name, span in raw_ranges.items():
            try:
                lo, hi = int(span[0]), int(span[1])
            except (TypeError, ValueError, IndexError, KeyError):
                print(f"[PERINGATAN] video_ranges['{name}'] tidak valid "
                      f"({span!r}), dilewati")
                continue
            if hi < lo:
                print(f"[PERINGATAN] video_ranges['{name}'] terbalik "
                      f"({lo} > {hi}), dilewati")
                continue
            self.video_ranges[str(name)] = [lo, hi]
        if self.video_ranges:
            covered = sum(hi - lo + 1 for lo, hi in self.video_ranges.values())
            print(f"[INFO] Batas video aktif ({len(self.video_ranges)} video, "
                  f"{covered} frame tercakup)")

    def key_for(self, stem: str) -> str:
        """Kunci grup dengan batas video yang dikonfigurasi."""
        return group_key(stem, self.group_size, self.video_ranges)

    # ------------------------------------------------------------------
    # Validasi
    # ------------------------------------------------------------------

    def validate_dataset(self) -> Tuple[dict, list]:
        """
        Validasi struktur dataset dan label.

        Label kosong TIDAK dianggap error: pada dataset deteksi, gambar tanpa
        kendaraan memang diberi label kosong agar model belajar background. Yang
        dianggap masalah adalah kelas di luar mapping dan nilai di luar rentang.

        Returns:
            Tuple (stats, issues)
        """
        print("=" * 60)
        print("VALIDASI DATASET")
        print("=" * 60)

        issues: List[str] = []
        stats = {s: {"images": 0, "labels": 0, "empty_labels": 0, "classes": Counter(),
                     "missing_labels": 0, "bad_values": 0}
                 for s in ("train", "val", "test")}

        for split in ("train", "val", "test"):
            img_dir = Path(self.dataset_cfg.get(f"{split}_images", ""))
            lbl_dir = Path(self.dataset_cfg.get(f"{split}_labels", ""))
            if not img_dir.is_dir():
                continue

            images = list_images(img_dir)
            stats[split]["images"] = len(images)
            if not lbl_dir.is_dir():
                issues.append(f"Folder label tidak ditemukan: {lbl_dir}")
                continue

            for img_file in images:
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if not lbl_file.is_file():
                    stats[split]["missing_labels"] += 1
                    issues.append(f"Label hilang: {lbl_file}")
                    continue
                stats[split]["labels"] += 1

                lines = [ln for ln in
                         lbl_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if not lines:
                    stats[split]["empty_labels"] += 1
                    continue

                for line in lines:
                    parts = line.split()
                    if len(parts) < 5:
                        stats[split]["bad_values"] += 1
                        issues.append(f"Baris label tidak lengkap: {lbl_file}")
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                        cx, cy, w, h = (float(v) for v in parts[1:5])
                    except ValueError:
                        stats[split]["bad_values"] += 1
                        issues.append(f"Nilai label tidak bisa dibaca: {lbl_file}")
                        continue

                    if cls_id not in self.class_names:
                        stats[split]["bad_values"] += 1
                        issues.append(
                            f"Kelas di luar mapping: id={cls_id} di {lbl_file} "
                            f"(mapping: {self.class_names})")
                        continue
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        stats[split]["bad_values"] += 1
                        issues.append(
                            f"Nilai di luar rentang 0-1: {lbl_file} -> {parts[1:5]}")
                        continue
                    stats[split]["classes"][self.class_names[cls_id]] += 1

        for split, s in stats.items():
            if not s["images"]:
                continue
            print(f"\n[{split.upper()}]")
            print(f"  Gambar            : {s['images']}")
            print(f"  Ada label         : {s['labels']}")
            print(f"  Label kosong      : {s['empty_labels']}  (gambar tanpa kendaraan)")
            print(f"  Label hilang      : {s['missing_labels']}")
            print(f"  Nilai tidak valid : {s['bad_values']}")
            print(f"  Instansi per kelas: {dict(s['classes'])}")

        # Peringatan ketidakseimbangan kelas
        train_cls = stats["train"]["classes"]
        if train_cls:
            total = sum(train_cls.values())
            print(f"\n[INFO] Total kelas di train: {total}")
            for name, n in sorted(train_cls.items()):
                pct = n / total * 100
                flag = "  <-- sangat sedikit" if pct < 1 else ""
                print(f"  {name:6s}: {n:6d} ({pct:5.2f}%){flag}")

        if issues:
            print(f"\n[PERINGATAN] {len(issues)} masalah ditemukan. 10 pertama:")
            for issue in issues[:10]:
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... dan {len(issues) - 10} lagi")
        else:
            print("\n[OK] Validasi dataset berhasil, tidak ada masalah.")

        return stats, issues

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    @staticmethod
    def _count_classes(groups: Dict[str, List[Path]],
                       labels_dir: Path) -> Dict[str, Counter]:
        """Hitung instans tiap kelas per grup dari file label YOLO."""
        counts: Dict[str, Counter] = {}
        for key, files in groups.items():
            per_class: Counter = Counter()
            for f in files:
                lbl = labels_dir / (f.stem + ".txt")
                if not lbl.is_file():
                    continue
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        per_class[int(float(parts[0]))] += 1
                    except ValueError:
                        continue  # baris rusak diabaikan (validate nanti lapor)
            counts[key] = per_class
        return counts

    @staticmethod
    def _stratified_group_split(groups: Dict[str, List[Path]],
                                counts: Dict[str, Counter],
                                keys: List[str],
                                train_ratio: float) -> Tuple[List[str], List[str]]:
        """
        Bagi grup utuh ke train/val dengan stratifikasi kelas (greedy).

        Grup diurutkan acak dulu, lalu tiap grup ditaruh ke split yang
        menaikkan deviasi distribusi kelas PALING KECIL. Deviasi dinormalkan
        per kelas (dibagi jumlah instans kelas itu), jadi kelas langka seperti
        bus ikut menentukan - tanpa ini, shuffle polos bisa menghabiskan
        hampir semua instance bus ke train dan val hanya menyisakan 1 bus,
        sehingga metrik val tidak bisa mengukur kelas itu sama sekali.

        Target per split: ``train_ratio`` dari tiap kelas DAN dari jumlah
        gambar. Grup tidak pernah dipecah (tetap anti-leakage).
        """
        total: Counter = Counter()
        for per_class in counts.values():
            total.update(per_class)
        n_total = sum(len(v) for v in groups.values())
        if n_total == 0 or not total:
            # Tidak ada instans sama sekali: pecah berdasar gambar saja.
            target_img = n_total * train_ratio
            train_keys, acc = [], 0
            for k in keys:
                if acc >= target_img:
                    break
                train_keys.append(k)
                acc += len(groups[k])
            train_set = set(train_keys)
            return train_keys, [k for k in keys if k not in train_set]

        target = {
            "train": {c: total[c] * train_ratio for c in total},
            "val": {c: total[c] * (1 - train_ratio) for c in total},
        }
        target_img = {"train": n_total * train_ratio,
                      "val": n_total * (1 - train_ratio)}
        assigned = {"train": Counter(), "val": Counter()}
        imgs = {"train": 0, "val": 0}
        result: Dict[str, List[str]] = {"train": [], "val": []}

        def delta(split: str, g: Counter, n: int) -> float:
            """Kenaikan deviasi bila grup masuk ke ``split``."""
            d = 0.0
            for c, target_c in target[split].items():
                before = abs(assigned[split][c] - target_c)
                after = abs(assigned[split][c] + g[c] - target_c)
                d += (after - before) / max(total[c], 1)
            d += (abs(imgs[split] + n - target_img[split])
                  - abs(imgs[split] - target_img[split])) / max(n_total, 1)
            return d

        for key in keys:
            g = counts[key]
            n = len(groups[key])
            # Urutan tetap train dulu supaya seri (deviasi sama) deterministik.
            cost_train, cost_val = delta("train", g, n), delta("val", g, n)
            chosen = "train" if cost_train <= cost_val else "val"
            result[chosen].append(key)
            assigned[chosen].update(g)
            imgs[chosen] += n

        return result["train"], result["val"]

    def select_test_groups(self, source_dir, test_ratio: float = 0.1,
                           seed: int = 42) -> dict:
        """
        Pilih beberapa grup UTUH sebagai split test, tersebar antar video.

        Kenapa grup utuh: frame dalam satu grup adalah potongan video yang
        berdekatan (kendaraan yang sama bisa muncul di frame tetangga).
        Kalau test berisi frame acak, sisanya tetap ada di train/val -
        model sudah "melihat" sekitarnya dan metrik test tergelembung.

        Kenapa tersebar: test yang hanya dari satu kamera tidak
        merepresentasikan kondisi gerbang. Grup diacak per video lalu
        dipilih round-robin antar video sampai mencapai ``test_ratio``.

        Seleksi deterministik (seed) - rebuild ulang dengan seed sama
        menghasilkan test set yang sama.

        Args:
            source_dir: pool berisi images/ dan labels/ (data/staging)
            test_ratio: porsi target frame untuk test (default 0.1)
            seed: seed pengacakan pemilihan grup

        Returns:
            dict jumlah grup, frame, dan sebaran per video
        """
        print("\n" + "=" * 60)
        print("SELEKSI TEST SET (grup utuh, tersebar antar video)")
        print("=" * 60)

        source = Path(source_dir)
        images_dir = source / "images"
        labels_dir = source / "labels"
        image_files = list_images(images_dir)
        if not image_files:
            raise FileNotFoundError(f"Tidak ada gambar di {images_dir}")
        if not 0 < test_ratio < 1:
            raise ValueError(f"--test-ratio harus di antara 0 dan 1, "
                             f"dapat {test_ratio}")

        groups: Dict[str, List[Path]] = {}
        for f in image_files:
            groups.setdefault(self.key_for(f.stem), []).append(f)

        # Kunci grup berbentuk "NAMA_VIDEO#blok" (video_ranges aktif) atau
        # "prefix#blok". Tanpa video_ranges semua frame dari video beda bisa
        # berada di satu "video" - sebaran round-robin tetap dijalankan.
        per_video: Dict[str, List[str]] = {}
        for key in sorted(groups):
            video = key.split("#", 1)[0]
            per_video.setdefault(video, []).append(key)

        rng = random.Random(seed)
        for keys in per_video.values():
            rng.shuffle(keys)

        target = len(image_files) * test_ratio
        chosen: List[str] = []
        total = 0
        cursor = {v: 0 for v in sorted(per_video)}
        while total < target:
            progressed = False
            for video in sorted(per_video):
                i = cursor[video]
                if i >= len(per_video[video]):
                    continue
                cursor[video] = i + 1
                chosen.append(per_video[video][i])
                total += len(groups[per_video[video][i]])
                progressed = True
                if total >= target:
                    break
            if not progressed:
                break

        if not chosen:
            raise RuntimeError("Tidak ada grup terpilih untuk test.")

        test_images_dir = Path(self.dataset_cfg["test_images"])
        test_labels_dir = Path(self.dataset_cfg["test_labels"])
        if test_images_dir.is_dir():
            rmtree_force(test_images_dir)
        if test_labels_dir.is_dir():
            rmtree_force(test_labels_dir)
        test_images_dir.mkdir(parents=True, exist_ok=True)
        test_labels_dir.mkdir(parents=True, exist_ok=True)

        per_video_count: Dict[str, int] = {}
        empty = 0
        for key in sorted(chosen):
            video = key.split("#", 1)[0]
            for img in groups[key]:
                shutil.copy2(img, test_images_dir / img.name)
                per_video_count[video] = per_video_count.get(video, 0) + 1
                lbl = labels_dir / (img.stem + ".txt")
                dst_lbl = test_labels_dir / (img.stem + ".txt")
                if lbl.is_file():
                    shutil.copy2(lbl, dst_lbl)
                else:
                    dst_lbl.write_text("")
                    empty += 1

        print(f"Target           : {test_ratio * 100:.0f}% dari "
              f"{len(image_files)} frame ({target:.0f})")
        print(f"Grup terpilih    : {len(chosen)} grup -> {total} frame")
        for video in sorted(per_video_count):
            print(f"  {video:<10} {per_video_count[video]:>4} frame")
        if empty:
            print(f"[OK] {empty} label kosong dibuat (tanpa kendaraan)")

        return {"groups": len(chosen), "frames": total,
                "per_video": per_video_count}

    def split_dataset(self, source_dir, train_ratio: float = 0.8, seed: int = 42,
                      clean: bool = False) -> dict:
        """
        Split dataset menjadi train/val TANPA leakage antar video.

        Frame dikelompokkan berdasarkan prefix nama (lihat ``group_key``),
        lalu grup utuh diletakkan ke train atau val. Split acak per gambar
        (versi lama) membuka kemungkinan frame berdekatan dari video yang
        sama tersebar ke dua split, sehingga metrik validasi tergelembung.

        Args:
            source_dir: direktori sumber (berisi images/ dan labels/)
            train_ratio: rasio data train (0-1)
            seed: seed untuk random shuffle antar grup
            clean: hapus isi folder train/val tujuan sebelum menyalin
        """
        print("\n" + "=" * 60)
        print("SPLIT DATASET (group-aware, tanpa leakage)")
        print("=" * 60)

        source = Path(source_dir)
        images_dir = source / "images"
        labels_dir = source / "labels"

        if not images_dir.is_dir():
            raise FileNotFoundError(f"Direktori gambar tidak ditemukan: {images_dir}")

        image_files = list_images(images_dir)
        if not image_files:
            raise FileNotFoundError(f"Tidak ada gambar di {images_dir}")

        # Split test adalah hold-out: frame yang namanya sudah ada di test
        # tidak boleh ikut train/val, walau isinya nanti berbeda. Tanpa ini,
        # split_dataset (yang membagi SEMUA gambar sumber) menaruh frame
        # dengan nama yang sama ke dua sisi sekaligus.
        test_images_cfg = self.dataset_cfg.get("test_images")
        test_labels_cfg = self.dataset_cfg.get("test_labels")
        test_files = list_images(Path(test_images_cfg)) if test_images_cfg else []
        test_stems = {p.stem for p in test_files}
        if test_stems:
            kept = [f for f in image_files if f.stem not in test_stems]
            if len(kept) != len(image_files):
                print(f"[INFO] {len(image_files) - len(kept)} frame "
                      "dieksklusi dari train/val (sudah di split test)")
            image_files = kept
            if not image_files:
                raise FileNotFoundError(
                    "Semua frame sumber sudah berada di split test - "
                    "tidak ada sisa untuk train/val.")

        groups: Dict[str, List[Path]] = {}
        for f in image_files:
            groups.setdefault(self.key_for(f.stem), []).append(f)

        keys = sorted(groups)
        rng = random.Random(seed)
        rng.shuffle(keys)

        group_counts = self._count_classes(groups, labels_dir)
        train_keys, val_keys = self._stratified_group_split(
            groups, group_counts, keys, train_ratio)

        train_files = [f for k in train_keys for f in groups[k]]
        val_files = [f for k in val_keys for f in groups[k]]

        print(f"Total gambar     : {len(image_files)}")
        print(f"Jumlah grup      : {len(groups)} (group_size={self.group_size})")
        print(f"Train            : {len(train_files)} gambar "
              f"({len(train_keys)} grup, {train_ratio * 100:.0f}% target)")
        print(f"Val              : {len(val_files)} gambar ({len(val_keys)} grup)")
        if any(group_counts.values()):
            train_cls, val_cls = Counter(), Counter()
            for k in train_keys:
                train_cls.update(group_counts[k])
            for k in val_keys:
                val_cls.update(group_counts[k])
            print("Instans per kelas (train / val):")
            for c in sorted(set(train_cls) | set(val_cls)):
                name = self.class_names.get(c, str(c))
                print(f"  {name:<8} {train_cls[c]:>5} / {val_cls[c]}")
        if set(train_keys) & set(val_keys):
            raise AssertionError("Leakage: grup ada di train DAN val")

        if clean:
            # Bersihkan gambar DAN label. Versi lama hanya menghapus folder
            # gambar, jadi label lama ikut tertinggal. Kalau split dijalankan
            # ulang dengan seed berbeda, label dari frame yang sekarang ada di
            # val bisa tetap ada di train - dan Ultralytics membaca label
            # berdasarkan nama file, sehingga gambar train diam-diam memakai
            # label yang salah.
            for split in ("train", "val"):
                for key in (f"{split}_images", f"{split}_labels"):
                    d = Path(self.dataset_cfg[key])
                    if d.is_dir():
                        rmtree_force(d)

        for split in ("train", "val"):
            Path(self.dataset_cfg[f"{split}_images"]).mkdir(parents=True, exist_ok=True)
            Path(self.dataset_cfg[f"{split}_labels"]).mkdir(parents=True, exist_ok=True)

        copied = {"train": 0, "val": 0}
        no_label = 0
        for split_name, files in (("train", train_files), ("val", val_files)):
            dst_img_dir = Path(self.dataset_cfg[f"{split_name}_images"])
            dst_lbl_dir = Path(self.dataset_cfg[f"{split_name}_labels"])
            for img_file in files:
                shutil.copy2(img_file, dst_img_dir / img_file.name)
                lbl_file = labels_dir / (img_file.stem + ".txt")
                if lbl_file.is_file():
                    shutil.copy2(lbl_file, dst_lbl_dir / lbl_file.name)
                else:
                    no_label += 1
                copied[split_name] += 1
            print(f"[OK] {split_name}: {copied[split_name]} gambar disalin")

        # Gambar tanpa label = tidak ada kendaraan, tapi Ultralytics tetap
        # butuh file .txt (boleh kosong). Tanpa ini, gambar diabaikan dan
        # background negatif hilang dari training, sehingga model jadi lebih
        # mudah salah mendeteksi.
        created_empty = 0
        for split_name in ("train", "val"):
            dst_img_dir = Path(self.dataset_cfg[f"{split_name}_images"])
            dst_lbl_dir = Path(self.dataset_cfg[f"{split_name}_labels"])
            for img_file in sorted(dst_img_dir.iterdir()):
                if img_file.suffix.lower() not in IMAGE_EXTS:
                    continue
                lbl = dst_lbl_dir / (img_file.stem + ".txt")
                if not lbl.is_file():
                    lbl.touch()
                    created_empty += 1
        if created_empty:
            print(f"[OK] {created_empty} label kosong dibuat untuk gambar "
                  "tanpa kendaraan")

        if no_label:
            print(f"[PERINGATAN] {no_label} gambar tidak punya file label. "
                  "Jalankan --action fix-labels bila ini tidak disengaja.")

        # Split test juga harus berasal dari versi pool yang sama dengan
        # train/val. Frame test yang dibiarkan dari ekstraksi lama bisa
        # IDENTIK dengan frame train baru (beda nama, satu frame sumber
        # yang sama) - yaitu leakage test -> train.
        if clean and test_files:
            self._refresh_test_split(source, test_files, test_labels_cfg)

        return {"train": copied["train"], "val": copied["val"],
                "groups": len(groups), "no_label": no_label}

    def _refresh_test_split(self, source: Path, test_files: List[Path],
                            test_labels_cfg: Optional[str]) -> None:
        """
        Timpa gambar/label test dengan versi terbaru dari ``source``.

        Nama file test dipertahankan (itu hold-out terpilih), hanya isi dan
        label-nya yang disegarkan dari pool staging supaya konsisten dengan
        train/val hasil split ini. Label kosong dibuat bila sumber tidak
        punya (gambar tanpa kendaraan).
        """
        src_images = source / "images"
        src_labels = source / "labels"
        dst_labels = Path(test_labels_cfg) if test_labels_cfg else None

        refreshed = empty = skipped = 0
        for img in test_files:
            src_img = src_images / img.name
            if not src_img.is_file():
                skipped += 1
                print(f"  [LEWAT] {img.name} tidak ada di pool sumber "
                      "(test tetap memakai isi lama)")
                continue
            shutil.copy2(src_img, img)
            refreshed += 1

            src_lbl = src_labels / (img.stem + ".txt")
            dst_lbl = dst_labels / (img.stem + ".txt") if dst_labels else None
            if dst_lbl is None:
                continue
            if src_lbl.is_file():
                shutil.copy2(src_lbl, dst_lbl)
            else:
                dst_lbl.write_text("")
                empty += 1

        if refreshed:
            print(f"[OK] test: {refreshed} gambar disegarkan dari pool "
                  f"({empty} label kosong dibuat)")
        if skipped:
            print(f"[PERINGATAN] {skipped} gambar test tidak ada di pool - "
                  "pertimbangkan regenerasi split test.")

    # ------------------------------------------------------------------
    # Laporan split tanpa mengubah file
    # ------------------------------------------------------------------

    def report_split_leakage(self) -> dict:
        """
        Ukur seberapa bocor split yang sedang dipakai, TANPA mengubah file.

        Split yang aktif sekarang adalah hasil ``split`` lama (acak per
        gambar) sehingga blok frame berdekatan bisa berada di train, val,
        dan test sekaligus. Fungsi ini menghitungnya supaya keputusan untuk
        regenerate split bisa diambil berdasarkan angka, bukan dugaan.

        Yang dihitung:

        * Berapa grup (blok ``group_size`` frame) yang pecah ke >1 split.
        * Untuk tiap grup, berapa frame tiap split - jadi kebocoran terlihat
          per-instance, bukan cuma "ada yang bocor".
        * Distribusi kelas per split, supaya terlihat kelas mana yang tipis.

        Tidak ada file yang ditulis. Jalankan sebelum ``--action split``.
        """
        print("\n" + "=" * 60)
        print("LAPORAN SPLIT (read-only, tidak mengubah file)")
        print("=" * 60)

        images_root = Path(self.dataset_cfg["val_images"]).parent
        split_dirs = {}
        for split in ("train", "val", "test"):
            d = images_root / split
            if d.is_dir():
                split_dirs[split] = {p.stem: p for p in list_images(d)}
        if not split_dirs:
            raise FileNotFoundError(
                f"Tidak ada folder split di {images_root}")

        labels_root = images_root.parent / "labels"

        # grup -> split -> jumlah frame
        groups: Dict[str, Dict[str, int]] = {}
        for split, files in split_dirs.items():
            for stem in files:
                key = self.key_for(stem)
                per_split = groups.setdefault(key, {})
                per_split[split] = per_split.get(split, 0) + 1

        leaky = {g: c for g, c in groups.items() if len(c) > 1}
        total_frames = sum(len(f) for f in split_dirs.values())
        leaky_frames = sum(sum(c.values()) for c in leaky.values())

        print(f"\nFolder gambar    : {images_root}")
        for split, files in split_dirs.items():
            print(f"  {split:<5} : {len(files)} gambar")
        print(f"Jumlah grup      : {len(groups)} (group_size={self.group_size})")
        print(f"Grup terbagi ke >1 split : {len(leaky)} "
              f"({len(leaky) * 100 / max(len(groups), 1):.1f}% dari grup)")
        print(f"Frame di grup bocor     : {leaky_frames} "
              f"({leaky_frames * 100 / max(total_frames, 1):.1f}% dari frame)")

        if leaky:
            print("\n[PERINGATAN] Split saat ini BOCOR. Contoh grup yang terpecah:")
            for g, c in sorted(leaky.items())[:5]:
                detail = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
                print(f"    {g}: {detail}")
            print("  Dampak: val/test bukan data baru, jadi metriknya terlalu")
            print("  tinggi dan tidak jujur mencerminkan kemampuan di lapangan.")

        # Distribusi kelas per split
        print("\nInstansi per kelas:")
        class_totals: Dict[str, Dict[str, int]] = {}
        for split in split_dirs:
            lbl_dir = labels_root / split
            counts: Dict[str, int] = {}
            for stem in split_dirs[split]:
                lbl = lbl_dir / f"{stem}.txt"
                if not lbl.is_file():
                    continue
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    try:
                        idx = int(float(parts[0]))
                    except ValueError:
                        continue
                    name = self.class_names.get(idx, f"kelas_{idx}")
                    counts[name] = counts.get(name, 0) + 1
            class_totals[split] = counts

        for split, counts in class_totals.items():
            total = sum(counts.values())
            if not total:
                print(f"  {split:<5} : (tidak ada label)")
                continue
            parts = ", ".join(f"{k}={v} ({v * 100 / total:.1f}%)"
                              for k, v in sorted(counts.items()))
            print(f"  {split:<5} : {total} instans -> {parts}")

        # Kelas minoritas: kalau di split baru kelas ini hilang dari val,
        # validasi jadi tidak bisa mengukur kelas itu sama sekali.
        if "train" in class_totals and "val" in class_totals:
            train_classes = set(class_totals["train"])
            missing_val = train_classes - set(class_totals["val"])
            if missing_val:
                print(f"\n[PERINGATAN] Kelas ada di train tapi tidak di val: "
                      f"{sorted(missing_val)}")

        return {
            "group_size": self.group_size,
            "total_frames": total_frames,
            "total_groups": len(groups),
            "leaky_groups": len(leaky),
            "leaky_frames": leaky_frames,
            "leakage_pct": round(leaky_frames * 100 / max(total_frames, 1), 2),
            "class_totals": class_totals,
        }

    # ------------------------------------------------------------------
    # dataset.yaml
    # ------------------------------------------------------------------

    def create_dataset_yaml(self, output_path="data/dataset.yaml") -> dict:
        """
        Buat dataset.yaml.

        ``path`` ditulis dengan forward slash karena Ultralytics mem-parsing
        YAML lalu memperlakukannya sebagai string; drive letter Windows dengan
        backslash bisa salah di-resolve.
        """
        train_images = Path(self.dataset_cfg["train_images"])
        val_images = Path(self.dataset_cfg["val_images"])
        test_images = Path(self.dataset_cfg.get("test_images", ""))

        root = train_images.parent.parent
        if val_images.parent.parent != root:
            raise ValueError(
                f"train dan val harus satu root yang sama, tapi: "
                f"{train_images} vs {val_images}")

        def rel(split_dir: Path) -> str:
            return (split_dir.parent.relative_to(root) / split_dir.name).as_posix()

        dataset_yaml = {
            "path": root.as_posix(),
            "train": rel(train_images),
            "val": rel(val_images),
            "nc": len(self.class_names),
            "names": self.class_names,
        }
        if test_images.is_dir() and list_images(test_images):
            dataset_yaml["test"] = rel(test_images)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.dump(dataset_yaml, default_flow_style=False,
                                 sort_keys=False), encoding="utf-8")

        print(f"\n[OK] Dataset YAML dibuat: {out}")
        for k, v in dataset_yaml.items():
            print(f"  {k}: {v}")
        return dataset_yaml

    # ------------------------------------------------------------------
    # Perbaikan label
    # ------------------------------------------------------------------

    def fix_missing_labels(self, image_dir, label_dir, overwrite_empty: bool = False) -> int:
        """
        Buat file label kosong untuk gambar tanpa label.

        File label kosong berarti "tidak ada kendaraan di gambar ini", yang
        valid untuk deteksi. File yang SUDAH ada tidak disentuh kecuali
        ``overwrite_empty=True`` (hanya untuk yang isinya kosong).

        Args:
            image_dir: direktori gambar
            label_dir: direktori label
            overwrite_empty: isi ulang file label yang sudah kosong
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        label_dir.mkdir(parents=True, exist_ok=True)

        created = overwritten = 0
        for img_file in list_images(image_dir):
            lbl_file = label_dir / (img_file.stem + ".txt")
            if not lbl_file.exists():
                lbl_file.write_text("", encoding="utf-8")
                created += 1
            elif overwrite_empty and not lbl_file.read_text(encoding="utf-8").strip():
                lbl_file.write_text("", encoding="utf-8")
                overwritten += 1

        print(f"[OK] Label dibuat   : {created}")
        print(f"[OK] Label di-reset : {overwritten}")
        print("    Ingat: label kosong = tidak ada kendaraan pada gambar itu.")
        return created

    # ------------------------------------------------------------------
    # LabelMe -> YOLO
    # ------------------------------------------------------------------

    def convert_labelme_to_yolo(self, labelme_dir, output_dir,
                                class_mapping: Dict[str, int],
                                image_dir=None) -> dict:
        """
        Konversi anotasi LabelMe ke format YOLO.

        YOLO: ``class_id center_x center_y width height`` (ternormalisasi 0-1).

        Koordinat di-clamp ke [0, 1] dan bbox di-degenerate (w atau h = 0)
        DIBUANG, karena YOLO menolak nilai di luar rentang dan box nol tidak
        memberi gradien apa pun saat training.

        Args:
            labelme_dir: direktori berisi JSON LabelMe
            output_dir: direktori output label YOLO
            class_mapping: pemetaan nama kelas -> ID
            image_dir: opsional, untuk memvalidasi ukuran gambar
        """
        labelme_dir = Path(labelme_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        converted = skipped_boxes = skipped_classes = no_image = 0
        for json_file in sorted(labelme_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(f"[PERINGATAN] {json_file.name}: JSON rusak ({exc})")
                continue

            image_w = float(data.get("imageWidth") or 0)
            image_h = float(data.get("imageHeight") or 0)
            if image_w <= 0 or image_h <= 0:
                print(f"[PERINGATAN] {json_file.name}: imageWidth/Height tidak valid")
                no_image += 1
                continue

            yolo_lines = []
            for shape in data.get("shapes", []):
                label = shape.get("label")
                if label not in class_mapping:
                    skipped_classes += 1
                    continue
                points = shape.get("points") or []
                if len(points) < 2:
                    skipped_boxes += 1
                    continue

                xs = [float(p[0]) for p in points]
                ys = [float(p[1]) for p in points]
                # Clamp ke area gambar
                x_min, x_max = max(0.0, min(xs)), min(image_w, max(xs))
                y_min, y_max = max(0.0, min(ys)), min(image_h, max(ys))
                if x_max <= x_min or y_max <= y_min:
                    skipped_boxes += 1
                    continue

                x_c = ((x_min + x_max) / 2.0) / image_w
                y_c = ((y_min + y_max) / 2.0) / image_h
                w = (x_max - x_min) / image_w
                h = (y_max - y_min) / image_h

                yolo_lines.append(f"{class_mapping[label]} {x_c:.6f} {y_c:.6f} "
                                  f"{w:.6f} {h:.6f}")

            (output_dir / (json_file.stem + ".txt")).write_text(
                "\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            converted += 1

        print(f"[OK] File dikonversi        : {converted}")
        print(f"[OK] Box degenerate dilewati: {skipped_boxes}")
        print(f"[OK] Kelas di luar mapping : {skipped_classes}")
        if no_image:
            print(f"[PERINGATAN] File tanpa ukuran gambar: {no_image}")
        return {"converted": converted, "skipped_boxes": skipped_boxes,
                "skipped_classes": skipped_classes, "no_image": no_image}


def main() -> int:
    parser = argparse.ArgumentParser(description="Alat Persiapan Dataset")
    parser.add_argument("--config", default=None, help="path config.yaml")
    parser.add_argument("--action", required=True,
                        choices=["validate", "split", "split-report", "yaml",
                                 "fix-labels", "convert-labelme", "select-test"],
                        help="aksi yang dijalankan. 'split-report' hanya "
                             "melaporkan kebocoran split tanpa mengubah file. "
                             "'select-test' pilih grup utuh sbg test set.")
    parser.add_argument("--source", help="direktori sumber")
    parser.add_argument("--output", help="path/ direktori output")
    parser.add_argument("--ratio", type=float, default=0.8, help="rasio split train")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="porsi frame untuk test set (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true",
                        help="hapus isi folder train/val sebelum split")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    preparator = DatasetPreparator(config)

    try:
        if args.action == "validate":
            preparator.validate_dataset()
        elif args.action == "split":
            if not args.source:
                parser.error("--action split butuh --source")
            preparator.split_dataset(args.source, train_ratio=args.ratio,
                                     seed=args.seed, clean=args.clean)
        elif args.action == "split-report":
            preparator.report_split_leakage()
        elif args.action == "select-test":
            if not args.source:
                parser.error("--action select-test butuh --source")
            preparator.select_test_groups(args.source,
                                          test_ratio=args.test_ratio,
                                          seed=args.seed)
        elif args.action == "yaml":
            preparator.create_dataset_yaml(args.output or "data/dataset.yaml")
        elif args.action == "fix-labels":
            if not args.source or not args.output:
                parser.error("--action fix-labels butuh --source dan --output")
            preparator.fix_missing_labels(args.source, args.output)
        elif args.action == "convert-labelme":
            if not args.source or not args.output:
                parser.error("--action convert-labelme butuh --source dan --output")
            mapping = {name: idx for idx, name in preparator.class_names.items()}
            preparator.convert_labelme_to_yolo(args.source, args.output, mapping)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
