"""
Script Persiapan Dataset untuk Deteksi Kendaraan
"""

import os
import shutil
import yaml
import argparse
import random
from pathlib import Path
from collections import Counter


def load_config(config_path="config/config.yaml"):
    """Memuat file konfigurasi proyek."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DatasetPreparator:
    """
    Persiapan dan validasi dataset untuk training YOLOv8.
    
    Fitur:
    - Validasi struktur dataset dan label
    - Split dataset menjadi train/val
    - Buat file dataset.yaml
    - Perbaiki label yang hilang
    - Konversi anotasi LabelMe ke format YOLO
    """

    def __init__(self, config):
        """
        Inisialisasi preparator dataset.
        
        Args:
            config: dict konfigurasi dari config.yaml
        """
        self.config = config
        self.dataset_cfg = config["dataset"]
        self.class_names = config["dataset"]["names"]

    def validate_dataset(self):
        """
        Validasi struktur dataset dan label.
        
        Pemeriksaan yang dilakukan:
        - Direktori gambar dan label ada
        - Setiap gambar memiliki label yang sesuai
        - Tidak ada kelas yang tidak dikenal
        
        Returns:
            Tuple (stats, issues)
        """
        print("=" * 60)
        print("VALIDASI DATASET")
        print("=" * 60)

        issues = []
        stats = {"train": {"images": 0, "labels": 0, "classes": Counter()},
                 "val": {"images": 0, "labels": 0, "classes": Counter()}}

        # Validasi untuk train dan val
        for split in ["train", "val"]:
            img_dir = Path(self.dataset_cfg[f"{split}_images"])
            lbl_dir = Path(self.dataset_cfg[f"{split}_labels"])

            if not img_dir.exists():
                issues.append(f"Direktori gambar tidak ditemukan: {img_dir}")
                continue

            if not lbl_dir.exists():
                issues.append(f"Direktori label tidak ditemukan: {lbl_dir}")
                continue

            # Hitung gambar
            img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            stats[split]["images"] = len(img_files)

            # Hitung label dan kelas
            for img_file in img_files:
                lbl_file = lbl_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    stats[split]["labels"] += 1
                    with open(lbl_file, "r") as f:
                        for line in f.readlines():
                            parts = line.strip().split()
                            if parts:
                                cls_id = int(parts[0])
                                if cls_id in self.class_names:
                                    stats[split]["classes"][self.class_names[cls_id]] += 1
                                else:
                                    issues.append(f"Kelas tidak dikenal {cls_id} di {lbl_file}")
                else:
                    issues.append(f"Label hilang: {lbl_file}")

        # Tampilkan statistik
        for split in ["train", "val"]:
            print(f"\n[{split.upper()}]")
            print(f"  Gambar: {stats[split]['images']}")
            print(f"  Label: {stats[split]['labels']}")
            print(f"  Kelas: {dict(stats[split]['classes'])}")

        # Tampilkan masalah
        if issues:
            print(f"\n[PERINGATAN] Ditemukan {len(issues)} masalah:")
            for issue in issues[:10]:
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... dan {len(issues) - 10} lagi")
        else:
            print("\n[OK] Validasi dataset berhasil!")

        return stats, issues

    def split_dataset(self, source_dir, train_ratio=0.8, seed=42):
        """
        Split dataset menjadi train/val.
        
        Args:
            source_dir: direktori sumber (berisi images/ dan labels/)
            train_ratio: rasio data train (0-1)
            seed: seed untuk random shuffle
        """
        print("\n" + "=" * 60)
        print("SPLIT DATASET")
        print("=" * 60)

        source = Path(source_dir)
        images_dir = source / "images"
        labels_dir = source / "labels"

        if not images_dir.exists():
            print(f"[ERROR] Direktori gambar tidak ditemukan: {images_dir}")
            return

        # Ambil semua gambar
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        random.seed(seed)
        random.shuffle(image_files)

        # Split berdasarkan rasio
        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        print(f"Total gambar: {len(image_files)}")
        print(f"Train: {len(train_files)} ({train_ratio*100:.0f}%)")
        print(f"Val: {len(val_files)} ({(1-train_ratio)*100:.0f}%)")

        # Buat direktori
        for split in ["train", "val"]:
            (Path(self.dataset_cfg[f"{split}_images"])).mkdir(parents=True, exist_ok=True)
            (Path(self.dataset_cfg[f"{split}_labels"])).mkdir(parents=True, exist_ok=True)

        # Salin file
        for split_name, files in [("train", train_files), ("val", val_files)]:
            for img_file in files:
                # Salin gambar
                dst_img = Path(self.dataset_cfg[f"{split_name}_images"]) / img_file.name
                shutil.copy2(img_file, dst_img)

                # Salin label
                lbl_file = labels_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    dst_lbl = Path(self.dataset_cfg[f"{split_name}_labels"]) / lbl_file.name
                    shutil.copy2(lbl_file, dst_lbl)

            print(f"[OK] Set {split_name} berhasil disalin")

    def create_dataset_yaml(self, output_path="data/dataset.yaml"):
        """
        Membuat file konfigurasi dataset.yaml.
        
        Format YOLOv8:
        - path: path ke root dataset
        - train: path relatif ke gambar train
        - val: path relatif ke gambar val
        - nc: jumlah kelas
        - names: nama kelas
        
        Args:
            output_path: path output file yaml
        """
        dataset_yaml = {
            "path": str(Path(self.dataset_cfg["train_images"]).parent),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        with open(output_path, "w") as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)

        print(f"\n[OK] Dataset YAML dibuat di: {output_path}")
        return dataset_yaml

    def fix_missing_labels(self, image_dir, label_dir, default_class=0):
        """
        Membuat file label kosong untuk gambar yang belum memiliki label.
        
        Args:
            image_dir: direktori gambar
            label_dir: direktori label
            default_class: ID kelas default (tidak digunakan, label kosong)
        """
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        label_dir.mkdir(parents=True, exist_ok=True)

        images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        fixed = 0

        for img_file in images:
            lbl_file = label_dir / (img_file.stem + ".txt")
            if not lbl_file.exists():
                lbl_file.touch()
                fixed += 1

        print(f"[OK] Berhasil perbaiki {fixed} file label yang hilang")

    def convert_labelme_to_yolo(self, labelme_dir, output_dir, class_mapping):
        """
        Mengkonversi anotasi LabelMe (JSON) ke format YOLO.
        
        Format YOLO:
        class_id center_x center_y width height (normalized 0-1)
        
        Args:
            labelme_dir: direktori berisi file JSON LabelMe
            output_dir: direktori output untuk label YOLO
            class_mapping: dict pemetaan nama kelas ke ID
        """
        import json

        labelme_dir = Path(labelme_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        converted = 0
        for json_file in labelme_dir.glob("*.json"):
            with open(json_file, "r") as f:
                data = json.load(f)

            image_w = data["imageWidth"]
            image_h = data["imageHeight"]

            yolo_lines = []
            for shape in data["shapes"]:
                label = shape["label"]
                if label not in class_mapping:
                    continue

                cls_id = class_mapping[label]
                points = shape["points"]

                # Konversi ke format YOLO (normalized center x, center y, width, height)
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                x_center = (x_min + x_max) / 2 / image_w
                y_center = (y_min + y_max) / 2 / image_h
                width = (x_max - x_min) / image_w
                height = (y_max - y_min) / image_h

                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Simpan label YOLO
            output_file = output_dir / (json_file.stem + ".txt")
            with open(output_file, "w") as f:
                f.write("\n".join(yolo_lines))

            converted += 1

        print(f"[OK] Berhasil konversi {converted} anotasi LabelMe ke format YOLO")


def main():
    """Fungsi utama untuk menjalankan script persiapan dataset dari command line."""
    parser = argparse.ArgumentParser(description="Alat Persiapan Dataset")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path file konfigurasi")
    parser.add_argument("--action", type=str, required=True,
                       choices=["validate", "split", "yaml", "fix-labels", "convert-labelme"],
                       help="Aksi yang akan dilakukan")
    parser.add_argument("--source", type=str,
                       help="Direktori sumber")
    parser.add_argument("--output", type=str,
                       help="Direktori output")
    parser.add_argument("--ratio", type=float, default=0.8,
                       help="Rasio split train")
    args = parser.parse_args()

    config = load_config(args.config)
    preparator = DatasetPreparator(config)

    # Jalankan aksi sesuai pilihan
    if args.action == "validate":
        preparator.validate_dataset()
    elif args.action == "split":
        preparator.split_dataset(args.source, train_ratio=args.ratio)
    elif args.action == "yaml":
        preparator.create_dataset_yaml(args.output)
    elif args.action == "fix-labels":
        preparator.fix_missing_labels(args.source, args.output)
    elif args.action == "convert-labelme":
        class_mapping = {name: idx for idx, name in config["dataset"]["names"].items()}
        preparator.convert_labelme_to_yolo(args.source, args.output, class_mapping)


if __name__ == "__main__":
    main()
