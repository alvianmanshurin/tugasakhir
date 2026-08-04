"""Dataset Preparation Script for Vehicle Detection"""

import os
import shutil
import yaml
import argparse
import random
from pathlib import Path
from collections import Counter


def load_config(config_path="config/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class DatasetPreparator:
    """Prepare and validate dataset for YOLOv8 training."""

    def __init__(self, config):
        self.config = config
        self.dataset_cfg = config["dataset"]
        self.class_names = config["dataset"]["names"]

    def validate_dataset(self):
        """Validate dataset structure and labels."""
        print("=" * 60)
        print("DATASET VALIDATION")
        print("=" * 60)

        issues = []
        stats = {"train": {"images": 0, "labels": 0, "classes": Counter()},
                 "val": {"images": 0, "labels": 0, "classes": Counter()}}

        for split in ["train", "val"]:
            img_dir = Path(self.dataset_cfg[f"{split}_images"])
            lbl_dir = Path(self.dataset_cfg[f"{split}_labels"])

            if not img_dir.exists():
                issues.append(f"Image directory not found: {img_dir}")
                continue

            if not lbl_dir.exists():
                issues.append(f"Label directory not found: {lbl_dir}")
                continue

            # Count images
            img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            stats[split]["images"] = len(img_files)

            # Count labels and classes
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
                                    issues.append(f"Unknown class {cls_id} in {lbl_file}")
                else:
                    issues.append(f"Missing label: {lbl_file}")

        # Print stats
        for split in ["train", "val"]:
            print(f"\n[{split.upper()}]")
            print(f"  Images: {stats[split]['images']}")
            print(f"  Labels: {stats[split]['labels']}")
            print(f"  Classes: {dict(stats[split]['classes'])}")

        # Print issues
        if issues:
            print(f"\n[WARNINGS] Found {len(issues)} issues:")
            for issue in issues[:10]:
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("\n[OK] Dataset validation passed!")

        return stats, issues

    def split_dataset(self, source_dir, train_ratio=0.8, seed=42):
        """Split dataset into train/val sets."""
        print("\n" + "=" * 60)
        print("SPLITTING DATASET")
        print("=" * 60)

        source = Path(source_dir)
        images_dir = source / "images"
        labels_dir = source / "labels"

        if not images_dir.exists():
            print(f"[ERROR] Images directory not found: {images_dir}")
            return

        # Get all images
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        random.seed(seed)
        random.shuffle(image_files)

        split_idx = int(len(image_files) * train_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        print(f"Total images: {len(image_files)}")
        print(f"Train: {len(train_files)} ({train_ratio*100:.0f}%)")
        print(f"Val: {len(val_files)} ({(1-train_ratio)*100:.0f}%)")

        # Create directories
        for split in ["train", "val"]:
            (Path(self.dataset_cfg[f"{split}_images"])).mkdir(parents=True, exist_ok=True)
            (Path(self.dataset_cfg[f"{split}_labels"])).mkdir(parents=True, exist_ok=True)

        # Copy files
        for split_name, files in [("train", train_files), ("val", val_files)]:
            for img_file in files:
                # Copy image
                dst_img = Path(self.dataset_cfg[f"{split_name}_images"]) / img_file.name
                shutil.copy2(img_file, dst_img)

                # Copy label
                lbl_file = labels_dir / (img_file.stem + ".txt")
                if lbl_file.exists():
                    dst_lbl = Path(self.dataset_cfg[f"{split_name}_labels"]) / lbl_file.name
                    shutil.copy2(lbl_file, dst_lbl)

            print(f"[OK] {split_name} set copied")

    def create_dataset_yaml(self, output_path="data/dataset.yaml"):
        """Create dataset YAML configuration file."""
        dataset_yaml = {
            "path": str(Path(self.dataset_cfg["train_images"]).parent),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        with open(output_path, "w") as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)

        print(f"\n[OK] Dataset YAML created at: {output_path}")
        return dataset_yaml

    def fix_missing_labels(self, image_dir, label_dir, default_class=0):
        """Create empty label files for images without labels."""
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

        print(f"[OK] Fixed {fixed} missing label files")

    def convert_labelme_to_yolo(self, labelme_dir, output_dir, class_mapping):
        """
        Convert LabelMe JSON annotations to YOLO format.

        Args:
            labelme_dir: Directory with LabelMe JSON files
            output_dir: Output directory for YOLO labels
            class_mapping: Dict mapping class names to IDs
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

                # Convert to YOLO format (normalized center x, center y, width, height)
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]

                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                x_center = (x_min + x_max) / 2 / image_w
                y_center = (y_min + y_max) / 2 / image_h
                width = (x_max - x_min) / image_w
                height = (y_max - y_min) / image_h

                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Save YOLO label
            output_file = output_dir / (json_file.stem + ".txt")
            with open(output_file, "w") as f:
                f.write("\n".join(yolo_lines))

            converted += 1

        print(f"[OK] Converted {converted} LabelMe annotations to YOLO format")


def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation Tool")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--action", type=str, required=True,
                       choices=["validate", "split", "yaml", "fix-labels", "convert-labelme"],
                       help="Action to perform")
    parser.add_argument("--source", type=str, help="Source directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--ratio", type=float, default=0.8, help="Train split ratio")
    args = parser.parse_args()

    config = load_config(args.config)
    preparator = DatasetPreparator(config)

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
