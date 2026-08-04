"""Comparison Script: YOLO vs Manual Counting"""

import os
import cv2
import yaml
import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path="config/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ComparisonEvaluator:
    """Compare YOLO automated counting vs manual counting."""

    def __init__(self, config):
        self.config = config
        self.model_cfg = config["model"]
        self.class_names = config["dataset"]["names"]

        # Load model
        model_path = "models/vehicle_detection/weights/best.pt"
        if not os.path.exists(model_path):
            model_path = self.model_cfg["architecture"]
            print(f"[WARNING] Custom model not found. Using pretrained: {model_path}")

        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)

    def load_manual_counts(self, json_path):
        """Load manual counting results from JSON file."""
        with open(json_path, "r") as f:
            return json.load(f)

    def create_manual_count_template(self, image_dir, output_path):
        """Create a template JSON file for manual counting."""
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

        template = {}
        for img_file in sorted(image_files):
            template[img_file.name] = {
                "motor": 0,
                "mobil": 0,
                "bus": 0,
                "truk": 0,
                "total": 0,
                "notes": ""
            }

        with open(output_path, "w") as f:
            json.dump(template, f, indent=4)

        print(f"[OK] Manual count template created at: {output_path}")
        print(f"[INFO] Fill in the counts for each image, then run comparison")
        return template

    def yolo_count_image(self, image_path):
        """Count vehicles in a single image using YOLO."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        results = self.model(
            img,
            conf=self.model_cfg["confidence_threshold"],
            iou=self.model_cfg["iou_threshold"],
            verbose=False,
        )

        counts = {name: 0 for name in self.class_names.values()}

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in self.class_names:
                        counts[self.class_names[cls_id]] += 1

        counts["total"] = sum(counts.values())
        return counts

    def compare_counts(self, yolo_counts, manual_counts):
        """Compare YOLO counts with manual counts."""
        result = {}

        for cls_name in self.class_names.values():
            yolo_count = yolo_counts.get(cls_name, 0)
            manual_count = manual_counts.get(cls_name, 0)

            diff = yolo_count - manual_count
            accuracy = (
                min(yolo_count, manual_count) / max(yolo_count, manual_count) * 100
                if max(yolo_count, manual_count) > 0 else 100
            )

            result[cls_name] = {
                "yolo": yolo_count,
                "manual": manual_count,
                "difference": diff,
                "accuracy": accuracy,
            }

        # Total comparison
        yolo_total = yolo_counts.get("total", 0)
        manual_total = manual_counts.get("total", 0)
        result["total"] = {
            "yolo": yolo_total,
            "manual": manual_total,
            "difference": yolo_total - manual_total,
            "accuracy": (
                min(yolo_total, manual_total) / max(yolo_total, manual_total) * 100
                if max(yolo_total, manual_total) > 0 else 100
            ),
        }

        return result

    def evaluate_dataset(self, image_dir, manual_counts_json):
        """Evaluate YOLO counting on entire dataset against manual counts."""
        print("\n" + "=" * 60)
        print("YOLO vs MANUAL COUNTING COMPARISON")
        print("=" * 60)

        image_dir = Path(image_dir)
        manual_counts = self.load_manual_counts(manual_counts_json)

        all_results = []
        total_yolo = {name: 0 for name in self.class_names.values()}
        total_manual = {name: 0 for name in self.class_names.values()}
        total_yolo["total"] = 0
        total_manual["total"] = 0

        for img_name, manual in sorted(manual_counts.items()):
            img_path = image_dir / img_name
            if not img_path.exists():
                print(f"[WARNING] Image not found: {img_path}")
                continue

            yolo_counts = self.yolo_count_image(img_path)
            if yolo_counts is None:
                continue

            comparison = self.compare_counts(yolo_counts, manual)
            comparison["image"] = img_name
            all_results.append(comparison)

            # Accumulate totals
            for cls_name in self.class_names.values():
                total_yolo[cls_name] += yolo_counts.get(cls_name, 0)
                total_manual[cls_name] += manual.get(cls_name, 0)
            total_yolo["total"] += yolo_counts.get("total", 0)
            total_manual["total"] += manual.get("total", 0)

            # Print per-image result
            print(f"\n{img_name}:")
            print(f"  {'Class':<10} {'YOLO':>6} {'Manual':>6} {'Diff':>6} {'Acc':>8}")
            print(f"  {'-'*40}")
            for cls_name in self.class_names.values():
                r = comparison[cls_name]
                print(f"  {cls_name:<10} {r['yolo']:>6} {r['manual']:>6} "
                      f"{r['difference']:>+6} {r['accuracy']:>7.1f}%")
            print(f"  {'TOTAL':<10} {comparison['total']['yolo']:>6} "
                  f"{comparison['total']['manual']:>6} "
                  f"{comparison['total']['difference']:>+6} "
                  f"{comparison['total']['accuracy']:>7.1f}%")

        # Overall summary
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)

        print(f"\nTotal images evaluated: {len(all_results)}")
        print(f"\n{'Class':<10} {'YOLO Total':>12} {'Manual Total':>14} {'Accuracy':>10}")
        print(f"{'-'*50}")

        for cls_name in self.class_names.values():
            yolo_total = total_yolo[cls_name]
            manual_total = total_manual[cls_name]
            accuracy = (
                min(yolo_total, manual_total) / max(yolo_total, manual_total) * 100
                if max(yolo_total, manual_total) > 0 else 100
            )
            print(f"{cls_name:<10} {yolo_total:>12} {manual_total:>14} {accuracy:>9.1f}%")

        yolo_grand = total_yolo["total"]
        manual_grand = total_manual["total"]
        overall_acc = (
            min(yolo_grand, manual_grand) / max(yolo_grand, manual_grand) * 100
            if max(yolo_grand, manual_grand) > 0 else 100
        )
        print(f"{'TOTAL':<10} {yolo_grand:>12} {manual_grand:>14} {overall_acc:>9.1f}%")

        # Calculate MAE and RMSE
        yolo_totals = [r["total"]["yolo"] for r in all_results]
        manual_totals = [r["total"]["manual"] for r in all_results]

        mae = np.mean(np.abs(np.array(yolo_totals) - np.array(manual_totals)))
        rmse = np.sqrt(np.mean((np.array(yolo_totals) - np.array(manual_totals))**2))

        print(f"\nMean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Square Error (RMSE): {rmse:.2f}")

        # Save results
        results_df = []
        for r in all_results:
            for cls_name in self.class_names.values():
                results_df.append({
                    "Image": r["image"],
                    "Class": cls_name,
                    "YOLO": r[cls_name]["yolo"],
                    "Manual": r[cls_name]["manual"],
                    "Difference": r[cls_name]["difference"],
                    "Accuracy": r[cls_name]["accuracy"],
                })

        df = pd.DataFrame(results_df)
        output_dir = Path(self.config["evaluation"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "comparison_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[INFO] Detailed results saved at: {csv_path}")

        return {
            "per_image": all_results,
            "totals": {"yolo": total_yolo, "manual": total_manual},
            "overall_accuracy": overall_acc,
            "mae": mae,
            "rmse": rmse,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLO vs Manual Vehicle Counting"
    )
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--action", type=str, required=True,
                       choices=["template", "compare"],
                       help="Action to perform")
    parser.add_argument("--image-dir", type=str, help="Image directory")
    parser.add_argument("--manual-json", type=str, help="Manual counts JSON file")
    parser.add_argument("--output", type=str, help="Output path for template")
    args = parser.parse_args()

    config = load_config(args.config)
    evaluator = ComparisonEvaluator(config)

    if args.action == "template":
        evaluator.create_manual_count_template(args.image_dir, args.output)
    elif args.action == "compare":
        evaluator.evaluate_dataset(args.image_dir, args.manual_json)


if __name__ == "__main__":
    main()
