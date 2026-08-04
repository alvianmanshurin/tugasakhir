"""Batch Processing Script for Vehicle Detection"""

import os
import cv2
import yaml
import argparse
import json
import time
import glob
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor


def load_config(config_path="config/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class BatchProcessor:
    """Batch process multiple images or videos for vehicle detection."""

    def __init__(self, config):
        self.config = config
        self.model_cfg = config["model"]
        self.det_cfg = config["detection"]
        self.class_names = config["dataset"]["names"]

        # Load model
        model_path = "models/vehicle_detection/weights/best.pt"
        if not os.path.exists(model_path):
            model_path = self.model_cfg["architecture"]
            print(f"[WARNING] Custom model not found. Using pretrained: {model_path}")

        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)

    def process_single_image(self, image_path, output_dir=None):
        """Process a single image and return results."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        start_time = time.time()

        results = self.model(
            img,
            conf=self.model_cfg["confidence_threshold"],
            iou=self.model_cfg["iou_threshold"],
            verbose=False,
        )

        inference_time = time.time() - start_time

        # Process results
        detections = []
        counts = {name: 0 for name in self.class_names.values()}

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()

                    if cls_id < len(self.class_names):
                        cls_name = self.class_names[cls_id]
                        detections.append({
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": xyxy,
                        })
                        counts[cls_name] += 1

        total_count = sum(counts.values())

        # Save detection result if output_dir specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Draw detections
            img_result = img.copy()
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

            for det in detections:
                x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
                color = colors[det["class_id"] % len(colors)]
                cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(img_result, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Add count summary
            y_offset = 30
            for cls_name, count in counts.items():
                cv2.putText(img_result, f"{cls_name}: {count}",
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)
                y_offset += 25
            cv2.putText(img_result, f"Total: {total_count}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 255), 2)

            # Save
            output_path = output_dir / f"{Path(image_path).stem}_detected.jpg"
            cv2.imwrite(str(output_path), img_result)

        return {
            "image": str(image_path),
            "detections": detections,
            "counts": counts,
            "total": total_count,
            "inference_time": inference_time,
        }

    def batch_process_images(self, input_dir, output_dir=None, max_workers=4):
        """Process all images in a directory."""
        input_dir = Path(input_dir)

        # Get all images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
            image_files.extend(input_dir.glob(ext))

        if not image_files:
            print(f"[WARNING] No images found in: {input_dir}")
            return []

        print(f"\n[INFO] Found {len(image_files)} images to process")
        print(f"[INFO] Using {max_workers} workers")

        results = []
        start_time = time.time()

        # Process images
        for idx, img_file in enumerate(sorted(image_files), 1):
            result = self.process_single_image(img_file, output_dir)
            if result:
                results.append(result)

            # Progress update
            if idx % 10 == 0 or idx == len(image_files):
                elapsed = time.time() - start_time
                fps = idx / elapsed if elapsed > 0 else 0
                print(f"  Processed {idx}/{len(image_files)} | "
                      f"FPS: {fps:.1f} | "
                      f"Elapsed: {elapsed:.1f}s")

        # Summary
        total_time = time.time() - start_time
        total_vehicles = sum(r["total"] for r in results)
        avg_fps = len(results) / total_time if total_time > 0 else 0

        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"  Total images: {len(results)}")
        print(f"  Total vehicles detected: {total_vehicles}")
        print(f"  Average vehicles per image: {total_vehicles/len(results):.1f}")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Average FPS: {avg_fps:.1f}")

        # Class distribution
        class_totals = {name: 0 for name in self.class_names.values()}
        for r in results:
            for cls_name, count in r["counts"].items():
                class_totals[cls_name] += count

        print(f"\n  Class distribution:")
        for cls_name, count in class_totals.items():
            if count > 0:
                pct = count / total_vehicles * 100 if total_vehicles > 0 else 0
                print(f"    {cls_name}: {count} ({pct:.1f}%)")

        # Save results to CSV
        if output_dir:
            self._save_results_csv(results, output_dir)

        return results

    def _save_results_csv(self, results, output_dir):
        """Save batch processing results to CSV."""
        output_dir = Path(output_dir)
        rows = []

        for r in results:
            row = {
                "image": Path(r["image"]).name,
                "total": r["total"],
                "inference_time_ms": r["inference_time"] * 1000,
            }
            for cls_name in self.class_names.values():
                row[cls_name] = r["counts"].get(cls_name, 0)
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = output_dir / "batch_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[INFO] Results saved at: {csv_path}")

    def generate_report(self, results, output_path):
        """Generate detailed processing report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_cfg["architecture"],
            "confidence_threshold": self.model_cfg["confidence_threshold"],
            "total_images": len(results),
            "total_vehicles": sum(r["total"] for r in results),
            "avg_vehicles_per_image": np.mean([r["total"] for r in results]),
            "avg_inference_time_ms": np.mean([r["inference_time"] for r in results]) * 1000,
            "avg_fps": len(results) / sum(r["inference_time"] for r in results),
            "class_totals": {},
        }

        for cls_name in self.class_names.values():
            report["class_totals"][cls_name] = sum(
                r["counts"].get(cls_name, 0) for r in results
            )

        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"[INFO] Report saved at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch Processing for Vehicle Detection")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, default="outputs/batch", help="Output directory")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model:
        config["model"]["architecture"] = args.model
    if args.conf:
        config["model"]["confidence_threshold"] = args.conf

    processor = BatchProcessor(config)
    results = processor.batch_process_images(args.input, args.output, args.workers)

    if results:
        processor.generate_report(results, Path(args.output) / "report.json")


if __name__ == "__main__":
    main()
