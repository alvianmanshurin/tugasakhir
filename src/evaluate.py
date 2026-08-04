"""Evaluation script for Vehicle Detection Model"""

import os
import time
import yaml
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report


def load_config(config_path="config/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ModelEvaluator:
    """Evaluate YOLOv8 model for vehicle detection."""

    def __init__(self, config):
        self.config = config
        self.eval_cfg = config["evaluation"]
        self.model_cfg = config["model"]
        self.dataset_cfg = config["dataset"]
        self.output_dir = Path(self.eval_cfg["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        model_path = "models/vehicle_detection/weights/best.pt"
        if not os.path.exists(model_path):
            print("[ERROR] Model not found. Please train the model first.")
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = config["dataset"]["names"]

    def evaluate_map(self):
        """Evaluate mAP, precision, and recall on validation set."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION - mAP, PRECISION, RECALL")
        print("=" * 60)

        results = self.model.val(data=self.dataset_cfg["yaml_path"])

        metrics = {
            "mAP50": float(results.box.map50),
            "mAP50-95": float(results.box.map),
            "Precision": float(results.box.mp),
            "Recall": float(results.box.mr),
        }

        # Per-class metrics
        per_class = {}
        for i, name in self.class_names.items():
            if i < len(results.box.ap_class_index):
                per_class[name] = {
                    "AP50": float(results.box.ap50[i]) if i < len(results.box.ap50) else 0,
                    "AP50-95": float(results.box.ap[i]) if i < len(results.box.ap) else 0,
                }

        print(f"\n[RESULTS] Overall Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        print(f"\n[RESULTS] Per-Class AP:")
        for name, ap in per_class.items():
            print(f"  {name}: AP50={ap['AP50']:.4f}, AP50-95={ap['AP50-95']:.4f}")

        return metrics, per_class, results

    def evaluate_fps(self, image_size=640, num_images=100):
        """Evaluate inference speed (FPS)."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION - FPS (Inference Speed)")
        print("=" * 60)

        # Create dummy image
        dummy_img = np.random.randint(
            0, 255, (image_size, image_size, 3), dtype=np.uint8
        )

        # Warmup
        for _ in range(10):
            self.model(dummy_img, verbose=False)

        # Measure FPS
        times = []
        for _ in range(num_images):
            start = time.time()
            self.model(dummy_img, verbose=False)
            times.append(time.time() - start)

        avg_time = np.mean(times)
        fps = 1.0 / avg_time

        metrics = {
            "avg_inference_time_ms": float(avg_time * 1000),
            "fps": float(fps),
            "num_images": num_images,
            "image_size": image_size,
        }

        print(f"\n[RESULTS] FPS Evaluation:")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  Number of images: {num_images}")
        print(f"  Average inference time: {avg_time*1000:.2f} ms")
        print(f"  FPS: {fps:.2f}")

        return metrics

    def generate_confusion_matrix(self, results):
        """Generate and save confusion matrix."""
        print("\n[INFO] Generating confusion matrix...")

        # Get predictions and ground truth
        true_labels = []
        pred_labels = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    pred_labels.append(int(box.cls[0]))
            if result.probs is not None:
                true_labels.append(int(result.probs.top1))

        if len(true_labels) > 0 and len(pred_labels) > 0:
            cm = confusion_matrix(
                true_labels, pred_labels, labels=list(self.class_names.keys())
            )

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=list(self.class_names.values()),
                yticklabels=list(self.class_names.values()),
            )
            plt.title("Confusion Matrix - Vehicle Detection")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()

            save_path = self.output_dir / "confusion_matrix.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[INFO] Confusion matrix saved at: {save_path}")

    def generate_pr_curve(self, results):
        """Generate and save Precision-Recall curve."""
        print("\n[INFO] Generating PR curve...")

        try:
            # Plot PR curve from results
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # Note: PR curve data is available via results.curves
            # For simplicity, we plot a basic placeholder
            pass

            plt.title("Precision-Recall Curve - Vehicle Detection")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.grid(True)
            plt.tight_layout()

            save_path = self.output_dir / "pr_curve.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[INFO] PR curve saved at: {save_path}")
        except Exception as e:
            print(f"[WARNING] Could not generate PR curve: {e}")

    def generate_f1_curve(self, results):
        """Generate and save F1 curve."""
        print("\n[INFO] Generating F1 curve...")

        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            plt.title("F1-Confidence Curve - Vehicle Detection")
            plt.xlabel("Confidence Threshold")
            plt.ylabel("F1 Score")
            plt.grid(True)
            plt.tight_layout()

            save_path = self.output_dir / "f1_curve.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[INFO] F1 curve saved at: {save_path}")
        except Exception as e:
            print(f"[WARNING] Could not generate F1 curve: {e}")

    def save_evaluation_report(self, map_metrics, per_class, fps_metrics):
        """Save evaluation report to JSON."""
        report = {
            "model": self.model_cfg["architecture"],
            "dataset": self.dataset_cfg["yaml_path"],
            "overall_metrics": map_metrics,
            "per_class_metrics": per_class,
            "fps_metrics": fps_metrics,
        }

        save_path = self.output_dir / "evaluation_report.json"
        with open(save_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"\n[INFO] Evaluation report saved at: {save_path}")

        # Also save as CSV for easy comparison
        df = pd.DataFrame(
            {
                "Metric": list(map_metrics.keys()) + ["FPS"],
                "Value": list(map_metrics.values()) + [fps_metrics["fps"]],
            }
        )
        csv_path = self.output_dir / "evaluation_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Metrics CSV saved at: {csv_path}")

    def full_evaluation(self):
        """Run complete evaluation pipeline."""
        print("\n" + "=" * 60)
        print("FULL MODEL EVALUATION")
        print("=" * 60)

        # 1. mAP, Precision, Recall
        map_metrics, per_class, results = self.evaluate_map()

        # 2. FPS
        fps_metrics = self.evaluate_fps(
            image_size=self.model_cfg["input_size"]
        )

        # 3. Confusion Matrix
        if self.eval_cfg["confusion_matrix"]:
            self.generate_confusion_matrix(results)

        # 4. Save Report
        self.save_evaluation_report(map_metrics, per_class, fps_metrics)

        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)

        return map_metrics, per_class, fps_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Vehicle Detection Model"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Config file path"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model path"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["map", "fps", "confusion", "all"],
        help="Evaluation task",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model:
        config["model"]["architecture"] = args.model

    evaluator = ModelEvaluator(config)

    if args.task == "map":
        evaluator.evaluate_map()
    elif args.task == "fps":
        evaluator.evaluate_fps()
    elif args.task == "confusion":
        map_metrics, per_class, results = evaluator.evaluate_map()
        evaluator.generate_confusion_matrix(results)
    else:
        evaluator.full_evaluation()


if __name__ == "__main__":
    main()
