"""Model Export Script for Vehicle Detection"""

import os
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path="config/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ModelExporter:
    """Export YOLOv8 model to various formats."""

    def __init__(self, config):
        self.config = config
        self.model_cfg = config["model"]

    def export_model(self, model_path=None, formats=None):
        """
        Export model to specified formats.

        Available formats:
        - torchscript: TorchScript
        - onnx: ONNX
        - openvino: OpenVINO
        - engine: TensorRT
        - coreml: CoreML
        - saved_model: TensorFlow SavedModel
        - pb: TensorFlow GraphDef
        - tflite: TensorFlow Lite
        - edgetpu: TensorFlow Lite Edge TPU
        - tfjs: TensorFlow.js
        - paddle: PaddlePaddle
        """
        if model_path is None:
            model_path = "models/vehicle_detection/weights/best.pt"

        if not os.path.exists(model_path):
            print(f"[ERROR] Model not found: {model_path}")
            return

        if formats is None:
            formats = ["onnx", "tflite"]

        print("=" * 60)
        print("MODEL EXPORT")
        print("=" * 60)
        print(f"\n[INFO] Source model: {model_path}")
        print(f"[INFO] Export formats: {formats}")

        model = YOLO(model_path)

        for fmt in formats:
            try:
                print(f"\n[INFO] Exporting to {fmt}...")
                exported_path = model.export(format=fmt, imgsz=640)
                print(f"[OK] Exported to {fmt}: {exported_path}")
            except Exception as e:
                print(f"[ERROR] Failed to export to {fmt}: {e}")

    def export_onnx(self, model_path=None, opset=12):
        """Export model to ONNX format."""
        if model_path is None:
            model_path = "models/vehicle_detection/weights/best.pt"

        print(f"\n[INFO] Exporting to ONNX (opset={opset})...")
        model = YOLO(model_path)
        exported_path = model.export(format="onnx", imgsz=640, opset=opset)
        print(f"[OK] ONNX model saved at: {exported_path}")
        return exported_path

    def export_tflite(self, model_path=None):
        """Export model to TFLite format."""
        if model_path is None:
            model_path = "models/vehicle_detection/weights/best.pt"

        print(f"\n[INFO] Exporting to TFLite...")
        model = YOLO(model_path)
        exported_path = model.export(format="tflite", imgsz=640)
        print(f"[OK] TFLite model saved at: {exported_path}")
        return exported_path

    def export_tensorrt(self, model_path=None, half=False):
        """Export model to TensorRT format."""
        if model_path is None:
            model_path = "models/vehicle_detection/weights/best.pt"

        print(f"\n[INFO] Exporting to TensorRT (half={half})...")
        model = YOLO(model_path)
        exported_path = model.export(format="engine", imgsz=640, half=half)
        print(f"[OK] TensorRT model saved at: {exported_path}")
        return exported_path

    def benchmark_model(self, model_path=None, imgsz=640):
        """Benchmark model in different formats."""
        if model_path is None:
            model_path = "models/vehicle_detection/weights/best.pt"

        print("\n" + "=" * 60)
        print("MODEL BENCHMARK")
        print("=" * 60)

        model = YOLO(model_path)

        # Benchmark PyTorch
        print("\n[INFO] Benchmarking PyTorch model...")
        results = model.val(data=self.config["dataset"]["yaml_path"], imgsz=imgsz)

        print(f"\n[RESULTS] PyTorch Benchmark:")
        print(f"  mAP50: {results.box.map50:.4f}")
        print(f"  mAP50-95: {results.box.map:.4f}")
        print(f"  Inference time: {results.speed['inference']:.1f}ms")
        print(f"  FPS: {1000/results.speed['inference']:.1f}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Export Vehicle Detection Model")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--format", type=str, nargs="+",
                       default=["onnx", "tflite"],
                       help="Export formats")
    parser.add_argument("--action", type=str, default="export",
                       choices=["export", "onnx", "tflite", "tensorrt", "benchmark"],
                       help="Export action")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--half", action="store_true", help="Export with half precision")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model:
        config["model"]["architecture"] = args.model

    exporter = ModelExporter(config)

    if args.action == "export":
        exporter.export_model(args.model, args.format)
    elif args.action == "onnx":
        exporter.export_onnx(args.model, args.opset)
    elif args.action == "tflite":
        exporter.export_tflite(args.model)
    elif args.action == "tensorrt":
        exporter.export_tensorrt(args.model, args.half)
    elif args.action == "benchmark":
        exporter.benchmark_model(args.model)


if __name__ == "__main__":
    main()
