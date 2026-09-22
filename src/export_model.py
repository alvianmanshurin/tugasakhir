"""
Script untuk export model ke berbagai format
Mendukung: ONNX, TFLite, CoreML, TorchScript, dll
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))


def load_model(model_path):
    """Muat model YOLOv8"""
    from ultralytics import YOLO
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        return None
    
    print(f"[INFO] Memuat model: {model_path}")
    model = YOLO(model_path)
    return model


def export_to_onnx(model_path, output_dir="exports", imgsz=640, dynamic=True):
    """
    Export model ke ONNX format
    
    Args:
        model_path: path ke model .pt
        output_dir: direktori output
        imgsz: ukuran gambar input
        dynamic: batch size dinamis
    """
    model = load_model(model_path)
    if model is None:
        return False
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Export ke ONNX (imgsz={imgsz}, dynamic={dynamic})...")
        model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=dynamic,
            simplify=True,
        )
        
        print(f"[OK] Export ONNX berhasil")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal export ONNX: {e}")
        return False


def export_to_tflite(model_path, output_dir="exports", imgsz=640):
    """
    Export model ke TFLite format (untuk mobile)
    
    Args:
        model_path: path ke model .pt
        output_dir: direktori output
        imgsz: ukuran gambar input
    """
    model = load_model(model_path)
    if model is None:
        return False
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Export ke TFLite (imgsz={imgsz})...")
        model.export(
            format="tflite",
            imgsz=imgsz,
        )
        
        print(f"[OK] Export TFLite berhasil")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal export TFLite: {e}")
        return False


def export_to_torchscript(model_path, output_dir="exports", imgsz=640):
    """
    Export model ke TorchScript format
    
    Args:
        model_path: path ke model .pt
        output_dir: direktori output
        imgsz: ukuran gambar input
    """
    model = load_model(model_path)
    if model is None:
        return False
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Export ke TorchScript (imgsz={imgsz})...")
        model.export(
            format="torchscript",
            imgsz=imgsz,
        )
        
        print(f"[OK] Export TorchScript berhasil")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal export TorchScript: {e}")
        return False


def export_to_coreml(model_path, output_dir="exports", imgsz=640):
    """
    Export model ke CoreML format (untuk iOS/macOS)
    
    Args:
        model_path: path ke model .pt
        output_dir: direktori output
        imgsz: ukuran gambar input
    """
    model = load_model(model_path)
    if model is None:
        return False
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Export ke CoreML (imgsz={imgsz})...")
        model.export(
            format="coreml",
            imgsz=imgsz,
        )
        
        print(f"[OK] Export CoreML berhasil")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal export CoreML: {e}")
        return False


def export_to_openvino(model_path, output_dir="exports", imgsz=640):
    """
    Export model ke OpenVINO format (untuk Intel hardware)
    
    Args:
        model_path: path ke model .pt
        output_dir: direktori output
        imgsz: ukuran gambar input
    """
    model = load_model(model_path)
    if model is None:
        return False
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"[INFO] Export ke OpenVINO (imgsz={imgsz})...")
        model.export(
            format="openvino",
            imgsz=imgsz,
        )
        
        print(f"[OK] Export OpenVINO berhasil")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal export OpenVINO: {e}")
        return False


def export_all_formats(model_path, output_dir="exports", imgsz=640):
    """
    Export model ke semua format yang didukung
    
    Args:
        model_path: path ke model .pt
        output_dir: direktori output
        imgsz: ukuran gambar input
    """
    print("\n" + "=" * 60)
    print("EXPORT MODEL KE SEMUA FORMAT")
    print("=" * 60)
    
    formats = [
        ("ONNX", export_to_onnx),
        ("TorchScript", export_to_torchscript),
        ("OpenVINO", export_to_openvino),
    ]
    
    results = {}
    for name, func in formats:
        print(f"\n--- Export {name} ---")
        success = func(model_path, output_dir, imgsz)
        results[name] = success
    
    # Ringkasan
    print("\n" + "=" * 60)
    print("RINGKASAN EXPORT")
    print("=" * 60)
    for name, success in results.items():
        status = "BERHASIL" if success else "GAGAL"
        print(f"  {name}: {status}")
    
    return results


def main():
    """Fungsi utama untuk menjalankan export model"""
    parser = argparse.ArgumentParser(
        description="Export Model YOLOv8 ke Berbagai Format"
    )
    parser.add_argument(
        "--model", type=str,
        default="models/yolov8n_vehicle/weights/best.pt",
        help="Path ke model .pt (default: models/yolov8n_vehicle/weights/best.pt)"
    )
    parser.add_argument(
        "--format", type=str,
        choices=["onnx", "tflite", "torchscript", "coreml", "openvino", "all"],
        default="all",
        help="Format export (default: all)"
    )
    parser.add_argument(
        "--output", type=str, default="exports",
        help="Direktori output (default: exports)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Ukuran gambar input (default: 640)"
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("EXPORT MODEL YOLOv8")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Format: {args.format}")
    print(f"Output: {args.output}")
    print(f"Image size: {args.imgsz}")
    print("=" * 60)
    
    # Jalankan export berdasarkan format
    if args.format == "all":
        export_all_formats(args.model, args.output, args.imgsz)
    elif args.format == "onnx":
        export_to_onnx(args.model, args.output, args.imgsz)
    elif args.format == "tflite":
        export_to_tflite(args.model, args.output, args.imgsz)
    elif args.format == "torchscript":
        export_to_torchscript(args.model, args.output, args.imgsz)
    elif args.format == "coreml":
        export_to_coreml(args.model, args.output, args.imgsz)
    elif args.format == "openvino":
        export_to_openvino(args.model, args.output, args.imgsz)


if __name__ == "__main__":
    main()
