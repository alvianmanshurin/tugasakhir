"""
Export model ke berbagai format: ONNX, TorchScript, OpenVINO, TFLite, CoreML.

Dua masalah versi lama:

* ``--output`` sama sekali tidak dipakai. Ultralytics ``model.export()``
  selalu menulis artefak di samping file ``.pt``, jadi hasil export mendarat
  di ``runs/detect/models/vehicle_detection/weights/`` walaupun pengguna
  sudah minta folder lain. Sekarang artefak dipindahkan ke folder tujuan.
* Folder output relatif ke current working directory, sehingga diekspor ke
  folder yang berbeda tiap kali skrip dijalankan dari direktori lain.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "exports"


def load_model(model_path):
    """Muat model YOLOv8"""
    from ultralytics import YOLO

    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        return None

    print(f"[INFO] Memuat model: {model_path}")
    return YOLO(model_path)


def _snapshot_weights(model_path: Path) -> dict:
    """
    Catat (nama -> (mtime, ukuran)) isi folder weights SEBELUR export.

    Deteksi harus berbasis perubahan state, bukan nama: export kedua kali ke
    format yang sama menimpa file yang sudah ada, jadi tidak ada "file baru" -
    padahal artefaknya baru saja dibuat ulang. Ukuran ikut disimpan karena
    filesystem Windows kadang punya resolusi mtime 1 detik, jadi file yang
    ditulis ulang dalam detik yang sama bisa punya mtime identik.
    """
    try:
        return {p.name: (p.stat().st_mtime, p.stat().st_size)
                for p in model_path.parent.iterdir()}
    except OSError:
        return {}


def _collect_new_artifacts(model_path: Path, before: dict) -> list:
    """
    Artefak yang baru dibuat atau ditulis ulang oleh ``model.export()``.

    Folder ikut disertakan karena CoreML menulis ``best.mlpackage`` sebagai
    direktori, bukan file. File ``.pt`` (termasuk ``best.pt`` dan
    ``last.pt``) dikecualikan supaya checkpoint asli tidak ikut terpindah.
    """
    new = []
    for p in model_path.parent.iterdir():
        if p.name.endswith(".pt"):
            continue
        old = before.get(p.name)
        if old is None:
            new.append(p)
            continue
        mtime, size = p.stat().st_mtime, p.stat().st_size
        if mtime > old[0] or (mtime == old[0] and size != old[1]):
            new.append(p)
    return new


def _export(model_path, output_dir, imgsz, fmt, dynamic=False,
            **extra) -> bool:
    """Bagian bersama semua fungsi export."""
    model = load_model(model_path)
    if model is None:
        return False

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = Path(model_path)
        before = _snapshot_weights(model_path)

        opts = {"format": fmt, "imgsz": imgsz}
        if dynamic:
            opts["dynamic"] = True
            opts["simplify"] = True
        opts.update(extra)

        print(f"[INFO] Export ke {fmt.upper()} (imgsz={imgsz}) -> {output_dir}")
        model.export(**opts)

        artifacts = _collect_new_artifacts(model_path, before)
        if not artifacts:
            print(f"[WARNING] Export {fmt.upper()} selesai, tapi tidak ada "
                  f"artefak baru di {model_path.parent}. "
                  f"Cek versi ultralytics.")
            return False

        # Pindahkan ke folder tujuan. Artefak format lain yang tertinggal di
        # folder weights (mis. best.onnx dari run sebelumnya) tidak ikut.
        moved = []
        for src in artifacts:
            dst = output_dir / src.name
            if dst.is_dir():
                shutil.rmtree(dst, ignore_errors=True)
            elif dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
            moved.append(dst)
            kind = "folder" if dst.is_dir() else "file"
            print(f"[OK] Dipindahkan ({kind}): {dst}")

        print(f"[OK] Export {fmt.upper()} berhasil")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal export {fmt.upper()}: {e}")
        return False


def export_to_onnx(model_path, output_dir=None, imgsz=640, dynamic=True):
    """Export model ke ONNX."""
    return _export(model_path, output_dir or DEFAULT_OUTPUT, imgsz, "onnx",
                   dynamic=dynamic)


def export_to_torchscript(model_path, output_dir=None, imgsz=640):
    """Export model ke TorchScript (ringan, tanpa ONNX Runtime)."""
    return _export(model_path, output_dir or DEFAULT_OUTPUT, imgsz, "torchscript")


def export_to_tflite(model_path, output_dir=None, imgsz=640):
    """Export model ke TFLite (untuk mobile)."""
    return _export(model_path, output_dir or DEFAULT_OUTPUT, imgsz, "tflite")


def export_to_coreml(model_path, output_dir=None, imgsz=640):
    """Export model ke CoreML (untuk iOS/macOS)."""
    return _export(model_path, output_dir or DEFAULT_OUTPUT, imgsz, "coreml")


def export_to_openvino(model_path, output_dir=None, imgsz=640):
    """Export model ke OpenVINO (untuk Intel hardware)."""
    return _export(model_path, output_dir or DEFAULT_OUTPUT, imgsz, "openvino")


def export_all_formats(model_path, output_dir=None, imgsz=640):
    """
    Coba export ke kelima format yang didukung Ultralytics.

    ``--format all`` di lama file ini hanya menjalankan ONNX + TorchScript
    tanpa penjelasan, jadi pengguna mengira semua format sudah diekspor.
    Sekarang semua format dicoba dan status per format dicetak, jadi format
    yang gagal karena dependency bisa langsung terlihat.
    """
    output_dir = output_dir or DEFAULT_OUTPUT
    print("\n" + "=" * 60)
    print("EXPORT MODEL KE SEMUA FORMAT")
    print("=" * 60)

    formats = [
        ("onnx", export_to_onnx),
        ("torchscript", export_to_torchscript),
        ("openvino", export_to_openvino),
        ("tflite", export_to_tflite),
        ("coreml", export_to_coreml),
    ]

    results = {}
    for name, func in formats:
        print(f"\n--- Export {name} ---")
        results[name] = func(model_path, output_dir, imgsz)

    print("\n" + "=" * 60)
    print("RINGKASAN EXPORT")
    print("=" * 60)
    for name, success in results.items():
        print(f"  {name:<12}: {'BERHASIL' if success else 'GAGAL'}")
    print(f"\nArtefak ada di: {output_dir}")
    print("Catatan: OpenVINO/TFLite/CoreML butuh dependency tambahan "
          "(openvino, tflite-support/litert, coremltools) dan sering gagal "
          "di CPU-only. Ekspor manual per format bila perlu.")

    return results


def main() -> int:
    """Fungsi utama untuk menjalankan export model"""
    parser = argparse.ArgumentParser(
        description="Export Model YOLO11 ke Berbagai Format"
    )
    parser.add_argument(
        "--model", type=str,
        default=None,
        help="Path ke model .pt (default: dari config, "
             "runs/detect/models/vehicle_detection/weights/best.pt)"
    )
    parser.add_argument(
        "--format", type=str,
        choices=["onnx", "tflite", "torchscript", "coreml", "openvino", "all"],
        default="onnx",
        help="Format export (default: onnx)"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Direktori output (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="Ukuran gambar input (default: dari config deteksi, yaitu 416)"
    )
    args = parser.parse_args()

    # Default: model yang benar-benar dilatih untuk proyek ini.
    # Path lama "models/yolov8n_vehicle/weights/best.pt" tidak pernah ada.
    model_path = args.model
    config = None
    if config is None:
        from utils.paths import load_config
        try:
            config = load_config()
        except FileNotFoundError as exc:
            print(f"[ERROR] {exc}")
            return 1

    if model_path is None:
        try:
            from utils.paths import get_model_path
            model_path = str(get_model_path(config))
        except Exception as exc:
            print(f"[ERROR] Tidak bisa menentukan model dari config: {exc}")
            print("        Jalankan dengan --model <path>.pt")
            return 1

    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        return 1

    imgsz = args.imgsz or int(config.get("detection", {}).get("imgsz", 416))

    print("=" * 60)
    print("EXPORT MODEL YOLO")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Format: {args.format}")
    print(f"Output: {args.output}")
    print(f"Image size: {imgsz}")
    print("=" * 60)

    if args.format == "all":
        results = export_all_formats(model_path, args.output, imgsz)
    else:
        func = {"onnx": export_to_onnx, "tflite": export_to_tflite,
                "torchscript": export_to_torchscript, "coreml": export_to_coreml,
                "openvino": export_to_openvino}[args.format]
        results = {args.format: func(model_path, args.output, imgsz)}

    if any(results.values()):
        return 0
    print("\n[ERROR] Semua format gagal diekspor.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
