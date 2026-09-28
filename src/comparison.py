"""
Perbandingan hasil model dari laporan ``evaluate.py``.

Versi lama file ini tidak punya CLI (``main()`` hanya print), dan
``compare_models()`` hanya bisa membaca ``evaluation_report.json`` yang
tersimpan di dalam folder masing-masing model - sedangkan evaluate.py
menulis semuanya ke satu folder ``outputs/evaluation``, jadi fungsi ini
tidak pernah menemukan apa pun.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd


def load_report(path: Path) -> dict:
    """Baca satu evaluation_report.json dengan pesan error yang jelas."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: JSON rusak ({exc})") from exc


def _row(label: str, report: dict, source: str) -> dict:
    overall = report.get("overall_metrics", {}) or {}
    fps = report.get("fps_metrics", {}) or {}
    cm = report.get("confusion_matrix", {}) or {}
    return {
        "model": label,
        "sumber": source,
        "mAP50": round(float(overall.get("mAP50", 0.0)), 4),
        "mAP50-95": round(float(overall.get("mAP50-95", 0.0)), 4),
        "Precision": round(float(overall.get("Precision", 0.0)), 4),
        "Recall": round(float(overall.get("Recall", 0.0)), 4),
        "F1": round(float(overall.get("F1", 0.0)), 4),
        "FPS": round(float(fps.get("fps", 0.0)), 2),
        "TP": cm.get("tp", ""),
        "FP": cm.get("fp", ""),
        "FN": cm.get("fn", ""),
    }


def compare_reports(paths, out_csv=None, sort_by="mAP50") -> pd.DataFrame:
    """
    Bandingkan beberapa ``evaluation_report.json``.

    Args:
        paths: file laporan, atau folder yang dicari rekursif.
        out_csv: simpan hasil ke CSV
        sort_by: kolom pengurutan
    """
    found: list = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            found.extend(sorted(p.rglob("evaluation_report.json")))
        elif p.is_file():
            found.append(p)
        else:
            print(f"[PERINGATAN] Tidak ditemukan: {p}")

    if not found:
        raise FileNotFoundError(
            "Tidak ada evaluation_report.json. Jalankan dulu:\n"
            "  python src/evaluate.py --task all")

    rows = []
    for f in found:
        report = load_report(f)
        label = report.get("model") or f.parent.name
        label = Path(str(label)).stem
        rows.append(_row(label, report, str(f)))

    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    print("\n=== PERBANDINGAN MODEL ===")
    print(df.to_string(index=False))

    if out_csv:
        out = Path(out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"\n[OK] Disimpan: {out}")

    return df


def per_class_report(report_path) -> pd.DataFrame:
    """Tampilkan metrik per kelas dari satu laporan."""
    report = load_report(Path(report_path))
    per_class = report.get("per_class_metrics") or {}
    if not per_class:
        raise ValueError("Laporan tidak punya per_class_metrics")

    rows = []
    for name, m in per_class.items():
        rows.append({
            "kelas": name,
            "class_id": m.get("class_id", ""),
            "AP50": round(float(m.get("AP50", 0.0)), 4),
            "AP50-95": round(float(m.get("AP50-95", 0.0)), 4),
            "Precision": round(float(m.get("Precision", 0.0)), 4),
            "Recall": round(float(m.get("Recall", 0.0)), 4),
            "F1": round(float(m.get("F1", 0.0)), 4),
        })
    df = pd.DataFrame(rows).sort_values("AP50", ascending=False)
    print("\n=== METRIK PER KELAS ===")
    print(df.to_string(index=False))
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bandingkan laporan evaluasi antar model")
    parser.add_argument("paths", nargs="*",
                        default=["outputs/evaluation"],
                        help="File evaluation_report.json atau folder "
                             "(default: outputs/evaluation)")
    parser.add_argument("--out-csv", default=None, help="Simpan hasil ke CSV")
    parser.add_argument("--sort-by", default="mAP50",
                        choices=["mAP50", "mAP50-95", "F1", "FPS", "Precision",
                                 "Recall"])
    parser.add_argument("--per-class", action="store_true",
                        help="Tampilkan metrik per kelas dari laporan pertama")
    args = parser.parse_args()

    try:
        if args.per_class:
            target = next(
                (p for p in args.paths if Path(p).is_file()), args.paths[0])
            per_class_report(target)
        else:
            compare_reports(args.paths, args.out_csv, args.sort_by)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
