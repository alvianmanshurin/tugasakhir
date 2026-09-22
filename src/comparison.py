"""
Script untuk perbandingan model
"""
import os
import json
import pandas as pd
from pathlib import Path

def compare_models(results_dir):
    """Membandingkan hasil beberapa model"""
    results = []
    
    for model_dir in Path(results_dir).iterdir():
        if model_dir.is_dir():
            report_path = model_dir / "evaluation_report.json"
            if report_path.exists():
                with open(report_path, "r") as f:
                    report = json.load(f)
                    results.append({
                        "model": model_dir.name,
                        "mAP50": report.get("overall_metrics", {}).get("mAP50", 0),
                        "Precision": report.get("overall_metrics", {}).get("Precision", 0),
                        "Recall": report.get("overall_metrics", {}).get("Recall", 0),
                        "FPS": report.get("fps_metrics", {}).get("fps", 0),
                    })
    
    if results:
        df = pd.DataFrame(results)
        print("\n=== PERBANDINGAN MODEL ===")
        print(df.to_string(index=False))
        return df
    else:
        print("Tidak ada hasil yang ditemukan")
        return None

def main():
    print("=== PERBANDINGAN MODEL ===")
    # Tambahkan kode sesuai kebutuhan

if __name__ == "__main__":
    main()
