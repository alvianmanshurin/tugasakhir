"""Script untuk Query dan Export Database Deteksi"""

import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from utils.database import DetectionDatabase


def show_sessions(db):
    """Tampilkan daftar sesi."""
    sessions = db.get_sessions()
    print("\n" + "=" * 70)
    print("DAFTAR SESI DETEKSI")
    print("=" * 70)
    
    if not sessions:
        print("  Belum ada sesi.")
        return
    
    for s in sessions:
        print(f"\n  ID: {s['id']}")
        print(f"  Nama: {s['session_name']}")
        print(f"  Sumber: {s['source_type']} - {s['source_path']}")
        print(f"  Waktu: {s['start_time']}")
        print(f"  Status: {s['status']}")
        print(f"  Frame: {s['total_frames']}")
        print(f"  Deteksi: {s['total_detections']}")
        print(f"  Terhitung: {s['total_counted']}")
        print(f"  FPS: {s['avg_fps']:.1f}")
        print("-" * 70)


def show_session_detail(db, session_id):
    """Tampilkan detail sesi."""
    stats = db.get_statistics(session_id)
    
    print("\n" + "=" * 70)
    print(f"DETAIL SESI #{session_id}")
    print("=" * 70)
    
    session = stats["session"]
    print(f"\n  Nama: {session['session_name']}")
    print(f"  Sumber: {session['source_type']}")
    print(f"  Status: {session['status']}")
    
    print(f"\n  Statistik:")
    print(f"    Total Deteksi: {stats['total_detections']}")
    print(f"    Total Terhitung: {stats['total_counted']}")
    
    print(f"\n  Per Kelas:")
    for cls, data in stats["by_class"].items():
        print(f"    {cls}:")
        print(f"      Total: {data['total']}")
        print(f"      Terhitung: {data['counted']}")
        print(f"      Avg Confidence: {data['avg_confidence']:.2f}")
    
    print(f"\n  Ringkasan Penghitungan:")
    for cls, directions in stats["counting_summary"].items():
        for direction, count in directions.items():
            print(f"    {cls} ({direction}): {count}")
    
    print("=" * 70)


def export_session(db, session_id, output_path=None):
    """Ekspor sesi ke JSON."""
    path = db.export_to_json(session_id, output_path)
    print(f"\n[OK] Data diekspor ke: {path}")


def show_detections(db, session_id, limit=20):
    """Tampilkan deteksi dari sesi."""
    detections = db.get_detections(session_id)
    
    print(f"\n{'='*70}")
    print(f"DETEKSI SESI #{session_id} (menampilkan {min(limit, len(detections))}/{len(detections)})")
    print(f"{'='*70}")
    print(f"{'ID':<10} {'Frame':<8} {'Kelas':<10} {'Conf':<8} {'Terhitung':<10}")
    print("-" * 70)
    
    for det in detections[:limit]:
        print(f"{det['vehicle_id']:<10} {det['frame_number'] or '-':<8} "
              f"{det['class_name']:<10} {det['confidence']:.2f}    "
              f"{'Ya' if det['counted'] else 'Tidak'}")
    
    print("=" * 70)


def show_accumulation(db, session_id):
    """Tampilkan akumulasi kendaraan per kelas."""
    accumulation = db.get_accumulation(session_id)
    
    print(f"\n{'='*70}")
    print(f"AKUMULASI KENDARAAN SESI #{session_id}")
    print(f"{'='*70}")
    
    if not accumulation:
        print("  Belum ada data akumulasi.")
        return
    
    print(f"\n{'Kelas':<12} {'Total':<10} {'Masuk (down)':<12} {'Keluar (up)':<12}")
    print("-" * 50)
    
    total_all = 0
    for cls, directions in accumulation.items():
        total = directions.get("total", 0)
        down = directions.get("down", 0)
        up = directions.get("up", 0)
        total_all += total
        print(f"{cls:<12} {total:<10} {down:<12} {up:<12}")
    
    print("-" * 50)
    print(f"{'TOTAL':<12} {total_all:<10}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Query Database Deteksi")
    parser.add_argument("--action", type=str, default="sessions",
                       choices=["sessions", "detail", "export", "detections", "accumulation"],
                       help="Aksi")
    parser.add_argument("--session", type=int, default=None, help="Session ID")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--limit", type=int, default=20, help="Limit hasil")
    args = parser.parse_args()
    
    db = DetectionDatabase()
    
    if args.action == "sessions":
        show_sessions(db)
    
    elif args.action == "detail":
        if not args.session:
            print("[ERROR] Session ID harus diisi: --session <id>")
            return
        show_session_detail(db, args.session)
    
    elif args.action == "export":
        if not args.session:
            print("[ERROR] Session ID harus diisi: --session <id>")
            return
        export_session(db, args.session, args.output)
    
    elif args.action == "detections":
        if not args.session:
            print("[ERROR] Session ID harus diisi: --session <id>")
            return
        show_detections(db, args.session, args.limit)
    
    elif args.action == "accumulation":
        if not args.session:
            print("[ERROR] Session ID harus diisi: --session <id>")
            return
        show_accumulation(db, args.session)
    
    db.close()


if __name__ == "__main__":
    main()
