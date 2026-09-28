"""
Query dan export database deteksi.

Versi lama file ini memakai ``DetectionDatabase()`` tanpa argumen, sehingga
path ``data/detections.db`` diselesaikan relatif ke current working
directory. Dijalankan dari folder lain, perintah ini diam-diam membuka -
atau membuat - database kedua yang kosong. Versi lama juga membaca kunci
statistik yang sudah tidak ada (``total_detections`` dan ``total``), jadi
``--action detail`` berakhir dengan KeyError, dan setiap kesalahan argumen
keluar dengan exit code 0 sehingga tidak terdeteksi oleh runner.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.database import DetectionDatabase
from utils.paths import DB_PATH, OUTPUT_DIR


def show_sessions(db) -> None:
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
        print(f"  Baris deteksi: {s['total_detections']}")
        print(f"  Kendaraan terhitung: {s['total_counted']}")
        print(f"  FPS: {s['avg_fps']:.1f}")
        print("-" * 70)


def show_session_detail(db, session_id) -> None:
    """Tampilkan detail sesi."""
    session = db.get_session(session_id)
    if session is None:
        print(f"[ERROR] Sesi #{session_id} tidak ada.")
        return

    stats = db.get_statistics(session_id)

    print("\n" + "=" * 70)
    print(f"DETAIL SESI #{session_id}")
    print("=" * 70)

    print(f"\n  Nama: {session['session_name']}")
    print(f"  Sumber: {session['source_type']} - {session['source_path']}")
    print(f"  Status: {session['status']}")
    print(f"  Waktu: {session['start_time']}")

    print("\n  Statistik:")
    print(f"    Frame diproses: {session['total_frames']}")
    # total_detection_rows = baris track-per-frame, BUKAN jumlah kendaraan.
    print(f"    Baris deteksi (track x frame): {stats['total_detection_rows']}")
    print(f"    Kendaraan terhitung (unik): {stats['total_counted']}")

    if stats["by_class"]:
        print("\n  Per Kelas:")
        print(f"    {'kelas':<10}{'unik':>8}{'terhitung':>12}"
              f"{'baris':>10}{'conf':>8}")
        for cls, data in stats["by_class"].items():
            print(f"    {cls:<10}{data['unique_vehicles']:>8}"
                  f"{data['counted']:>12}{data['track_frames']:>10}"
                  f"{data['avg_confidence']:>8.2f}")

    print("\n  Ringkasan Penghitungan:")
    summary = stats["counting_summary"]
    if not summary:
        print("    (belum ada ringkasan)")
    for cls, directions in summary.items():
        for direction, count in directions.items():
            print(f"    {cls} ({direction}): {count}")

    print("=" * 70)


def export_session(db, session_id, output_path=None) -> int:
    """Ekspor sesi ke JSON."""
    if db.get_session(session_id) is None:
        print(f"[ERROR] Sesi #{session_id} tidak ada.")
        return 1
    # Default lama menulis "data/session_N_export.json" relatif ke cwd.
    output_path = output_path or (OUTPUT_DIR / f"session_{session_id}_export.json")
    path = db.export_to_json(session_id, output_path)
    print(f"\n[OK] Data diekspor ke: {path}")
    return 0


def show_detections(db, session_id, limit=20, class_name=None) -> int:
    """Tampilkan deteksi dari sesi."""
    if db.get_session(session_id) is None:
        print(f"[ERROR] Sesi #{session_id} tidak ada.")
        return 1

    detections = db.get_detections(session_id, class_name=class_name)
    if not detections:
        print(f"\nTidak ada baris deteksi untuk sesi #{session_id}.")
        return 0

    print(f"\n{'=' * 70}")
    print(f"DETEKSI SESI #{session_id} "
          f"(menampilkan {min(limit, len(detections))}/{len(detections)})")
    print(f"{'=' * 70}")
    print(f"{'vehicle_id':<12}{'Frame':<8}{'Kelas':<10}{'Conf':<8}"
          f"{'Arah':<8}{'Terhitung':<10}")
    print("-" * 70)

    for det in detections[:limit]:
        frame = det["frame_number"]
        direction = det.get("direction") or "-"
        print(f"{det['vehicle_id']:<12}"
              f"{(frame if frame is not None else '-'):<8}"
              f"{det['class_name']:<10}{det['confidence']:<8.2f}"
              f"{direction:<8}"
              f"{'Ya' if det['counted'] else 'Tidak'}")

    print("=" * 70)
    return 0


def show_accumulation(db, session_id, class_name=None) -> int:
    """Tampilkan akumulasi kendaraan per kelas."""
    if db.get_session(session_id) is None:
        print(f"[ERROR] Sesi #{session_id} tidak ada.")
        return 1

    accumulation = db.get_accumulation(session_id, class_name=class_name)

    print(f"\n{'=' * 70}")
    print(f"AKUMULASI KENDARAAN SESI #{session_id}")
    print(f"{'=' * 70}")

    if not accumulation:
        print("  Belum ada data akumulasi.")
        return 0

    print(f"\n{'Kelas':<12}{'Total':<10}{'Masuk (down)':<14}{'Keluar (up)':<14}")
    print("-" * 52)

    total_all = 0
    for cls, directions in accumulation.items():
        total = directions.get("total", 0)
        down = directions.get("down", 0)
        up = directions.get("up", 0)
        total_all += total
        print(f"{cls:<12}{total:<10}{down:<14}{up:<14}")

    print("-" * 52)
    print(f"{'TOTAL':<12}{total_all:<10}")
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Query Database Deteksi")
    parser.add_argument("--action", type=str, default="sessions",
                        choices=["sessions", "detail", "export", "detections",
                                 "accumulation"],
                        help="Aksi")
    parser.add_argument("--session", type=int, default=None,
                        help="Session ID (wajib selain action 'sessions')")
    parser.add_argument("--db", default=str(DB_PATH),
                        help=f"Path database (default: {DB_PATH})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path untuk export")
    parser.add_argument("--class", dest="class_name", default=None,
                        help="Filter kelas (mis. mobil)")
    parser.add_argument("--limit", type=int, default=20, help="Limit hasil")
    args = parser.parse_args()

    if not Path(args.db).is_file() and args.action != "sessions":
        print(f"[ERROR] Database tidak ditemukan: {args.db}")
        return 1

    db = DetectionDatabase(args.db)
    try:
        if args.action == "sessions":
            show_sessions(db)
            return 0

        if args.session is None:
            print(f"[ERROR] --session wajib untuk action '{args.action}'")
            return 1

        if args.action == "detail":
            if db.get_session(args.session) is None:
                print(f"[ERROR] Sesi #{args.session} tidak ada.")
                return 1
            show_session_detail(db, args.session)
            return 0
        if args.action == "export":
            return export_session(db, args.session, args.output)
        if args.action == "detections":
            return show_detections(db, args.session, args.limit, args.class_name)
        return show_accumulation(db, args.session, args.class_name)
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
