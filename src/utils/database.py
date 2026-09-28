"""Modul Database untuk Menyimpan Hasil Deteksi Kendaraan

SQLite untuk penyimpanan lokal.

Catatan desain penting:
- ``detections`` menyimpan SATU BARIS PER TRACK PER FRAME. Jadi
  ``COUNT(*)`` dari tabel itu menghitung "frame di mana kendaraan terlihat",
  bukan jumlah kendaraan. Semua metrik jumlah kendaraan harus memakai
  ``COUNT(DISTINCT vehicle_id)`` - lihat ``get_counted_vehicles()``.
- Koneksi dibuka dengan ``check_same_thread=False`` + ``WAL`` karena dipakai
  dari worker thread GUI, dan dilindungi ``threading.Lock``.
"""

import json
import sqlite3
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

SCHEMA_VERSION = 2

# DDL tabel dipisah dari _create_tables() supaya bisa dipakai ulang saat
# migrasi membangun ulang tabel untuk menambahkan UNIQUE constraint.
# {table} diganti nama tabel (atau nama sementara saat rebuild).
_TABLE_DDL = {
    "counting_summary": """
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            direction TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """,
    "vehicle_accumulation": """
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            direction TEXT NOT NULL DEFAULT 'total',
            count INTEGER DEFAULT 0,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """,
}

# Constraint UNIQUE yang wajib ada agar UPSERT di save_counting_summary()
# dan accumulate_vehicles_batch() bekerja.
_REQUIRED_UNIQUE = {
    "counting_summary": ("session_id", "class_name", "direction"),
    "vehicle_accumulation": ("session_id", "class_name", "direction"),
}


def _now_iso() -> str:
    """Timestamp lokal sebagai string ISO (portabel, tidak deprecated)."""
    return datetime.now().isoformat(sep=" ", timespec="seconds")


class DetectionDatabase:
    """
    Database untuk menyimpan hasil deteksi kendaraan.

    Struktur tabel:
    - sessions: Sesi deteksi (video/cctv/webcam)
    - detections: Data track per frame (satu baris per track per frame)
    - frame_stats: Statistik per frame
    - counting_summary: Ringkasan penghitungan per sesi
    - vehicle_accumulation: Akumulasi kendaraan per kelas
    """

    def __init__(self, db_path=None):
        # Default harus absolut terhadap root project. Versi lama memakai
        # "data/detections.db" yang relatif ke current working directory,
        # sehingga database yang sama tercipta di beberapa folder tergantung
        # dari mana skrip dijalankan.
        if db_path is None:
            try:
                from utils.paths import DB_PATH
                db_path = DB_PATH
            except ImportError:
                db_path = Path(__file__).resolve().parent.parent.parent \
                    / "data" / "detections.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn: Optional[sqlite3.Connection] = None
        self._create_tables()
        self._migrate()
        self._create_indexes()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Buat koneksi dan tabel jika belum ada."""
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=15.0,
        )
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA foreign_keys=ON")

            cur.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_path TEXT,
                    start_time TEXT DEFAULT CURRENT_TIMESTAMP,
                    end_time TEXT,
                    total_frames INTEGER DEFAULT 0,
                    total_detections INTEGER DEFAULT 0,
                    total_counted INTEGER DEFAULT 0,
                    avg_fps REAL DEFAULT 0,
                    status TEXT DEFAULT 'running'
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    class_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    bbox_x1 REAL NOT NULL,
                    bbox_y1 REAL NOT NULL,
                    bbox_x2 REAL NOT NULL,
                    bbox_y2 REAL NOT NULL,
                    center_x REAL NOT NULL,
                    center_y REAL NOT NULL,
                    frame_number INTEGER,
                    timestamp REAL,
                    direction TEXT,
                    counted BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS frame_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    frame_number INTEGER NOT NULL,
                    detections_count INTEGER DEFAULT 0,
                    after_roi_count INTEGER DEFAULT 0,
                    tracked_count INTEGER DEFAULT 0,
                    counted_count INTEGER DEFAULT 0,
                    fps REAL DEFAULT 0,
                    timestamp REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
            """)

            # counting_summary dan vehicle_accumulation dibuat dari _TABLE_DDL
            # supaya DDL-nya hanya ada di satu tempat. Both-nya langsung
            # dibuat DENGAN UNIQUE; kalau tabel lama sudah ada tanpa UNIQUE,
            # _migrate() yang membangun ulang.
            for table, key in _REQUIRED_UNIQUE.items():
                cur.execute(self._ddl_with_unique(table, key))

            self.conn.commit()

    def _create_indexes(self) -> None:
        """
        Index untuk query per sesi.

        Harus dipanggil SETELAH _migrate(): db lama belum punya kolom
        ``counted``, jadi CREATE INDEX akan gagal dengan
        "no such column: counted".
        """
        with self._lock:
            cur = self.conn.cursor()
            cur.execute("CREATE INDEX IF NOT EXISTS idx_det_session ON detections(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_det_session_counted "
                        "ON detections(session_id, counted)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_det_vehicle "
                        "ON detections(session_id, vehicle_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_frame_session "
                        "ON frame_stats(session_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_summary_session "
                        "ON counting_summary(session_id)")
            self.conn.commit()

    def _migrate(self) -> None:
        """
        Terapkan migrasi skema secara bertahap via PRAGMA user_version.

        Hanya ``CREATE TABLE IF NOT EXISTS`` tidak cukup: menambahkan kolom
        ke file .db yang sudah ada akan gagal diam-diam saat INSERT dengan
        "no such column".
        """
        with self._lock:
            cur = self.conn.cursor()
            current = cur.execute("PRAGMA user_version").fetchone()[0]

            if current >= SCHEMA_VERSION:
                # user_version tidak menjamin constraint sudah benar:
                # build sebelumnya sudah pernah men-stamp 2 tanpa UNIQUE, dan
                # early-return di sini membuat finalize_session() berikutnya
                # gagal dengan "ON CONFLICT clause does not match any
                # PRIMARY KEY or UNIQUE constraint". Pemeriksaan UNIQUE
                # harus tetap jalan pada database yang sudah di-stamp.
                self._dedupe_summary_tables()
                for table, key in _REQUIRED_UNIQUE.items():
                    self._ensure_unique_constraint(table, key)
                self.conn.commit()
                return

            if current < 1:
                self._add_column_if_missing("detections", "direction", "TEXT")
                self._add_column_if_missing("detections", "counted", "BOOLEAN DEFAULT 0")
                self._add_column_if_missing("detections", "created_at",
                                            "TEXT DEFAULT CURRENT_TIMESTAMP")

            # Baris ganda di counting_summary dari versi lama (INSERT tanpa
            # UNIQUE) harus dihapus SEBELUM tabel dibangun ulang, kalau tidak
            # INSERT ... SELECT ke tabel yang sudah punya UNIQUE gagal dengan
            # IntegrityError dan database tidak bisa dibuka sama sekali.
            self._dedupe_summary_tables()

            # SQLite tidak bisa menambahkan constraint UNIQUE ke tabel yang
            # sudah ada, dan save_counting_summary memakai
            # "ON CONFLICT(session_id, class_name, direction) DO UPDATE".
            # Di database lama, counting_summary dibuat tanpa UNIQUE -
            # migrasi di atas hanya menghapus duplikat, TIDAK menambahkan
            # constraint, jadi setiap finalize_session() berikutnya gagal
            # dengan "ON CONFLICT clause does not match any PRIMARY KEY or
            # UNIQUE constraint". Jadi tabelnya harus dibangun ulang.
            for table, key in _REQUIRED_UNIQUE.items():
                self._ensure_unique_constraint(table, key)

            cur.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self.conn.commit()

    def _dedupe_summary_tables(self) -> None:
        """
        Hapus baris ganda pada tabel ringkasan sesuai kunci UNIQUE-nya.

        Versi lama bisa punya beberapa baris untuk kombinasi
        (session_id, class_name, direction) yang sama. Baris dengan id
        terbesar dipertahankan karena itu hasil hitungan terakhir.
        """
        cur = self.conn.cursor()
        for table, key in _REQUIRED_UNIQUE.items():
            exists = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,)).fetchone()
            if not exists:
                continue
            columns = {r["name"] for r in
                       cur.execute(f"PRAGMA table_info({table})").fetchall()}
            if not set(key).issubset(columns):
                continue
            cur.execute(
                f"DELETE FROM {table} WHERE id NOT IN ("
                f"SELECT MAX(id) FROM {table} GROUP BY {', '.join(key)})")

    @staticmethod
    def _ddl_with_unique(base: str, key: Tuple[str, ...], name: Optional[str] = None) -> str:
        """
        DDL tabel dengan UNIQUE(key) disisipkan.

        Args:
            base: kunci di _TABLE_DDL ("counting_summary" / "vehicle_accumulation")
            key: kolom UNIQUE
            name: nama tabel di hasil DDL (default: sama dengan ``base``)
        """
        table = name or base
        ddl = _TABLE_DDL[base].format(table=table)
        close = ddl.rstrip().rfind(")")
        return (ddl[:close] + f",\n    UNIQUE({', '.join(key)})\n" + ddl[close:])

    def _ensure_unique_constraint(self, table: str, columns: Tuple[str, ...]) -> None:
        """
        Pastikan tabel punya UNIQUE constraint di ``columns``.

        Kalau constraint sudah ada, tidak melakukan apa-apa. Kalau belum -
        termasuk pada tabel lama yang dibuat tanpa UNIQUE - tabel dibangun
        ulang: salin baris yang tersisa, drop tabel lama, buat ulang dengan
        DDL yang sama plus UNIQUE, lalu salin balik. Baris duplikat sudah
        dihapus migrasi sebelumnya, jadi tidak ada data yang hilang.
        """
        cur = self.conn.cursor()
        key = tuple(columns)

        for row in cur.execute(f"PRAGMA index_list({table})").fetchall():
            if not row["unique"]:
                continue
            indexed = tuple(r["name"] for r in
                            cur.execute(f"PRAGMA index_info({row['name']})").fetchall())
            if indexed == key:
                return

        if table not in _TABLE_DDL:
            return

        existing = {r["name"] for r in cur.execute(f"PRAGMA table_info({table})")}
        if not existing:
            return  # tabel belum ada; _create_tables yang akan membuatnya

        # Hanya bangun ulang kalau semua kolom kunci ada.
        if not set(key).issubset(existing):
            return

        self._rebuild_table_with_unique(table, key)

    def _rebuild_table_with_unique(self, table: str, key: Tuple[str, ...]) -> None:
        """
        Bangun ulang ``table`` dengan UNIQUE constraint, pertahankan data.

        Tabel baru dibuat dari ``_TABLE_DDL`` - DDL yang dipakai kode sekarang
        - lalu datanya disalin ulang hanya untuk kolom yang ada di kedua
        skema. Kolom sisa versi lama sengaja dibuang: kalau dipertahankan,
        kolom NOT NULL yang tidak lagi diisi kode membuat setiap INSERT
        gagal dengan "NOT NULL constraint failed".
        """
        cur = self.conn.cursor()
        old_cols = [r["name"] for r in
                    cur.execute(f"PRAGMA table_info({table})").fetchall()]
        if not old_cols:
            return

        tmp = f"{table}_rebuild"
        cur.execute(f"DROP TABLE IF EXISTS {tmp}")
        cur.execute(self._ddl_with_unique(table, key, name=tmp))
        new_cols = [r["name"] for r in
                    cur.execute(f"PRAGMA table_info({tmp})").fetchall()]

        copy_cols = [c for c in old_cols if c in new_cols]
        dropped = [c for c in old_cols if c not in new_cols]
        if dropped:
            print(f"[DB] Kolom lama di {table} tidak lagi dipakai dan "
                  f"dibuang saat migrasi: {', '.join(dropped)}")

        col_ddl = ", ".join(copy_cols)
        cur.execute(f"INSERT INTO {tmp} ({col_ddl}) SELECT {col_ddl} FROM {table}")
        cur.execute(f"DROP TABLE {table}")
        cur.execute(f"ALTER TABLE {tmp} RENAME TO {table}")
        self.conn.commit()

    def _add_column_if_missing(self, table: str, column: str, ddl: str) -> None:
        cur = self.conn.cursor()
        cols = {r["name"] for r in cur.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    # ------------------------------------------------------------------
    # Sesi
    # ------------------------------------------------------------------

    def start_session(self, session_name, source_type, source_path=None):
        """Mulai sesi deteksi baru. Returns: ID sesi."""
        with self._lock:
            cur = self.conn.execute("""
                INSERT INTO sessions (session_name, source_type, source_path,
                                      start_time, status)
                VALUES (?, ?, ?, ?, 'running')
            """, (session_name, source_type, source_path, _now_iso()))
            self.conn.commit()
            return cur.lastrowid

    def end_session(self, session_id, total_frames=0, total_detections=0,
                    total_counted=0, avg_fps=0.0):
        """Tandai sesi selesai beserta ringkasannya."""
        with self._lock:
            self.conn.execute("""
                UPDATE sessions
                SET end_time = ?, total_frames = ?, total_detections = ?,
                    total_counted = ?, avg_fps = ?, status = 'completed'
                WHERE id = ?
            """, (_now_iso(), total_frames, total_detections, total_counted,
                  float(avg_fps), session_id))
            self.conn.commit()

    def mark_session_interrupted(self, session_id) -> None:
        """
        Tandai sesi yang mati di tengah jalan (mis. proses di-kill).

        Sesi seperti ini akan terlihat di query sebagai ``interrupted``
        alih-alih ``completed``, supaya tidak tercampur dengan run yang
        benar-benar selesai.
        """
        with self._lock:
            self.conn.execute(
                "UPDATE sessions SET status = 'interrupted' WHERE id = ? AND end_time IS NULL",
                (session_id,),
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    # Penyimpanan
    # ------------------------------------------------------------------

    @staticmethod
    def _detection_row(session_id, det: dict) -> tuple:
        x1, y1, x2, y2 = det["bbox"]
        return (
            session_id,
            det.get("vehicle_id", ""),
            det.get("class_id", 0),
            det.get("class_name", "unknown"),
            det.get("confidence", 0.0),
            x1, y1, x2, y2,
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            det.get("frame_number"),
            det.get("timestamp"),
            det.get("direction"),
            1 if det.get("counted") else 0,
        )

    _INSERT_DETECTION = """
        INSERT INTO detections (session_id, vehicle_id, class_id, class_name,
                                confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                center_x, center_y, frame_number, timestamp,
                                direction, counted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    def save_detection(self, session_id, vehicle_id, class_id, class_name,
                       confidence, bbox, frame_number=None, timestamp=None,
                       direction=None, counted=False):
        """Simpan satu deteksi (1 baris). Untuk loop, pakai save_detections_batch."""
        det = {
            "vehicle_id": vehicle_id,
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "bbox": bbox,
            "frame_number": frame_number,
            "timestamp": timestamp,
            "direction": direction,
            "counted": counted,
        }
        with self._lock:
            self.conn.execute(self._INSERT_DETECTION, self._detection_row(session_id, det))
            self.conn.commit()

    def save_detections_batch(self, session_id, detections_list):
        """
        Simpan semua track satu frame dalam SATU transaksi.

        Dipakai di loop video: satu commit per frame, bukan satu commit
        per kendaraan per frame.
        """
        if not detections_list:
            return
        rows = [self._detection_row(session_id, d) for d in detections_list]
        with self._lock:
            self.conn.executemany(self._INSERT_DETECTION, rows)
            self.conn.commit()

    def save_frame_stats(self, session_id, frame_number, detections_count,
                         after_roi_count, tracked_count, counted_count, fps,
                         timestamp=None):
        """Simpan statistik per frame."""
        with self._lock:
            self.conn.execute("""
                INSERT INTO frame_stats (session_id, frame_number, detections_count,
                                        after_roi_count, tracked_count, counted_count,
                                        fps, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, frame_number, detections_count, after_roi_count,
                  tracked_count, counted_count, fps, timestamp))
            self.conn.commit()

    def save_counting_summary(self, session_id, class_counts, direction_counts):
        """
        Simpan ringkasan hitungan. Aman dipanggil ulang (UPSERT).
        """
        rows = []
        for class_name, count in (class_counts or {}).items():
            rows.append((session_id, class_name, "total", int(count)))
        for direction, counts in (direction_counts or {}).items():
            for class_name, count in (counts or {}).items():
                rows.append((session_id, class_name, direction, int(count)))

        if not rows:
            return

        with self._lock:
            self.conn.executemany("""
                INSERT INTO counting_summary (session_id, class_name, direction, count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id, class_name, direction)
                DO UPDATE SET count = excluded.count
            """, rows)
            self.conn.commit()

    def accumulate_vehicles_batch(self, session_id, class_counts, direction_counts=None):
        """Akumulasi jumlah kendaraan per kelas (menambah, bukan menimpa)."""
        rows = []
        for class_name, count in (class_counts or {}).items():
            if count:
                rows.append((session_id, class_name, "total", int(count)))
        for direction, counts in (direction_counts or {}).items():
            for class_name, count in (counts or {}).items():
                if count:
                    rows.append((session_id, class_name, direction, int(count)))

        if not rows:
            return

        now = _now_iso()
        with self._lock:
            self.conn.executemany("""
                INSERT INTO vehicle_accumulation
                    (session_id, class_name, direction, count, last_updated)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id, class_name, direction)
                DO UPDATE SET count = count + excluded.count,
                              last_updated = excluded.last_updated
            """, [r + (now,) for r in rows])
            self.conn.commit()

    def finalize_session(self, session_id, summary: dict) -> None:
        """
        Tulis seluruh penutup sesi dalam SATU transaksi.

        Sebelumnya tiga panggilan terpisah, sehingga proses yang mati di
        tengah meninggalkan sesi setengah tertulis.
        """
        with self._lock:
            try:
                self.conn.execute("""
                    UPDATE sessions
                    SET end_time = ?, total_frames = ?, total_detections = ?,
                        total_counted = ?, avg_fps = ?, status = 'completed'
                    WHERE id = ?
                """, (_now_iso(), summary.get("total_frames", 0),
                      summary.get("total_detections", 0),
                      summary.get("total_counted", 0),
                      float(summary.get("avg_fps", 0.0)), session_id))

                rows = []
                for class_name, count in (summary.get("by_class") or {}).items():
                    rows.append((session_id, class_name, "total", int(count)))
                for direction, counts in (summary.get("by_direction") or {}).items():
                    for class_name, count in (counts or {}).items():
                        rows.append((session_id, class_name, direction, int(count)))

                if rows:
                    self.conn.executemany("""
                        INSERT INTO counting_summary (session_id, class_name, direction, count)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(session_id, class_name, direction)
                        DO UPDATE SET count = excluded.count
                    """, rows)

                acc_rows = []
                for class_name, count in (summary.get("by_class") or {}).items():
                    if count:
                        acc_rows.append((session_id, class_name, "total", int(count)))
                for direction, counts in (summary.get("by_direction") or {}).items():
                    for class_name, count in (counts or {}).items():
                        if count:
                            acc_rows.append((session_id, class_name, direction, int(count)))
                if acc_rows:
                    self.conn.executemany("""
                        INSERT INTO vehicle_accumulation (session_id, class_name, direction, count)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(session_id, class_name, direction)
                        DO UPDATE SET count = count + excluded.count
                    """, acc_rows)

                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Pembacaan
    # ------------------------------------------------------------------

    @staticmethod
    def _rows_to_dicts(rows) -> list:
        return [dict(r) for r in rows]

    def get_session(self, session_id):
        """Data satu sesi, atau None bila tidak ada."""
        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def get_sessions(self, limit=50):
        """Daftar sesi terbaru."""
        rows = self.conn.execute(
            "SELECT * FROM sessions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return self._rows_to_dicts(rows)

    def get_detections(self, session_id, class_name=None, counted_only=False,
                       limit=None):
        """
        Baris deteksi (satu per track per frame).

        Jangan dipakai untuk menghitung "berapa kendaraan" - pakai
        get_counted_vehicles().
        """
        query = "SELECT * FROM detections WHERE session_id = ?"
        params = [session_id]
        if class_name:
            query += " AND class_name = ?"
            params.append(class_name)
        if counted_only:
            query += " AND counted = 1"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        return self._rows_to_dicts(self.conn.execute(query, params).fetchall())

    def get_counted_vehicles(self, session_id):
        """
        Kendaraan yang terhitung, satu entri per vehicle_id.

        Ini angka yang benar untuk "jumlah kendaraan" - bukan jumlah baris.
        """
        rows = self.conn.execute("""
            SELECT vehicle_id,
                   MIN(class_id)   AS class_id,
                   MIN(class_name) AS class_name,
                   MAX(direction)  AS direction,
                   MIN(frame_number) AS first_frame,
                   MAX(frame_number) AS last_frame,
                   COUNT(*)        AS frames_visible
            FROM detections
            WHERE session_id = ? AND counted = 1
            GROUP BY vehicle_id
            ORDER BY first_frame
        """, (session_id,)).fetchall()
        return self._rows_to_dicts(rows)

    def count_counted_vehicles(self, session_id) -> int:
        """Jumlah kendaraan terhitung (DISTINCT vehicle_id)."""
        row = self.conn.execute("""
            SELECT COUNT(DISTINCT vehicle_id)
            FROM detections WHERE session_id = ? AND counted = 1
        """, (session_id,)).fetchone()
        return int(row[0] or 0)

    def get_counted_by_class(self, session_id) -> dict:
        """Jumlah kendaraan terhitung per kelas (DISTINCT vehicle_id)."""
        rows = self.conn.execute("""
            SELECT class_name, COUNT(DISTINCT vehicle_id) AS n
            FROM detections
            WHERE session_id = ? AND counted = 1
            GROUP BY class_name
        """, (session_id,)).fetchall()
        return {r["class_name"]: int(r["n"]) for r in rows}

    def get_counting_summary(self, session_id):
        """Ringkasan hitungan per sesi."""
        rows = self.conn.execute("""
            SELECT class_name, direction, count
            FROM counting_summary WHERE session_id = ?
        """, (session_id,)).fetchall()
        summary = defaultdict(dict)
        for r in rows:
            summary[r["class_name"]][r["direction"]] = int(r["count"])
        return dict(summary)

    def get_accumulation(self, session_id, class_name=None):
        """Akumulasi kendaraan lintas sesi."""
        if class_name:
            rows = self.conn.execute("""
                SELECT class_name, direction, count FROM vehicle_accumulation
                WHERE session_id = ? AND class_name = ?
            """, (session_id, class_name)).fetchall()
        else:
            rows = self.conn.execute("""
                SELECT class_name, direction, count FROM vehicle_accumulation
                WHERE session_id = ?
            """, (session_id,)).fetchall()

        out = defaultdict(dict)
        for r in rows:
            out[r["class_name"]][r["direction"]] = int(r["count"])
        return dict(out)

    def get_frame_stats(self, session_id, limit=None):
        """Statistik per frame."""
        query = "SELECT * FROM frame_stats WHERE session_id = ? ORDER BY frame_number"
        params = [session_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        return self._rows_to_dicts(self.conn.execute(query, params).fetchall())

    def get_statistics(self, session_id):
        """
        Statistik lengkap sesi.

        ``total_detections`` = jumlah baris track-per-frame (baris DB).
        ``total_counted``    = jumlah kendaraan unik (DISTINCT vehicle_id).
        Keduanya sengaja dibedakan; sebelumnya keduanya salah memakai
        jumlah baris sehingga "2 kendaraan" dilaporkan sebagai "85".
        """
        session = self.get_session(session_id)
        detection_rows = self.conn.execute(
            "SELECT COUNT(*) FROM detections WHERE session_id = ?", (session_id,)
        ).fetchone()[0]

        class_stats = {}
        for r in self.conn.execute("""
            SELECT class_name,
                   COUNT(DISTINCT vehicle_id) AS unique_vehicles,
                   COUNT(*)                   AS track_frames,
                   AVG(confidence)            AS avg_confidence
            FROM detections WHERE session_id = ?
            GROUP BY class_name
        """, (session_id,)).fetchall():
            class_stats[r["class_name"]] = {
                "unique_vehicles": int(r["unique_vehicles"]),
                "track_frames": int(r["track_frames"]),
                "avg_confidence": float(r["avg_confidence"] or 0.0),
                "counted": 0,
            }

        counted_by_class = self.get_counted_by_class(session_id)
        for cls, n in counted_by_class.items():
            class_stats.setdefault(cls, {
                "unique_vehicles": 0, "track_frames": 0,
                "avg_confidence": 0.0, "counted": 0,
            })
            class_stats[cls]["counted"] = n

        return {
            "session": session,
            "total_detection_rows": int(detection_rows),
            "total_counted": self.count_counted_vehicles(session_id),
            "by_class": class_stats,
            "counting_summary": self.get_counting_summary(session_id),
        }

    def export_to_json(self, session_id, output_path=None):
        """Ekspor data sesi ke JSON."""
        stats = self.get_statistics(session_id)
        if output_path is None:
            output_path = f"data/session_{session_id}_export.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)
        return output_path

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Tutup koneksi database (idempotent)."""
        with self._lock:
            if self.conn is not None:
                self.conn.commit()
                self.conn.close()
                self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
