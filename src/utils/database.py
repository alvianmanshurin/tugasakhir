"""Modul Database untuk Menyimpan Hasil Deteksi Kendaraan
Menggunakan SQLite untuk penyimpanan lokal
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict


class DetectionDatabase:
    """
    Database untuk menyimpan hasil deteksi kendaraan.
    
    Struktur tabel:
    - sessions: Sesi deteksi (video/cctv/webcam)
    - detections: Data deteksi per kendaraan
    - counting_summary: Ringkasan penghitungan per sesi
    """

    def __init__(self, db_path="data/detections.db"):
        """
        Inisialisasi database.
        
        Args:
            db_path: Path file database SQLite
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._create_tables()

    def _create_tables(self):
        """Membuat tabel jika belum ada."""
        self.conn = sqlite3.connect(str(self.db_path))
        cursor = self.conn.cursor()

        # Tabel sesi deteksi
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_path TEXT,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                total_frames INTEGER DEFAULT 0,
                total_detections INTEGER DEFAULT 0,
                total_counted INTEGER DEFAULT 0,
                avg_fps REAL DEFAULT 0,
                status TEXT DEFAULT 'running'
            )
        """)

        # Tabel deteksi per kendaraan
        cursor.execute("""
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Tabel ringkasan penghitungan
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS counting_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                direction TEXT NOT NULL,
                count INTEGER DEFAULT 0,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        # Tabel statistik per frame
        cursor.execute("""
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
            )
        """)

        # Tabel akumulasi kendaraan per kelas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_accumulation (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                direction TEXT NOT NULL DEFAULT 'total',
                count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id),
                UNIQUE(session_id, class_name, direction)
            )
        """)

        self.conn.commit()

    def start_session(self, session_name, source_type, source_path=None):
        """
        Memulai sesi deteksi baru.
        
        Args:
            session_name: Nama sesi
            source_type: Jenis sumber (video/cctv/webcam/image)
            source_path: Path sumber (opsional)
            
        Returns:
            ID sesi yang baru dibuat
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_name, source_type, source_path, start_time, status)
            VALUES (?, ?, ?, ?, 'running')
        """, (session_name, source_type, source_path, datetime.now()))
        self.conn.commit()
        return cursor.lastrowid

    def end_session(self, session_id, total_frames=0, total_detections=0, 
                    total_counted=0, avg_fps=0):
        """
        Mengakhiri sesi deteksi.
        
        Args:
            session_id: ID sesi
            total_frames: Total frame diproses
            total_detections: Total deteksi
            total_counted: Total terhitung
            avg_fps: FPS rata-rata
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE sessions 
            SET end_time = ?, total_frames = ?, total_detections = ?,
                total_counted = ?, avg_fps = ?, status = 'completed'
            WHERE id = ?
        """, (datetime.now(), total_frames, total_detections, 
              total_counted, avg_fps, session_id))
        self.conn.commit()

    def save_detection(self, session_id, vehicle_id, class_id, class_name,
                      confidence, bbox, frame_number=None, timestamp=None,
                      direction=None, counted=False):
        """
        Menyimpan data deteksi kendaraan.
        
        Args:
            session_id: ID sesi
            vehicle_id: ID unik kendaraan (dari tracker)
            class_id: ID kelas (0=motor, 1=mobil, 2=bus, 3=truk)
            class_name: Nama kelas
            confidence: Confidence score
            bbox: Bounding box [x1, y1, x2, y2]
            frame_number: Nomor frame
            timestamp: Timestamp deteksi
            direction: Arah gerak (up/down)
            counted: Apakah sudah dihitung
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO detections (session_id, vehicle_id, class_id, class_name,
                                   confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                   center_x, center_y, frame_number, timestamp,
                                   direction, counted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, vehicle_id, class_id, class_name, confidence,
              x1, y1, x2, y2, center_x, center_y, frame_number,
              timestamp, direction, counted))
        self.conn.commit()

    def save_detections_batch(self, session_id, detections_list):
        """
        Menyimpan banyak deteksi sekaligus (batch).
        
        Args:
            session_id: ID sesi
            detections_list: List dict deteksi
        """
        cursor = self.conn.cursor()
        rows = []
        
        for det in detections_list:
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            rows.append((
                session_id,
                det.get("vehicle_id", ""),
                det.get("class_id", 0),
                det.get("class_name", "unknown"),
                det.get("confidence", 0),
                x1, y1, x2, y2,
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                det.get("frame_number"),
                det.get("timestamp"),
                det.get("direction"),
                det.get("counted", False),
            ))

        cursor.executemany("""
            INSERT INTO detections (session_id, vehicle_id, class_id, class_name,
                                   confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                   center_x, center_y, frame_number, timestamp,
                                   direction, counted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        self.conn.commit()

    def save_frame_stats(self, session_id, frame_number, detections_count,
                        after_roi_count, tracked_count, counted_count, fps, timestamp=None):
        """Menyimpan statistik per frame."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO frame_stats (session_id, frame_number, detections_count,
                                    after_roi_count, tracked_count, counted_count,
                                    fps, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, frame_number, detections_count, after_roi_count,
              tracked_count, counted_count, fps, timestamp))
        self.conn.commit()

    def accumulate_vehicle_count(self, session_id, class_name, direction="total", increment=1):
        """
        Mengakumulasi jumlah kendaraan per kelas.
        
        Args:
            session_id: ID sesi
            class_name: Nama kelas (motor/mobil/bus/truk)
            direction: Arah (total/up/down)
            increment: Jumlah penambahan
        """
        cursor = self.conn.cursor()
        
        # Cek apakah sudah ada data
        cursor.execute("""
            SELECT count FROM vehicle_accumulation 
            WHERE session_id = ? AND class_name = ? AND direction = ?
        """, (session_id, class_name, direction))
        
        row = cursor.fetchone()
        
        if row:
            # Update jumlah
            new_count = row[0] + increment
            cursor.execute("""
                UPDATE vehicle_accumulation 
                SET count = ?, last_updated = CURRENT_TIMESTAMP
                WHERE session_id = ? AND class_name = ? AND direction = ?
            """, (new_count, session_id, class_name, direction))
        else:
            # Insert baru
            cursor.execute("""
                INSERT INTO vehicle_accumulation (session_id, class_name, direction, count)
                VALUES (?, ?, ?, ?)
            """, (session_id, class_name, direction, increment))
        
        self.conn.commit()

    def accumulate_vehicles_batch(self, session_id, class_counts, direction_counts=None):
        """
        Mengakumulasi banyak kendaraan sekaligus.
        
        Args:
            session_id: ID sesi
            class_counts: Dict {class_name: count} untuk total
            direction_counts: Dict {"up": {class: count}, "down": {class: count}}
        """
        cursor = self.conn.cursor()
        
        # Akumulasi total per kelas
        for class_name, count in class_counts.items():
            if count > 0:
                cursor.execute("""
                    INSERT INTO vehicle_accumulation (session_id, class_name, direction, count)
                    VALUES (?, ?, 'total', ?)
                    ON CONFLICT(session_id, class_name, direction) 
                    DO UPDATE SET count = count + ?, last_updated = CURRENT_TIMESTAMP
                """, (session_id, class_name, count, count))
        
        # Akumulasi per arah
        if direction_counts:
            for direction, counts in direction_counts.items():
                for class_name, count in counts.items():
                    if count > 0:
                        cursor.execute("""
                            INSERT INTO vehicle_accumulation (session_id, class_name, direction, count)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(session_id, class_name, direction) 
                            DO UPDATE SET count = count + ?, last_updated = CURRENT_TIMESTAMP
                        """, (session_id, class_name, direction, count, count))
        
        self.conn.commit()

    def get_accumulation(self, session_id, class_name=None):
        """
        Mendapatkan data akumulasi kendaraan.
        
        Args:
            session_id: ID sesi
            class_name: Filter per kelas (opsional)
            
        Returns:
            Dict akumulasi kendaraan
        """
        cursor = self.conn.cursor()
        
        if class_name:
            cursor.execute("""
                SELECT class_name, direction, count 
                FROM vehicle_accumulation 
                WHERE session_id = ? AND class_name = ?
            """, (session_id, class_name))
        else:
            cursor.execute("""
                SELECT class_name, direction, count 
                FROM vehicle_accumulation 
                WHERE session_id = ?
            """, (session_id,))
        
        accumulation = {}
        for cls, direction, count in cursor.fetchall():
            if cls not in accumulation:
                accumulation[cls] = {}
            accumulation[cls][direction] = count
        
        return accumulation

    def save_counting_summary(self, session_id, class_counts, direction_counts):
        """
        Menyimpan ringkasan penghitungan.
        
        Args:
            session_id: ID sesi
            class_counts: Dict {class_name: count}
            direction_counts: Dict {"up": {class: count}, "down": {class: count}}
        """
        cursor = self.conn.cursor()
        
        # Simpan total per kelas
        for class_name, count in class_counts.items():
            cursor.execute("""
                INSERT INTO counting_summary (session_id, class_name, direction, count)
                VALUES (?, ?, 'total', ?)
            """, (session_id, class_name, count))
        
        # Simpan per arah
        for direction, counts in direction_counts.items():
            for class_name, count in counts.items():
                cursor.execute("""
                    INSERT INTO counting_summary (session_id, class_name, direction, count)
                    VALUES (?, ?, ?, ?)
                """, (session_id, class_name, direction, count))
        
        self.conn.commit()

    def get_session(self, session_id):
        """Mendapatkan data sesi."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, row))
        return None

    def get_sessions(self, limit=50):
        """Mendapatkan daftar sesi."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?", (limit,))
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_detections(self, session_id, class_name=None, counted_only=False):
        """Mendapatkan deteksi dari sesi."""
        cursor = self.conn.cursor()
        query = "SELECT * FROM detections WHERE session_id = ?"
        params = [session_id]
        
        if class_name:
            query += " AND class_name = ?"
            params.append(class_name)
        
        if counted_only:
            query += " AND counted = 1"
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_counting_summary(self, session_id):
        """Mendapatkan ringkasan penghitungan sesi."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT class_name, direction, count 
            FROM counting_summary 
            WHERE session_id = ?
        """, (session_id,))
        
        summary = defaultdict(lambda: defaultdict(int))
        for class_name, direction, count in cursor.fetchall():
            summary[class_name][direction] = count
        
        return dict(summary)

    def get_statistics(self, session_id):
        """Mendapatkan statistik lengkap sesi."""
        session = self.get_session(session_id)
        detections = self.get_detections(session_id)
        counted = self.get_detections(session_id, counted_only=True)
        summary = self.get_counting_summary(session_id)
        
        # Hitung per kelas
        class_stats = defaultdict(lambda: {"total": 0, "counted": 0, "avg_confidence": 0})
        conf_sum = defaultdict(float)
        
        for det in detections:
            cls = det["class_name"]
            class_stats[cls]["total"] += 1
            conf_sum[cls] += det["confidence"]
        
        for det in counted:
            cls = det["class_name"]
            class_stats[cls]["counted"] += 1
        
        for cls in class_stats:
            if class_stats[cls]["total"] > 0:
                class_stats[cls]["avg_confidence"] = conf_sum[cls] / class_stats[cls]["total"]
        
        return {
            "session": session,
            "total_detections": len(detections),
            "total_counted": len(counted),
            "by_class": dict(class_stats),
            "counting_summary": summary,
        }

    def export_to_json(self, session_id, output_path=None):
        """Ekspor data sesi ke JSON."""
        stats = self.get_statistics(session_id)
        
        if output_path is None:
            output_path = f"data/session_{session_id}_export.json"
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)
        
        return output_path

    def close(self):
        """Menutup koneksi database."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
