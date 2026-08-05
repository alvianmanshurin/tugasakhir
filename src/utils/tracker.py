"""Pelacak Objek Ringan (terinspirasi ByteTrack) untuk CPU
Tanpa dependensi eksternal - menggunakan penugasan Hungarian dengan scipy atau numpy.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class Track:
    """
    Data class untuk satu objek yang sedang dilacak.
    
    Menyimpan informasi:
    - ID pelacakan unik
    - Kelas objek (motor/mobil/bus/truk)
    - Bounding box terkini
    - Riwayat pusat objek
    - Kecepatan objek
    - Status pelacakan (usia, jumlah kecocokan, waktu sejak pembaruan)
    """
    track_id: int          # ID unik pelacakan
    class_id: int          # ID kelas objek
    class_name: str        # Nama kelas objek
    bbox: List[float]      # Bounding box [x1, y1, x2, y2]
    confidence: float      # Confidence score
    age: int = 0           # Jumlah frame sejak pembuatan
    hits: int = 1          # Total deteksi yang cocok
    time_since_update: int = 0  # Jumlah frame sejak pembaruan terakhir
    centers: list = field(default_factory=list)  # Riwayat pusat objek
    velocity: Tuple[float, float] = (0.0, 0.0)  # Kecepatan (vx, vy)

    @property
    def center(self):
        """Menghitung pusat bounding box saat ini."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def is_confirmed) -> bool:
        """Mengecek apakah track sudah terkonfirmasi (minimal 3 kecocokan)."""
        return self.hits >= 3

    @property
    def is_lost(self) -> bool:
        """Mengecek apakah track sudah hilang (tidak cocok > 30 frame)."""
        return self.time_since_update > 30


class ObjectTracker:
    """
    Pelacak multi-objek ringan terinspirasi ByteTrack.
    
    Algoritma:
    1. Deteksi high-confidence (>= 0.5) dicocokkan terlebih dahulu
    2. Deteksi low-confidence (< 0.5) dicocokkan ke track yang belum cocok
    3. Track baru dibuat untuk deteksi high-confidence yang belum cocok
    4. Track yang tidak cocok diusia dan dihapus jika melewati batas
    
    Fitur:
    - Pencocokan berbasis jarak pusat (center distance)
    - Estimasi kecepatan objek menggunakan EMA
    - Pelacakan riwayat pusat untuk analisis lintasan
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        max_distance: float = 80.0,
        track_buffer: int = 50,
    ):
        """
        Inisialisasi pelacak objek.
        
        Args:
            max_age: hapus track setelah N frame tanpa kecocokan
            min_hits: jumlah kecocokan minimum sebelum track terkonfirmasi
            max_distance: jarak pixel maksimum untuk pencocokan
            track_buffer: jumlah riwayat pusat yang disimpan
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.track_buffer = track_buffer

        self.tracks: Dict[int, Track] = {}  # Dictionary track aktif
        self.next_id = 0  # ID berikutnya yang akan ditugaskan
        self.frame_count = 0  # Penghitung frame

    def reset(self):
        """Mereset semua pelacak ke kondisi awal."""
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0

    def update(self, detections: List[dict]) -> List[dict]:
        """
        Memperbarui pelacak dengan deteksi baru.
        
        Proses utama:
        1. Pisahkan deteksi high-confidence dan low-confidence
        2. Cocokkan high-confidence ke track yang ada
        3. Cocokkan low-confidence ke track yang belum cocok
        4. Buat track baru untuk high-confidence yang belum cocok
        5. Usia semua track yang tidak cocok
        6. Hapus track yang sudah mati
        
        Args:
            detections: daftar dict deteksi dengan key:
                - class_id: ID kelas objek
                - class_name: nama kelas objek
                - confidence: skor confidence
                - bbox: koordinat [x1, y1, x2, y2]
            
        Returns:
            Daftar objek yang dilacak (deteksi + track_id + velocity)
        """
        self.frame_count += 1

        # Jika tidak ada deteksi, usia semua track
        if not detections:
            self._age_unmatched([])
            return self._get_confirmed_tracks()

        # Langkah 1: Pisahkan high-confidence dan low-confidence
        high_conf = [d for d in detections if d["confidence"] >= 0.5]
        low_conf = [d for d in detections if d["confidence"] < 0.5]

        matched_ids = set()

        # Cocokkan high-confidence terlebih dahulu
        if high_conf:
            matches = self._match_tracks(high_conf)
            for det_idx, track_id in matches:
                matched_ids.add(det_idx)
                self._update_track(track_id, high_conf[det_idx])

        # Cocokkan low-confidence (hanya ke track yang belum cocok)
        if low_conf:
            matches = self._match_tracks(low_conf, exclude_ids=matched_ids)
            for det_idx, track_id in matches:
                self._update_track(track_id, low_conf[det_idx])

        # Buat track baru untuk high-confidence yang belum cocok
        for i, det in enumerate(high_conf):
            if i not in matched_ids:
                self._create_track(det)

        # Usia semua track yang tidak cocok
        all_matched_track_ids = set(m[1] for m in self._match_tracks(high_conf + low_conf))
        for tid in list(self.tracks.keys()):
            if tid not in all_matched_track_ids:
                self.tracks[tid].time_since_update += 1
                self.tracks[tid].age += 1

        # Hapus track yang sudah mati
        self._cleanup()

        return self._get_confirmed_tracks()

    def _match_tracks(
        self, detections: List[dict], exclude_ids: set = None
    ) -> List[Tuple[int, int]]:
        """
        Mencocokkan deteksi ke track yang ada menggunakan jarak pusat.
        
        Algoritma pencocokan greedy:
        1. Buat matriks biaya (jarak pusat + penalti kelas berbeda)
        2. Urutkan semua pasangan berdasarkan biaya
        3. Pilih pasangan dengan biaya terkecil secara berurutan
        4. Lewati jika sudah ada deteksi/track yang terpakai
        
        Args:
            detections: daftar deteksi
            exclude_ids: set ID track yang dikecualikan
            
        Returns:
            Daftar pasangan (deteksi_idx, track_id)
        """
        if not self.tracks or not detections:
            return []

        # Bangun matriks biaya
        track_ids = [tid for tid in self.tracks
                     if self.tracks[tid].time_since_update <= 5
                     and (exclude_ids is None or tid not in exclude_ids)]

        if not track_ids:
            return []

        cost_matrix = np.zeros((len(detections), len(track_ids)))

        for i, det in enumerate(detections):
            det_center = self._bbox_center(det["bbox"])
            for j, tid in enumerate(track_ids):
                track = self.tracks[tid]
                track_center = track.center

                # Hitung jarak pusat Euclidean
                dist = np.sqrt(
                    (det_center[0] - track_center[0]) ** 2 +
                    (det_center[1] - track_center[1]) ** 2
                )

                # Tambahkan penalti jika kelas berbeda
                if det["class_id"] != track.class_id:
                    dist += 500

                cost_matrix[i, j] = dist

        # Pencocokan greedy (cepat, cukup baik untuk real-time)
        matches = []
        used_dets = set()
        used_tracks = set()

        # Urutkan berdasarkan biaya (terkecil ke terbesar)
        indices = np.argsort(cost_matrix.ravel())

        for flat_idx in indices:
            det_idx = flat_idx // len(track_ids)
            track_idx = flat_idx % len(track_ids)

            # Lewati jika sudah terpakai
            if det_idx in used_dets or track_idx in used_tracks:
                continue

            # Berhenti jika biaya terlalu besar
            if cost_matrix[det_idx, track_idx] > self.max_distance:
                break

            matches.append((det_idx, track_ids[track_idx]))
            used_dets.add(det_idx)
            used_tracks.add(track_idx)

        return matches

    def _create_track(self, det: dict):
        """
        Membuat track baru dari deteksi.
        
        Args:
            det: dict deteksi dengan key bbox, class_id, class_name, confidence
        """
        tid = self.next_id
        self.next_id += 1
        center = self._bbox_center(det["bbox"])
        self.tracks[tid] = Track(
            track_id=tid,
            class_id=det["class_id"],
            class_name=det["class_name"],
            bbox=det["bbox"],
            confidence=det["confidence"],
            centers=[center],
        )

    def _update_track(self, track_id: int, det: dict):
        """
        Memperbarui track yang ada dengan deteksi baru.
        
        Pembaruan meliputi:
        - Bounding box baru
        - Confidence baru
        - Kecepatan baru (menggunakan EMA untuk kehalusan)
        - Riwayat pusat baru
        - Increment hits dan reset time_since_update
        
        Args:
            track_id: ID track yang akan diperbarui
            det: dict deteksi baru
        """
        track = self.tracks[track_id]
        old_center = track.center
        new_center = self._bbox_center(det["bbox"])

        # Perbarui kecepatan menggunakan EMA (Exponential Moving Average)
        vx = new_center[0] - old_center[0]
        vy = new_center[1] - old_center[1]
        track.velocity = (
            0.7 * track.velocity[0] + 0.3 * vx,
            0.7 * track.velocity[1] + 0.3 * vy,
        )

        # Perbarui atribut track
        track.bbox = det["bbox"]
        track.confidence = det["confidence"]
        track.class_name = det["class_name"]
        track.class_id = det["class_id"]
        track.hits += 1
        track.time_since_update = 0
        track.age += 1
        track.centers.append(new_center)

        # Batasi riwayat pusat
        if len(track.centers) > self.track_buffer:
            track.centers = track.centers[-self.track_buffer:]

    def _age_unmatched(self, _):
        """
        Menambah usia semua track yang tidak cocok.
        
        Dipanggil ketika tidak ada deteksi atau ada track yang tidak cocok.
        """
        for tid in self.tracks:
            self.tracks[tid].time_since_update += 1
            self.tracks[tid].age += 1

    def _cleanup(self):
        """
        Menghapus track yang sudah melewati batas usia (max_age).
        
        Track dianggap mati jika time_since_update > max_age.
        """
        to_remove = [
            tid for tid, t in self.tracks.items()
            if t.time_since_update > self.max_age
        ]
        for tid in to_remove:
            del self.tracks[tid]

    def _get_confirmed_tracks(self) -> List[dict]:
        """
        Mengembalikan track yang sudah terkonfirmasi sebagai dict deteksi.
        
        Track dianggap terkonfirmasi jika:
        - hits >= min_hits (minimal 3 kecocokan), ATAU
        - time_since_update == 0 (baru saja cocok)
        
        Returns:
            Daftar dict dengan key track_id, class_id, class_name, bbox,
            confidence, velocity, age, hits
        """
        results = []
        for tid, track in self.tracks.items():
            if track.hits >= self.min_hits or track.time_since_update == 0:
                results.append({
                    "track_id": tid,
                    "class_id": track.class_id,
                    "class_name": track.class_name,
                    "bbox": track.bbox,
                    "confidence": track.confidence,
                    "velocity": track.velocity,
                    "age": track.age,
                    "hits": track.hits,
                })
        return results

    @staticmethod
    def _bbox_center(bbox):
        """
        Menghitung pusat bounding box.
        
        Args:
            bbox: koordinat [x1, y1, x2, y2]
            
        Returns:
            Tuple (cx, cy) pusat bounding box
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_track_count(self) -> dict:
        """
        Mendapatkan jumlah objek unik berdasarkan kelas.
        
        Returns:
            Dict dengan key nama kelas dan value jumlah track
            Contoh: {"motor": 5, "mobil": 3}
        """
        counts = defaultdict(int)
        for t in self.tracks.values():
            if t.hits >= self.min_hits:
                counts[t.class_name] += 1
        return dict(counts)
