"""Pelacak Objek Ringan (terinspirasi ByteTrack) untuk CPU
Tanpa dependensi eksternal - pencocokan greedy berbasis numpy.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Deteksi di bawah ambang ini dianggap "low confidence" dan hanya dipakai
# untuk rescuing track yang sudah ada (tidak pernah membuat track baru).
# Nilai ini WAJIB di bawah model.confidence_threshold, kalau tidak tahap
# low-confidence tidak akan pernah berjalan karena NMS sudah membuang
# deteksi yang lebih rendah dari ambang tersebut.
LOW_CONF_THRESHOLD = 0.25


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


class ObjectTracker:
    """
    Pelacak multi-objek ringan terinspirasi ByteTrack.

    Algoritma:
    1. Deteksi high-confidence (>= LOW_CONF_THRESHOLD) dicocokkan ke track
    2. Sisa deteksi high-confidence dicocokkan lagi dengan deteksi
       low-confidence ke track yang belum cocok (rescue)
    3. Track baru dibuat hanya dari deteksi high-confidence
    4. Track yang tidak cocok diusia dan dihapus jika melewati max_age

    Catatan: track baru dibuat dengan hits=1 dan TIDAK langsung
    dikembalikan sebagai track terkonfirmasi. Sebuah track baru hanya
    mulai dikembalikan setelah mencapai min_hits, sehingga kedipan
    deteksi 1-frame tidak pernah jadi track.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        max_distance: float = 80.0,
        track_buffer: int = 50,
        low_conf_threshold: float = LOW_CONF_THRESHOLD,
    ):
        """
        Inisialisasi pelacak objek.

        Args:
            max_age: hapus track setelah N frame tanpa kecocokan
            min_hits: jumlah kecocokan minimum sebelum track dikembalikan
            max_distance: jarak pixel maksimum untuk pencocokan
            track_buffer: jumlah riwayat pusat yang disimpan
            low_conf_threshold: ambang pemisah deteksi high/low confidence
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.track_buffer = track_buffer
        self.low_conf_threshold = low_conf_threshold

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

        Args:
            detections: daftar dict deteksi dengan key:
                - class_id: ID kelas objek
                - class_name: nama kelas objek
                - confidence: skor confidence
                - bbox: koordinat [x1, y1, x2, y2]

        Returns:
            Daftar track terkonfirmasi sebagai dict deteksi, masing-masing
            sudah membawa ``track_id`` (ruang ID milik tracker ini).
        """
        self.frame_count += 1

        matched_track_ids: set = set()

        if detections:
            high_conf = [d for d in detections if d["confidence"] >= self.low_conf_threshold]
            low_conf = [d for d in detections if d["confidence"] < self.low_conf_threshold]

            matched_det_indices: set = set()

            # Tahap 1: high-confidence ke track yang ada
            if high_conf:
                for det_idx, track_id in self._match_tracks(high_conf):
                    matched_det_indices.add(det_idx)
                    matched_track_ids.add(track_id)
                    self._update_track(track_id, high_conf[det_idx])

            # Tahap 2: high-confidence yang belum cocok dicoba rescue dengan
            # deteksi low-confidence, hanya ke track yang juga belum cocok.
            leftover = [
                d for i, d in enumerate(high_conf) if i not in matched_det_indices
            ]
            if leftover and low_conf:
                for det_idx, track_id in self._match_tracks(
                    low_conf, exclude_ids=matched_track_ids
                ):
                    matched_track_ids.add(track_id)
                    self._update_track(track_id, low_conf[det_idx])

            # Tahap 3: track baru hanya dari high-confidence yang belum cocok
            for i, det in enumerate(high_conf):
                if i not in matched_det_indices:
                    matched_track_ids.add(self._create_track(det))

        # Usia track yang tidak mendapat match di frame ini
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self.tracks[tid].time_since_update += 1
                self.tracks[tid].age += 1

        self._cleanup()

        return self._get_confirmed_tracks()

    def _match_tracks(
        self, detections: List[dict], exclude_ids: set = None
    ) -> List[Tuple[int, int]]:
        """
        Mencocokkan deteksi ke track yang ada menggunakan jarak pusat.

        Args:
            detections: daftar deteksi
            exclude_ids: set track_id yang dikecualikan

        Returns:
            Daftar pasangan (deteksi_idx, track_id)
        """
        if not self.tracks or not detections:
            return []

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

                dist = float(np.hypot(
                    det_center[0] - track.center[0],
                    det_center[1] - track.center[1],
                ))

                # Kelas berbeda = tidak boleh dipasangkan, bukan "dikit mahal".
                # DICOMPAT: memakai +500 akan otomatis benar selama
                # max_distance < 500, tapi salah diam-diam bila max_distance
                # dinaikkan. Gerbang eksplisit lebih aman.
                if det["class_id"] != track.class_id:
                    cost_matrix[i, j] = np.inf
                    continue

                cost_matrix[i, j] = dist

        # Pencocokan greedy (cepat, cukup baik untuk real-time)
        matches = []
        used_dets = set()
        used_tracks = set()

        indices = np.argsort(cost_matrix.ravel())

        for flat_idx in indices:
            det_idx = flat_idx // len(track_ids)
            track_idx = flat_idx % len(track_ids)

            if det_idx in used_dets or track_idx in used_tracks:
                continue

            cost = cost_matrix[det_idx, track_idx]
            if not np.isfinite(cost) or cost > self.max_distance:
                continue

            matches.append((det_idx, track_ids[track_idx]))
            used_dets.add(det_idx)
            used_tracks.add(track_idx)

        return matches

    def _create_track(self, det: dict) -> int:
        """
        Membuat track baru dari deteksi.

        Returns:
            track_id yang baru dibuat
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
        return tid

    def _update_track(self, track_id: int, det: dict):
        """
        Memperbarui track yang ada dengan deteksi baru.

        Pembaruan meliputi:
        - Bounding box baru
        - Confidence baru
        - Kecepatan baru (menggunakan EMA untuk kehalusan)
        - Riwayat pusat baru
        - Increment hits dan reset time_since_update
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

        track.bbox = det["bbox"]
        track.confidence = det["confidence"]
        track.class_name = det["class_name"]
        track.class_id = det["class_id"]
        track.hits += 1
        track.time_since_update = 0
        track.age += 1
        track.centers.append(new_center)

        if len(track.centers) > self.track_buffer:
            track.centers = track.centers[-self.track_buffer:]

    def _cleanup(self):
        """
        Menghapus track yang sudah melewati batas usia (max_age).
        """
        to_remove = [
            tid for tid, t in self.tracks.items()
            if t.time_since_update > self.max_age
        ]
        for tid in to_remove:
            del self.tracks[tid]

    def _get_confirmed_tracks(self) -> List[dict]:
        """
        Track yang TERBARU di-update frame ini DAN sudah mencapai min_hits.

        Dua syarat, keduanya wajib:
          - ``time_since_update == 0``: track benar-benar punya deteksi di
            frame ini. Tanpa syarat ini, track yang hilang (terhalang, blur,
            keluar ROI) tetap dikembalikan sebagai "deteksi" sampai max_age
            frame, lalu digambar, ditulis ke database, dan menambah
            ``observations`` di counter tanpa ada ukuran baru.
          - ``hits >= min_hits``: track harus matang sebelum dilacak.

        Track yang tidak ter-update tetap DISIMPAN supaya bisa dicocokkan
        lagi di frame berikutnya (coasting), hanya tidak dikembalikan.
        """
        results = []
        for tid, track in self.tracks.items():
            if track.time_since_update != 0:
                continue
            if track.hits < self.min_hits:
                continue
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
