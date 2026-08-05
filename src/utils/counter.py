"""Modul Penghitung Kendaraan untuk melacak dan menghitung kendaraan yang melewati garis"""

import numpy as np
from collections import defaultdict


class VehicleCounter:
    """
    Penghitung kendaraan yang melintasi garis tertentu pada frame video.
    
    Fitur:
    - Pelacakan lintasan objek
    - Deteksi persilangan garis (masuk/keluar)
    - Penghitungan per kelas (motor/mobil/bus/truk)
    - Penghitungan berdasarkan arah (atas/bawah)
    """

    def __init__(
        self,
        line_position=0.5,
        direction="both",
        min_track_length=5,
        max_lost_frames=30,
    ):
        """
        Inisialisasi penghitung kendaraan.
        
        Args:
            line_position: Posisi garis penghitung (0.0 sampai 1.0, rasio tinggi frame)
                0.0 = bagian atas frame, 1.0 = bagian bawah frame
            direction: Arah penghitungan - "up" (ke atas), "down" (ke bawah), atau "both" (kedua arah)
            min_track_length: Jumlah frame minimum lintasan harus ada sebelum dihitung
            max_lost_frames: Jumlah frame maksimum untuk menyimpan lintasan yang hilang
        """
        self.line_position = line_position
        self.direction = direction
        self.min_track_length = min_track_length
        self.max_lost_frames = max_lost_frames

        # Manajemen lintasan
        self.tracks = {}  # Dictionary lintasan aktif
        self.next_track_id = 0  # ID berikutnya yang akan ditugaskan
        self.counted_ids = set()  # Set ID yang sudah dihitung

        # Penghitung
        self.total_count = 0  # Total kendaraan yang melewati garis
        self.class_counts = defaultdict(int)  # Penghitungan per kelas
        self.counts_up = defaultdict(int)  # Penghitungan ke atas per kelas
        self.counts_down = defaultdict(int)  # Penghitungan ke bawah per kelas

    def reset(self):
        """Mereset semua penghitung dan lintasan ke kondisi awal."""
        self.tracks = {}
        self.next_track_id = 0
        self.counted_ids = set()
        self.total_count = 0
        self.class_counts = defaultdict(int)
        self.counts_up = defaultdict(int)
        self.counts_down = defaultdict(int)

    def _get_center(self, bbox):
        """
        Mendapatkan titik pusat dari bounding box.
        
        Args:
            bbox: koordinat [x1, y1, x2, y2]
            
        Returns:
            Tuple (cx, cy) titik pusat
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _check_line_crossing(self, prev_center, curr_center, frame_height):
        """
        Mengecek apakah objek melewati garis penghitung.
        
        Deteksi persilangan dilakukan dengan membandingkan posisi y
        objek pada frame sebelumnya dengan posisi y saat ini.
        
        Args:
            prev_center: pusat objek pada frame sebelumnya
            curr_center: pusat objek pada frame saat ini
            frame_height: tinggi frame
            
        Returns:
            True jika objek melewati garis
        """
        # Hitung posisi y garis penghitung
        line_y = frame_height * self.line_position

        # Cek apakah objek sebelumnya dan saat ini di atas atau bawah garis
        prev_above = prev_center[1] < line_y
        curr_above = curr_center[1] < line_y

        if self.direction == "up":
            # Hanya hitung yang bergerak ke atas (dari bawah ke atas)
            return prev_above and not curr_above
        elif self.direction == "down":
            # Hanya hitung yang bergerak ke bawah (dari atas ke bawah)
            return not prev_above and curr_above
        else:  # both
            # Hitung kedua arah (persilangan garis)
            return prev_above != curr_above

    def update(self, detections, frame_height=None):
        """
        Memperbarui penghitung dengan deteksi baru.
        
        Proses:
        1. Untuk setiap deteksi, cari lintasan yang cocok atau buat baru
        2. Jika lintasan cocok, cek apakah melewati garis penghitung
        3. Jika melewati garis dan belum dihitung, tambahkan ke penghitung
        4. Perbarui lintasan dengan pusat baru
        5. Hapus lintasan yang sudah hilang terlalu lama
        
        Args:
            detections: daftar dict deteksi dengan key class_id, class_name, confidence, bbox
            frame_height: tinggi frame (diperlukan untuk deteksi persilangan garis)
        """
        # Jika frame_height tidak diberikan, estimasi dari bbox
        if frame_height is None and detections:
            max_y = max(d["bbox"][3] for d in detections)
            frame_height = max_y * 1.5  # Estimasi
        elif frame_height is None:
            return

        current_ids = set()

        for det in detections:
            bbox = det["bbox"]
            class_name = det["class_name"]
            center = self._get_center(bbox)

            # Cari lintasan yang cocok atau buat baru
            track_id = self._find_matching_track(center)

            if track_id is None:
                # Buat lintasan baru
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    "centers": [center],
                    "class_name": class_name,
                    "bbox": bbox,
                    "lost_frames": 0,
                }
            else:
                # Perbarui lintasan yang ada
                track = self.tracks[track_id]
                prev_center = track["centers"][-1] if track["centers"] else center

                # Cek persilangan garis
                if (
                    track_id not in self.counted_ids
                    and len(track["centers"]) >= self.min_track_length
                ):
                    if self._check_line_crossing(prev_center, center, frame_height):
                        # Objek melewati garis, tambahkan ke penghitung
                        self.counted_ids.add(track_id)
                        self.total_count += 1
                        self.class_counts[class_name] += 1

                        # Hitung berdasarkan arah
                        line_y = frame_height * self.line_position
                        if center[1] < prev_center[1]:
                            self.counts_up[class_name] += 1
                        else:
                            self.counts_down[class_name] += 1

                # Perbarui lintasan
                track["centers"].append(center)
                track["bbox"] = bbox
                track["class_name"] = class_name
                track["lost_frames"] = 0

                # Batasi riwayat pusat
                if len(track["centers"]) > 50:
                    track["centers"] = track["centers"][-50:]

            current_ids.add(track_id)

        # Perbarui frame hilang untuk lintasan yang tidak cocok
        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                self.tracks[track_id]["lost_frames"] += 1

                # Hapus lintasan yang sudah hilang terlalu lama
                if self.tracks[track_id]["lost_frames"] > self.max_lost_frames:
                    del self.tracks[track_id]

    def _find_matching_track(self, center, max_distance=80):
        """
        Mencari lintasan yang paling dekat dengan pusat yang diberikan.
        
        Menggunakan pencocokan berbasis jarak Euclidean.
        Hanya mempertimbangkan lintasan yang masih aktif (lost_frames <= 5).
        
        Args:
            center: titik pusat deteksi saat ini
            max_distance: jarak maksimum untuk pencocokan
            
        Returns:
            ID lintasan yang cocok atau None jika tidak ada
        """
        min_dist = max_distance
        best_id = None

        for track_id, track in self.tracks.items():
            # Lewati lintasan yang sudah hilang terlalu lama
            if track["lost_frames"] > 5:
                continue

            last_center = track["centers"][-1] if track["centers"] else None
            if last_center is None:
                continue

            # Hitung jarak Euclidean
            dist = np.sqrt(
                (center[0] - last_center[0]) ** 2
                + (center[1] - last_center[1]) ** 2
            )

            if dist < min_dist:
                min_dist = dist
                best_id = track_id

        return best_id

    def get_count_summary(self):
        """
        Mendapatkan ringkasan semua penghitungan.
        
        Returns:
            Dict dengan struktur:
            {
                "total": jumlah total kendaraan,
                "by_class": {kelas: jumlah},
                "by_direction": {"up": {kelas: jumlah}, "down": {kelas: jumlah}},
                "active_tracks": jumlah lintasan aktif
            }
        """
        return {
            "total": self.total_count,
            "by_class": dict(self.class_counts),
            "by_direction": {
                "up": dict(self.counts_up),
                "down": dict(self.counts_down),
            },
            "active_tracks": len(self.tracks),
        }
