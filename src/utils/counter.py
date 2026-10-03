"""Modul Penghitung Kendaraan Dual-Line.

Hitungan dilakukan atas ``track_id`` yang sudah diberikan ``ObjectTracker``.
Counter ini TIDAK menjalankan tracker kedua: memakai ID sendiri akan
menciptakan dua ruang ID yang tidaksinkron, sehingga kolom ``counted``
di database bisa bernilai False untuk kendaraan yang sebenarnya sudah
terhitung.

Kendtaraan dihitung hanya bila benar-benar melewati kedua garis secara
terurut, bukan sekadar "pernah di atas line1" dan "pernah di bawah line2"
pada frame yang tidak berurutan.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import numpy as np


class VehicleCounter:
    """
    Penghitung kendaraan dual-line.

    Sebuah lintasan dihitung bila:
      - arah "down" (masuk): lintasan pertama terlihat di atas line1,
        lalu kemudian melewati line2 ke bawah.
      - arah "up" (keluar): lintasan pertama terlihat di bawah line2,
        lalu kemudian melewati line1 ke atas.

    Arah ditentukan oleh urutan lintasan, bukan oleh perbandingan dua
    sampel terakhir, sehingga satu frame jitter tidak dapat salah
    menentukan arah.
    """

    def __init__(
        self,
        line1_position: float = 0.6676,
        line2_position: float = 0.7139,
        direction: str = "both",
        min_track_length: int = 3,
        min_displacement: float = 25.0,
        suppression_frames: int = 30,
    ):
        if not 0.0 <= line1_position < line2_position <= 1.0:
            raise ValueError(
                f"line1_position ({line1_position}) harus < line2_position ({line2_position})"
            )

        self.line1_position = line1_position
        self.line2_position = line2_position
        self.direction = direction
        self.min_track_length = min_track_length
        self.min_displacement = min_displacement
        # Cegah hitung ganda ketika track ter-fragmentasi: track baru yang
        # muncul di dekat posisi track yang baru saja terhitung, dalam
        # N frame, dianggap kelanjutan kendaraan yang sama.
        self.suppression_frames = suppression_frames

        self.track_status: Dict[int, Dict[str, Any]] = {}
        self.counted_ids: Set[int] = set()
        self.counted_directions: Dict[int, str] = {}

        self.total_count = 0
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.counts_up: Dict[str, int] = defaultdict(int)
        self.counts_down: Dict[str, int] = defaultdict(int)

        self._frame_index = 0
        self._recent_counts: List[Dict[str, Any]] = []

    def reset(self) -> None:
        """Reset seluruh state. Dipanggil sebelum run baru."""
        self.track_status = {}
        self.counted_ids = set()
        self.counted_directions = {}
        self.total_count = 0
        self.class_counts = defaultdict(int)
        self.counts_up = defaultdict(int)
        self.counts_down = defaultdict(int)
        self._frame_index = 0
        self._recent_counts = []

    @staticmethod
    def _get_center(bbox) -> tuple:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _is_suppressed(self, track_id: int, center: tuple, class_name: str) -> bool:
        """True bila track ini kemungkinan kelanjutan kendaraan terhitung."""
        if not self._recent_counts:
            return False

        age = self._frame_index - self._recent_counts[-1]["frame"]
        if age > self.suppression_frames:
            return False

        last = self._recent_counts[-1]
        if last["class_name"] != class_name:
            return False

        dist = float(np.hypot(center[0] - last["center"][0], center[1] - last["center"][1]))
        # toleransi 2x min_displacement: cukup untuk jitter track, terlalu kecil
        # untuk kendaraan berbeda yang kebetulan lewat berdekatan
        return dist <= 2 * self.min_displacement

    def update(self, detections: List[dict], frame_height: float) -> None:
        """
        Perbarui hitungan untuk satu frame.

        Args:
            detections: hasil ``ObjectTracker.update()``, tiap item harus punya
                key ``track_id``, ``class_name``, dan ``bbox``.
            frame_height: tinggi frame dalam pixel.
        """
        if not frame_height:
            raise ValueError("frame_height wajib diisi (nilai 0 tidak valid)")

        self._frame_index += 1

        line1_y = frame_height * self.line1_position
        line2_y = frame_height * self.line2_position
        tolerance = self.min_displacement

        for det in detections:
            if "track_id" not in det:
                raise KeyError(
                    "deteksi dari tracker harus punya 'track_id'; "
                    "counter tidak lagi menjalankan tracker sendiri"
                )

            track_id = det["track_id"]
            class_name = det["class_name"]
            center = self._get_center(det["bbox"])
            cy = center[1]

            status = self.track_status.get(track_id)

            if status is None:
                # first sighting: zona awal menentukan arah yang mungkin
                if self._is_suppressed(track_id, center, class_name):
                    continue

                if cy < line1_y:
                    initial_zone = "above1"
                elif cy > line2_y:
                    initial_zone = "below2"
                else:
                    initial_zone = "between"

                status = {
                    "class_name": class_name,
                    "first_y": cy,
                    "min_y": cy,
                    "max_y": cy,
                    "last_y": cy,
                    "observations": 1,
                    "initial_zone": initial_zone,
                    "counted": False,
                    "counted_direction": None,
                }
                self.track_status[track_id] = status
            else:
                status["observations"] += 1
                status["min_y"] = min(status["min_y"], cy)
                status["max_y"] = max(status["max_y"], cy)
                status["last_y"] = cy
                status["class_name"] = class_name

            if status["counted"]:
                continue

            # Syarat minimum lintasan sudah cukup panjang DAN benar-benar
            # bergerak (bukan jitter di tempat).
            if status["observations"] < self.min_track_length:
                continue
            if (status["max_y"] - status["min_y"]) < tolerance:
                continue

            # Deteksi lintasan terurut. min/max dicari pada frame saat ini
            # juga, sehingga ambang tidak meleset satu frame.
            if status["initial_zone"] == "above1" and status["max_y"] > line2_y + tolerance:
                direction = "down"
            elif status["initial_zone"] == "below2" and status["min_y"] < line1_y - tolerance:
                direction = "up"
            else:
                continue

            if self.direction not in ("both", direction):
                continue

            status["counted"] = True
            status["counted_direction"] = direction
            self.counted_ids.add(track_id)
            self.counted_directions[track_id] = direction
            self.total_count += 1
            self.class_counts[class_name] += 1

            if direction == "up":
                self.counts_up[class_name] += 1
            else:
                self.counts_down[class_name] += 1

            self._recent_counts.append(
                {"frame": self._frame_index, "center": center, "class_name": class_name}
            )
            if len(self._recent_counts) > 32:
                self._recent_counts.pop(0)

    def direction_of(self, track_id: int) -> Optional[str]:
        """Arah yang dicatat untuk track_id, atau None bila belum terhitung."""
        return self.counted_directions.get(track_id)

    def is_counted(self, track_id: int) -> bool:
        """True bila track_id sudah masuk hitungan (ruang ID tracker)."""
        return track_id in self.counted_ids

    def get_count_summary(self) -> dict:
        return {
            "total": self.total_count,
            "by_class": dict(self.class_counts),
            "by_direction": {
                "up": dict(self.counts_up),
                "down": dict(self.counts_down),
            },
            "active_tracks": len(self.track_status),
        }
