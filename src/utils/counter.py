"""Modul Penghitung Kendaraan Dual-Line
Menghitung kendaraan yang melintasi GARIS PERTAMA dan GARIS KEDUA
"""

import numpy as np
from collections import defaultdict


class VehicleCounter:
    """
    Penghitung kendaraan dual-line.
    
    Kendaraan dihitung jika melintasi kedua garis:
    - Down: Garis 1 (atas) → Garis 2 (bawah) = masuk
    - Up: Garis 2 (bawah) → Garis 1 (atas) = keluar
    """

    def __init__(
        self,
        line1_position=0.35,
        line2_position=0.65,
        direction="both",
        min_track_length=3,
        max_lost_frames=30,
    ):
        self.line1_position = line1_position
        self.line2_position = line2_position
        self.direction = direction
        self.min_track_length = min_track_length
        self.max_lost_frames = max_lost_frames

        # Manajemen lintasan
        self.tracks = {}
        self.next_track_id = 0
        
        # Status lintasan: track_id -> {"seen_above_line1", "seen_below_line2", "counted"}
        self.track_status = {}
        
        self.counted_ids = set()

        # Penghitung
        self.total_count = 0
        self.class_counts = defaultdict(int)
        self.counts_up = defaultdict(int)
        self.counts_down = defaultdict(int)

    def reset(self):
        self.tracks = {}
        self.next_track_id = 0
        self.track_status = {}
        self.counted_ids = set()
        self.total_count = 0
        self.class_counts = defaultdict(int)
        self.counts_up = defaultdict(int)
        self.counts_down = defaultdict(int)

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def update(self, detections, frame_height=None):
        if frame_height is None and detections:
            max_y = max(d["bbox"][3] for d in detections)
            frame_height = max_y * 1.5
        elif frame_height is None:
            return

        line1_y = frame_height * self.line1_position
        line2_y = frame_height * self.line2_position

        current_ids = set()

        for det in detections:
            bbox = det["bbox"]
            class_name = det["class_name"]
            center = self._get_center(bbox)
            cy = center[1]

            track_id = self._find_matching_track(center)

            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    "centers": [center],
                    "class_name": class_name,
                    "bbox": bbox,
                    "lost_frames": 0,
                }
                self.track_status[track_id] = {
                    "seen_above_line1": cy < line1_y,
                    "seen_below_line2": cy > line2_y,
                    "counted": False,
                    "counted_direction": None,
                }
            else:
                track = self.tracks[track_id]
                status = self.track_status.get(track_id, {
                    "seen_above_line1": False,
                    "seen_below_line2": False,
                    "counted": False,
                    "counted_direction": None,
                })

                # Update posisi relatif terhadap garis
                if cy < line1_y:
                    status["seen_above_line1"] = True
                if cy > line2_y:
                    status["seen_below_line2"] = True

                # Cek apakah sudah melewati kedua garis DAN belum dihitung
                if not status["counted"] and status["seen_above_line1"] and status["seen_below_line2"]:
                    if len(track["centers"]) >= self.min_track_length:
                        # Tentukan arah berdasarkan posisi terakhir
                        last_y = track["centers"][-1][1] if track["centers"] else cy
                        prev_y = track["centers"][-2][1] if len(track["centers"]) > 1 else last_y

                        if last_y > prev_y:
                            counted_direction = "down"  # Atas → Bawah
                        else:
                            counted_direction = "up"    # Bawah → Atas

                        # Cek arah yang diizinkan
                        if self.direction == "both" or self.direction == counted_direction:
                            status["counted"] = True
                            status["counted_direction"] = counted_direction
                            self.counted_ids.add(track_id)
                            self.total_count += 1
                            self.class_counts[class_name] += 1

                            if counted_direction == "up":
                                self.counts_up[class_name] += 1
                            else:
                                self.counts_down[class_name] += 1

                # Perbarui lintasan
                track["centers"].append(center)
                track["bbox"] = bbox
                track["class_name"] = class_name
                track["lost_frames"] = 0

                if len(track["centers"]) > 50:
                    track["centers"] = track["centers"][-50:]

                self.track_status[track_id] = status

            current_ids.add(track_id)

        # Update lost frames
        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                self.tracks[track_id]["lost_frames"] += 1
                if self.tracks[track_id]["lost_frames"] > self.max_lost_frames:
                    del self.tracks[track_id]
                    if track_id in self.track_status:
                        del self.track_status[track_id]

    def _find_matching_track(self, center, max_distance=80):
        min_dist = max_distance
        best_id = None

        for track_id, track in self.tracks.items():
            if track["lost_frames"] > 5:
                continue

            last_center = track["centers"][-1] if track["centers"] else None
            if last_center is None:
                continue

            dist = np.sqrt(
                (center[0] - last_center[0]) ** 2
                + (center[1] - last_center[1]) ** 2
            )

            if dist < min_dist:
                min_dist = dist
                best_id = track_id

        return best_id

    def get_count_summary(self):
        return {
            "total": self.total_count,
            "by_class": dict(self.class_counts),
            "by_direction": {
                "up": dict(self.counts_up),
                "down": dict(self.counts_down),
            },
            "active_tracks": len(self.tracks),
        }
