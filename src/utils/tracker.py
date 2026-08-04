"""Lightweight Object Tracker (ByteTrack-inspired) for CPU
No external dependencies - uses Hungarian assignment with scipy or numpy.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class Track:
    """Single tracked object."""
    track_id: int
    class_id: int
    class_name: str
    bbox: List[float]
    confidence: float
    age: int = 0              # frames since creation
    hits: int = 1             # total matched detections
    time_since_update: int = 0  # frames since last matched
    centers: list = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3

    @property
    def is_lost(self) -> bool:
        return self.time_since_update > 30


class ObjectTracker:
    """Lightweight multi-object tracker (ByteTrack-inspired)."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        max_distance: float = 80.0,
        track_buffer: int = 50,
    ):
        """
        Args:
            max_age: Remove track after this many frames without match
            min_hits: Hits before track is considered confirmed
            max_distance: Max pixel distance for matching (IoU or center)
            track_buffer: Max center history to keep
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.track_buffer = track_buffer

        self.tracks: Dict[int, Track] = {}
        self.next_id = 0
        self.frame_count = 0

    def reset(self):
        self.tracks.clear()
        self.next_id = 0
        self.frame_count = 0

    def update(self, detections: List[dict]) -> List[dict]:
        """
        Update tracker with new detections.

        Args:
            detections: list of dicts with keys:
                class_id, class_name, confidence, bbox (x1,y1,x2,y2)

        Returns:
            List of tracked objects (detections + track_id + velocity)
        """
        self.frame_count += 1

        if not detections:
            self._age_unmatched([])
            return self._get_confirmed_tracks()

        # Step 1: High-confidence detections -> match first
        high_conf = [d for d in detections if d["confidence"] >= 0.5]
        low_conf = [d for d in detections if d["confidence"] < 0.5]

        matched_ids = set()

        # Match high confidence
        if high_conf:
            matches = self._match_tracks(high_conf)
            for det_idx, track_id in matches:
                matched_ids.add(det_idx)
                self._update_track(track_id, high_conf[det_idx])

        # Match low confidence (only to unmatched tracks)
        if low_conf:
            matches = self._match_tracks(low_conf, exclude_ids=matched_ids)
            for det_idx, track_id in matches:
                self._update_track(track_id, low_conf[det_idx])

        # Create new tracks for unmatched high-conf detections
        for i, det in enumerate(high_conf):
            if i not in matched_ids:
                self._create_track(det)

        # Age unmatched tracks
        all_matched_track_ids = set(m[1] for m in self._match_tracks(high_conf + low_conf))
        for tid in list(self.tracks.keys()):
            if tid not in all_matched_track_ids:
                self.tracks[tid].time_since_update += 1
                self.tracks[tid].age += 1

        # Remove dead tracks
        self._cleanup()

        return self._get_confirmed_tracks()

    def _match_tracks(
        self, detections: List[dict], exclude_ids: set = None
    ) -> List[Tuple[int, int]]:
        """Match detections to existing tracks using center distance."""
        if not self.tracks or not detections:
            return []

        # Build cost matrix
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

                # Center distance
                dist = np.sqrt(
                    (det_center[0] - track_center[0]) ** 2 +
                    (det_center[1] - track_center[1]) ** 2
                )

                # Penalize class mismatch
                if det["class_id"] != track.class_id:
                    dist += 500

                cost_matrix[i, j] = dist

        # Greedy matching (fast, good enough for real-time)
        matches = []
        used_dets = set()
        used_tracks = set()

        # Sort by cost
        indices = np.argsort(cost_matrix.ravel())

        for flat_idx in indices:
            det_idx = flat_idx // len(track_ids)
            track_idx = flat_idx % len(track_ids)

            if det_idx in used_dets or track_idx in used_tracks:
                continue

            if cost_matrix[det_idx, track_idx] > self.max_distance:
                break

            matches.append((det_idx, track_ids[track_idx]))
            used_dets.add(det_idx)
            used_tracks.add(track_idx)

        return matches

    def _create_track(self, det: dict):
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
        track = self.tracks[track_id]
        old_center = track.center
        new_center = self._bbox_center(det["bbox"])

        # Update velocity (EMA)
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

    def _age_unmatched(self, _):
        for tid in self.tracks:
            self.tracks[tid].time_since_update += 1
            self.tracks[tid].age += 1

    def _cleanup(self):
        to_remove = [
            tid for tid, t in self.tracks.items()
            if t.time_since_update > self.max_age
        ]
        for tid in to_remove:
            del self.tracks[tid]

    def _get_confirmed_tracks(self) -> List[dict]:
        """Return confirmed tracks as detection dicts with track_id."""
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
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_track_count(self) -> dict:
        """Get count of unique vehicles by class."""
        counts = defaultdict(int)
        for t in self.tracks.values():
            if t.hits >= self.min_hits:
                counts[t.class_name] += 1
        return dict(counts)
