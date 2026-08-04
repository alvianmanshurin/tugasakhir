"""Vehicle Counter Module for tracking and counting vehicles crossing a line"""

import numpy as np
from collections import defaultdict


class VehicleCounter:
    """Count vehicles crossing a defined line in the video frame."""

    def __init__(
        self,
        line_position=0.5,
        direction="both",
        min_track_length=5,
        max_lost_frames=30,
    ):
        """
        Initialize the vehicle counter.

        Args:
            line_position: Position of counting line (0.0 to 1.0, ratio of frame height)
            direction: Counting direction - "up", "down", or "both"
            min_track_length: Minimum frames a track must exist to be counted
            max_lost_frames: Maximum frames to keep lost tracks
        """
        self.line_position = line_position
        self.direction = direction
        self.min_track_length = min_track_length
        self.max_lost_frames = max_lost_frames

        # Track management
        self.tracks = {}
        self.next_track_id = 0
        self.counted_ids = set()

        # Counters
        self.total_count = 0
        self.class_counts = defaultdict(int)
        self.counts_up = defaultdict(int)
        self.counts_down = defaultdict(int)

    def reset(self):
        """Reset all counters and tracks."""
        self.tracks = {}
        self.next_track_id = 0
        self.counted_ids = set()
        self.total_count = 0
        self.class_counts = defaultdict(int)
        self.counts_up = defaultdict(int)
        self.counts_down = defaultdict(int)

    def _get_center(self, bbox):
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _check_line_crossing(self, prev_center, curr_center, frame_height):
        """Check if the object crossed the counting line."""
        line_y = frame_height * self.line_position

        prev_above = prev_center[1] < line_y
        curr_above = curr_center[1] < line_y

        if self.direction == "up":
            return prev_above and not curr_above
        elif self.direction == "down":
            return not prev_above and curr_above
        else:  # both
            return prev_above != curr_above

    def update(self, detections, frame_height=None):
        """
        Update counter with new detections.

        Args:
            detections: List of dicts with keys: class_id, class_name, confidence, bbox
            frame_height: Height of the frame (needed for line crossing detection)
        """
        # If no frame_height provided, estimate from bboxes
        if frame_height is None and detections:
            max_y = max(d["bbox"][3] for d in detections)
            frame_height = max_y * 1.5  # Estimate
        elif frame_height is None:
            return

        current_ids = set()

        for det in detections:
            bbox = det["bbox"]
            class_name = det["class_name"]
            center = self._get_center(bbox)

            # Find matching existing track or create new
            track_id = self._find_matching_track(center)

            if track_id is None:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    "centers": [center],
                    "class_name": class_name,
                    "bbox": bbox,
                    "lost_frames": 0,
                }
            else:
                # Update existing track
                track = self.tracks[track_id]
                prev_center = track["centers"][-1] if track["centers"] else center

                # Check line crossing
                if (
                    track_id not in self.counted_ids
                    and len(track["centers"]) >= self.min_track_length
                ):
                    if self._check_line_crossing(prev_center, center, frame_height):
                        self.counted_ids.add(track_id)
                        self.total_count += 1
                        self.class_counts[class_name] += 1

                        # Direction counts
                        line_y = frame_height * self.line_position
                        if center[1] < prev_center[1]:
                            self.counts_up[class_name] += 1
                        else:
                            self.counts_down[class_name] += 1

                # Update track
                track["centers"].append(center)
                track["bbox"] = bbox
                track["class_name"] = class_name
                track["lost_frames"] = 0

                # Keep limited history
                if len(track["centers"]) > 50:
                    track["centers"] = track["centers"][-50:]

            current_ids.add(track_id)

        # Update lost frames for unmatched tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in current_ids:
                self.tracks[track_id]["lost_frames"] += 1

                # Remove lost tracks
                if self.tracks[track_id]["lost_frames"] > self.max_lost_frames:
                    del self.tracks[track_id]

    def _find_matching_track(self, center, max_distance=80):
        """Find the closest existing track to the given center."""
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
        """Get a summary of all counts."""
        return {
            "total": self.total_count,
            "by_class": dict(self.class_counts),
            "by_direction": {
                "up": dict(self.counts_up),
                "down": dict(self.counts_down),
            },
            "active_tracks": len(self.tracks),
        }
