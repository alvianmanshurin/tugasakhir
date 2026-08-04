"""ROI (Region of Interest) Module for 20m road boundary filtering"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ROIBoundary:
    """Define ROI as a trapezoid representing 20m road perspective."""
    # Source points (trapezoid on frame - user adjusts these)
    top_left: Tuple[int, int] = (150, 120)
    top_right: Tuple[int, int] = (490, 120)
    bottom_left: Tuple[int, int] = (0, 416)
    bottom_right: Tuple[int, int] = (640, 416)

    # Distance mapping (pixels -> meters, approximate)
    # bottom = camera position (0m), top = 20m away
    max_distance_m: float = 20.0

    def get_polygon(self) -> np.ndarray:
        return np.array([
            self.top_left, self.top_right,
            self.bottom_right, self.bottom_left
        ], dtype=np.int32)

    def pixel_to_distance(self, y_pixel: int, frame_height: int) -> float:
        """Estimate distance in meters from y-pixel position.
        Bottom of ROI = 0m (near camera), top of ROI = max_distance_m.
        Uses linear mapping (good enough for flat road)."""
        roi_top = self.top_left[1]
        roi_bottom = self.bottom_left[1]

        if y_pixel < roi_top:
            return self.max_distance_m
        if y_pixel > roi_bottom:
            return 0.0

        ratio = (roi_bottom - y_pixel) / (roi_bottom - roi_top)
        return ratio * self.max_distance_m


@dataclass
class ROIConfig:
    """Full ROI configuration."""
    enabled: bool = True
    boundary: ROIBoundary = field(default_factory=ROIBoundary)
    max_distance_m: float = 20.0
    # How many pixels from frame bottom to ROI bottom (0 = frame edge)
    bottom_offset: int = 0
    # Confidence threshold for ROI filtering
    min_confidence: float = 0.35
    # Minimum bounding box height (pixels) to be considered valid
    min_bbox_height: int = 20


class ROIFilter:
    """Filter detections based on Region of Interest (road area)."""

    def __init__(self, config: ROIConfig = None):
        self.config = config or ROIConfig()
        self.boundary = self.config.boundary
        self.polygon = self.boundary.get_polygon()

    def is_inside_roi(self, bbox: List[float], frame_shape: tuple) -> bool:
        """Check if bounding box center is inside the ROI polygon."""
        if not self.config.enabled:
            return True

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Check point-in-polygon
        result = cv2.pointPolygonTest(
            self.polygon, (float(cx), float(cy)), False
        )
        return result >= 0

    def get_distance_m(self, bbox: List[float], frame_shape: tuple) -> float:
        """Get estimated distance in meters for a detection."""
        x1, y1, x2, y2 = bbox
        cy = (y1 + y2) / 2
        return self.boundary.pixel_to_distance(cy, frame_shape[0])

    def filter_detections(
        self, detections: List[dict], frame_shape: tuple
    ) -> List[dict]:
        """Filter detections: only keep those inside ROI and within distance."""
        if not self.config.enabled:
            return detections

        filtered = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox

            # Check minimum bbox height
            bbox_h = y2 - y1
            if bbox_h < self.config.min_bbox_height:
                continue

            # Check inside ROI polygon
            if not self.is_inside_roi(bbox, frame_shape):
                continue

            # Check distance
            dist = self.get_distance_m(bbox, frame_shape)
            if dist > self.config.max_distance_m:
                continue

            # Add distance info to detection
            det["distance_m"] = round(dist, 1)
            filtered.append(det)

        return filtered

    def draw_roi(self, frame: np.ndarray, color=(0, 255, 255), thickness=2) -> np.ndarray:
        """Draw ROI boundary on frame."""
        result = frame.copy()
        if not self.config.enabled:
            return result

        # Draw filled polygon (semi-transparent)
        overlay = result.copy()
        cv2.fillPoly(overlay, [self.polygon], color)
        cv2.addWeighted(overlay, 0.15, result, 0.85, 0, result)

        # Draw boundary lines
        cv2.polylines(result, [self.polygon], True, color, thickness)

        # Draw distance labels
        roi_top = self.boundary.top_left[1]
        roi_bottom = self.boundary.bottom_left[1]
        mid_x = (self.boundary.top_left[0] + self.boundary.top_right[0]) // 2

        for dist_m in [0, 5, 10, 15, 20]:
            ratio = dist_m / self.boundary.max_distance_m
            y_pos = int(roi_bottom - ratio * (roi_bottom - roi_top))
            label = f"{dist_m}m"
            cv2.putText(result, label, (mid_x + 5, y_pos + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return result

    def draw_distance_tags(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw distance tag on each detection."""
        result = frame.copy()
        for det in detections:
            if "distance_m" not in det:
                continue
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            dist = det["distance_m"]
            label = f"{dist}m"
            cv2.putText(result, label, (x2 + 5, y1 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return result
