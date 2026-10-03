"""Modul ROI (Region of Interest) untuk pemfilteran area jalan.

Koordinat boundary di config ditulis pada satu resolusi referensi
(1920x1080 untuk kamera gerbang ITERA) lalu otomatis diskalakan ke
resolusi frame yang sedang diproses. Tanpa penskalaan ini, ROI hanya
benar pada 1080p dan akan membuang/menyisakan hampir seluruh frame
pada resolusi lain.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

DEFAULT_REFERENCE_W = 1920
DEFAULT_REFERENCE_H = 1080


@dataclass
class ROIBoundary:
    """
    Mendefinisikan ROI sebagai trapesium pada area jalan.

    Koordinat dalam ``reference_resolution``; ROIFilter yang
    menormalisasikannya ke ukuran frame.
    """
    top_left: Tuple[int, int] = (63, 460)        # Kiri atas
    top_right: Tuple[int, int] = (372, 440)      # Kanan atas
    bottom_left: Tuple[int, int] = (122, 1002)   # Kiri bawah
    bottom_right: Tuple[int, int] = (1143, 796)  # Kanan bawah
    ref_width: int = DEFAULT_REFERENCE_W
    ref_height: int = DEFAULT_REFERENCE_H

    def get_polygon(self) -> np.ndarray:
        """Poligon pada resolusi referensi."""
        return np.array([
            self.top_left, self.top_right,
            self.bottom_right, self.bottom_left
        ], dtype=np.float32)

    def polygon_for(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Poligon yang sudah diskalakan ke ukuran frame tertentu."""
        poly = self.get_polygon().copy()
        poly[:, 0] *= frame_width / float(self.ref_width)
        poly[:, 1] *= frame_height / float(self.ref_height)
        return poly.astype(np.int32)

    @classmethod
    def from_config(cls, boundary_cfg: dict, reference: Optional[dict] = None) -> "ROIBoundary":
        """Bangun boundary dari blok ``roi`` di config.yaml."""
        reference = reference or {}
        return cls(
            top_left=tuple(boundary_cfg.get("top_left", (63, 460))),
            top_right=tuple(boundary_cfg.get("top_right", (372, 440))),
            bottom_left=tuple(boundary_cfg.get("bottom_left", (122, 1002))),
            bottom_right=tuple(boundary_cfg.get("bottom_right", (1143, 796))),
            ref_width=int(reference.get("width", DEFAULT_REFERENCE_W)),
            ref_height=int(reference.get("height", DEFAULT_REFERENCE_H)),
        )


@dataclass
class ROIConfig:
    """
    Konfigurasi lengkap untuk ROI.

    Args:
        enabled: aktifkan pemfilteran ROI
        boundary: titik trapesium + resolusi referensi
        min_bbox_height: tinggi bbox minimum dalam pixel frame
    """
    enabled: bool = True
    boundary: ROIBoundary = field(default_factory=ROIBoundary)
    min_bbox_height: int = 20


class ROIFilter:
    """
    Filter deteksi berdasarkan Region of Interest (area jalan).

    Fitur:
    - Pemeriksaan apakah pusat bbox berada di dalam poligon ROI
    - Filtering berdasarkan ukuran bbox
    - Penskalaan otomatis poligon ke resolusi frame
    - Gambar visualisasi ROI pada frame

    Poligon yang sudah diskalakan di-cache per ukuran frame, jadi
    ``filter_detections`` tidak menghitung ulang tiap deteksi.
    """

    def __init__(self, config: Optional[ROIConfig] = None):
        self.config = config or ROIConfig()
        self.boundary = self.config.boundary
        self._cache_key: Optional[Tuple[int, int]] = None
        self.polygon = self.boundary.polygon_for(self.boundary.ref_width,
                                                 self.boundary.ref_height)

    def _ensure_polygon(self, frame_shape) -> np.ndarray:
        """Pastikan ``self.polygon`` sesuai ukuran frame ini."""
        height, width = int(frame_shape[0]), int(frame_shape[1])
        key = (width, height)
        if key != self._cache_key:
            self.polygon = self.boundary.polygon_for(width, height)
            self._cache_key = key
        return self.polygon

    def is_inside_roi(self, bbox: List[float], frame_shape: tuple) -> bool:
        """
        True jika pusat bbox berada di dalam poligon ROI.

        Args:
            bbox: koordinat bounding box [x1, y1, x2, y2] dalam pixel frame
            frame_shape: bentuk frame (height, width, channels)
        """
        if not self.config.enabled:
            return True

        polygon = self._ensure_polygon(frame_shape)

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0

    def filter_detections(self, detections: List[dict],
                          frame_shape: tuple) -> List[dict]:
        """
        Saring deteksi: hanya yang di dalam ROI dan cukup besar.

        Args:
            detections: daftar dict deteksi dengan key bbox, class_id, dll
            frame_shape: bentuk frame (height, width, channels)
        """
        if not self.config.enabled:
            return detections

        polygon = self._ensure_polygon(frame_shape)
        min_h = self.config.min_bbox_height

        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            if (y2 - y1) < min_h:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) < 0:
                continue

            filtered.append(det)

        return filtered

    def coverage_ratio(self, frame_shape: tuple) -> float:
        """
        Berapa persen frame yang tertutup ROI.

        Berguna sebagai sanity check: kalau nilainya 0% atau 100%+
        (poligon keluar frame) berarti koordinat ROI tidak sesuai dengan
        video yang dipakai.
        """
        polygon = self._ensure_polygon(frame_shape)
        area = abs(float(cv2.contourArea(polygon)))
        frame_area = float(frame_shape[0] * frame_shape[1])
        return area / frame_area if frame_area else 0.0

    def draw_roi(self, frame: np.ndarray, color=(0, 255, 255),
                 thickness=2) -> np.ndarray:
        """Gambar batas ROI semi-transparan pada frame."""
        result = frame.copy()
        if not self.config.enabled:
            return result

        polygon = self._ensure_polygon(frame.shape)

        overlay = result.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, 0.15, result, 0.85, 0, result)

        cv2.polylines(result, [polygon], True, color, thickness)

        return result
