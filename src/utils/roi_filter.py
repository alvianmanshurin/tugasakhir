"""Modul ROI (Region of Interest) untuk pemfilteran area jalan"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ROIBoundary:
    """
    Mendefinisikan ROI sebagai trapesium pada area jalan.
    
    Koordinat disesuaikan dengan sudut pandang kamera gerbang ITERA.
    """
    # Titik sumber (trapesium pada frame)
    top_left: Tuple[int, int] = (60, 497)       # Kiri atas
    top_right: Tuple[int, int] = (390, 484)     # Kanan atas
    bottom_left: Tuple[int, int] = (111, 1007)  # Kiri bawah
    bottom_right: Tuple[int, int] = (979, 822)  # Kanan bawah

    def get_polygon(self) -> np.ndarray:
        """Membuat array numpy dari koordinat trapesium untuk digunakan cv2.pointPolygonTest."""
        return np.array([
            self.top_left, self.top_right,
            self.bottom_right, self.bottom_left
        ], dtype=np.int32)


@dataclass
class ROIConfig:
    """
    Konfigurasi lengkap untuk ROI.
    
    Berisi parameter untuk:
    - Status aktif/tidaknya ROI
    - Batas area ROI (trapesium)
    - Ambang confidence minimum
    - Ukuran bounding box minimum
    """
    enabled: bool = True
    boundary: ROIBoundary = field(default_factory=ROIBoundary)
    # Ambang confidence untuk pemfilteran ROI
    min_confidence: float = 0.35
    # Tinggi bounding box minimum (pixel) agar dianggap valid
    min_bbox_height: int = 20


class ROIFilter:
    """
    Kelas untuk memfilter deteksi berdasarkan Region of Interest (area jalan).
    
    Fitur:
    - Pemeriksaan apakah pusat bbox berada di dalam poligon ROI
    - Filtering berdasarkan ukuran bbox
    - Gambar visualisasi ROI pada frame
    """

    def __init__(self, config: ROIConfig = None):
        """
        Inisialisasi ROIFilter dengan konfigurasi yang diberikan.
        
        Args:
            config: objek ROIConfig (jika None, menggunakan default)
        """
        self.config = config or ROIConfig()
        self.boundary = self.config.boundary
        self.polygon = self.boundary.get_polygon()

    def is_inside_roi(self, bbox: List[float], frame_shape: tuple) -> bool:
        """
        Mengecek apakah pusat bounding box berada di dalam poligon ROI.
        
        Menggunakan fungsi cv2.pointPolygonTest untuk pengecekan titik dalam poligon.
        
        Args:
            bbox: koordinat bounding box [x1, y1, x2, y2]
            frame_shape: bentuk frame (height, width, channels)
            
        Returns:
            True jika pusat bbox di dalam ROI, False jika di luar
        """
        if not self.config.enabled:
            return True

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2  # Pusat x
        cy = (y1 + y2) / 2  # Pusat y

        # Pemeriksaan titik dalam poligon menggunakan OpenCV
        result = cv2.pointPolygonTest(
            self.polygon, (float(cx), float(cy)), False
        )
        return result >= 0

    def filter_detections(
        self, detections: List[dict], frame_shape: tuple
    ) -> List[dict]:
        """
        Memfilter deteksi: hanya menyimpan yang berada di dalam ROI.
        
        Proses filtering:
        1. Cek tinggi bounding box minimum
        2. Cek apakah pusat bbox di dalam poligon ROI
        
        Args:
            detections: daftar dict deteksi dengan key bbox, class_id, dll
            frame_shape: bentuk frame (height, width, channels)
            
        Returns:
            Daftar deteksi yang sudah difilter
        """
        if not self.config.enabled:
            return detections

        filtered = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox

            # Cek tinggi minimum bounding box
            bbox_h = y2 - y1
            if bbox_h < self.config.min_bbox_height:
                continue

            # Cek apakah pusat bbox di dalam ROI
            if not self.is_inside_roi(bbox, frame_shape):
                continue

            filtered.append(det)

        return filtered

    def draw_roi(self, frame: np.ndarray, color=(0, 255, 255), thickness=2) -> np.ndarray:
        """
        Menggambar batas ROI pada frame.
        
        Fitur yang digambar:
        - Poligon ROI semi-transparan
        - Garis batas ROI
        
        Args:
            frame: frame gambar asli
            color: warna garis ROI (default: kuning)
            thickness: ketebalan garis
            
        Returns:
            Frame dengan gambar ROI
        """
        result = frame.copy()
        if not self.config.enabled:
            return result

        # Menggambar poligon ROI semi-transparan
        overlay = result.copy()
        cv2.fillPoly(overlay, [self.polygon], color)
        cv2.addWeighted(overlay, 0.15, result, 0.85, 0, result)

        # Menggambar garis batas ROI
        cv2.polylines(result, [self.polygon], True, color, thickness)

        return result
