"""Modul ROI (Region of Interest) untuk pemfilteran batas jalan 20 meter"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ROIBoundary:
    """
    Mendefinisikan ROI sebagai trapesium yang mewakili perspektif jalan 20 meter.
    
    Koordinat disesuaikan dengan sudut pandang kamera gerbang ITERA.
    Bagian bawah frame = posisi kamera (0m)
    Bagian atas trapesium = 20 meter dari kamera
    """
    # Titik sumber (trapesium pada frame - pengguna menyesuaikan ini)
    top_left: Tuple[int, int] = (150, 120)      # Titik kiri atas
    top_right: Tuple[int, int] = (490, 120)     # Titik kanan atas
    bottom_left: Tuple[int, int] = (0, 416)     # Titik kiri bawah
    bottom_right: Tuple[int, int] = (640, 416)  # Titik kanan bawah

    # Pemetaan jarak (pixel -> meter, perkiraan)
    # bawah = posisi kamera (0m), atas = jarak 20m
    max_distance_m: float = 20.0

    def get_polygon(self) -> np.ndarray:
        """Membuat array numpy dari koordinat trapesium untuk digunakan cv2.pointPolygonTest."""
        return np.array([
            self.top_left, self.top_right,
            self.bottom_right, self.bottom_left
        ], dtype=np.int32)

    def pixel_to_distance(self, y_pixel: int, frame_height: int) -> float:
        """
        Mengestimasi jarak dalam meter dari posisi y-pixel.
        
        Logika:
        - Bagian bawah ROI = 0m (dekat kamera)
        - Bagian atas ROI = max_distance_m (20m)
        - Menggunakan pemetaan linear (cukup akurat untuk jalan datar)
        
        Args:
            y_pixel: posisi y pixel yang akan dikonversi
            frame_height: tinggi frame gambar
            
        Returns:
            Jarak dalam meter
        """
        roi_top = self.top_left[1]
        roi_bottom = self.bottom_left[1]

        # Jika pixel di atas ROI, kembalikan jarak maksimum
        if y_pixel < roi_top:
            return self.max_distance_m
        # Jika pixel di bawah ROI, kembalikan 0 meter
        if y_pixel > roi_bottom:
            return 0.0

        # Pemetaan linear dari pixel ke meter
        ratio = (roi_bottom - y_pixel) / (roi_bottom - roi_top)
        return ratio * self.max_distance_m


@dataclass
class ROIConfig:
    """
    Konfigurasi lengkap untuk ROI.
    
    Berisi parameter untuk:
    - Status aktif/tidaknya ROI
    - Batas area ROI (trapesium)
    - Jarak maksimum yang dideteksi
    - Ambang confidence minimum
    - Ukuran bounding box minimum
    """
    enabled: bool = True
    boundary: ROIBoundary = field(default_factory=ROIBoundary)
    max_distance_m: float = 20.0
    # Jarak pixel dari bawah frame ke bawah ROI (0 = tepi frame)
    bottom_offset: int = 0
    # Ambang confidence untuk pemfilteran ROI
    min_confidence: float = 0.35
    # Tinggi bounding box minimum (pixel) agar dianggap valid
    min_bbox_height: int = 20


class ROIFilter:
    """
    Kelas untuk memfilter deteksi berdasarkan Region of Interest (area jalan).
    
    Fitur:
    - Pemeriksaan apakah pusat bbox berada di dalam poligon ROI
    - Estimasi jarak objek dari kamera (meter)
    - Filtering berdasarkan jarak dan ukuran bbox
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

    def get_distance_m(self, bbox: List[float], frame_shape: tuple) -> float:
        """
        Mendapatkan estimasi jarak dalam meter untuk sebuah deteksi.
        
        Args:
            bbox: koordinat bounding box [x1, y1, x2, y2]
            frame_shape: bentuk frame (height, width, channels)
            
        Returns:
            Jarak dalam meter
        """
        x1, y1, x2, y2 = bbox
        cy = (y1 + y2) / 2  # Pusat y bbox
        return self.boundary.pixel_to_distance(cy, frame_shape[0])

    def filter_detections(
        self, detections: List[dict], frame_shape: tuple
    ) -> List[dict]:
        """
        Memfilter deteksi: hanya menyimpan yang berada di dalam ROI dan dalam jarak.
        
        Proses filtering:
        1. Cek tinggi bounding box minimum
        2. Cek apakah pusat bbox di dalam poligon ROI
        3. Cek apakah jarak objek <= max_distance_m
        
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

            # Cek jarak objek
            dist = self.get_distance_m(bbox, frame_shape)
            if dist > self.config.max_distance_m:
                continue

            # Tambahkan info jarak ke deteksi
            det["distance_m"] = round(dist, 1)
            filtered.append(det)

        return filtered

    def draw_roi(self, frame: np.ndarray, color=(0, 255, 255), thickness=2) -> np.ndarray:
        """
        Menggambar batas ROI pada frame.
        
        Fitur yang digambar:
        - Poligon ROI semi-transparan
        - Garis batas ROI
        - Label jarak (0m, 5m, 10m, 15m, 20m)
        
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

        # Menggambar label jarak
        roi_top = self.boundary.top_left[1]
        roi_bottom = self.boundary.bottom_left[1]
        mid_x = (self.boundary.top_left[0] + self.boundary.top_right[0]) // 2

        # Menampilkan label jarak setiap 5 meter
        for dist_m in [0, 5, 10, 15, 20]:
            ratio = dist_m / self.boundary.max_distance_m
            y_pos = int(roi_bottom - ratio * (roi_bottom - roi_top))
            label = f"{dist_m}m"
            cv2.putText(result, label, (mid_x + 5, y_pos + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return result

    def draw_distance_tags(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Menggambar tag jarak pada setiap deteksi.
        
        Args:
            frame: frame gambar asli
            detections: daftar deteksi yang sudah memiliki key distance_m
            
        Returns:
            Frame dengan tag jarak pada setiap objek
        """
        result = frame.copy()
        for det in detections:
            if "distance_m" not in det:
                continue
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            dist = det["distance_m"]
            label = f"{dist}m"
            # Menampilkan label jarak di sebelah kanan atas bbox
            cv2.putText(result, label, (x2 + 5, y1 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return result
