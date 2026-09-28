"""
Pipeline lengkap untuk deteksi + tracking + penghitungan kendaraan.

Ini adalah SATU-SATUNYA tempat logika pipeline berada. Modul lain
(``detect_with_tracking.py``, GUI, CCTV) memanggil kelas ini, bukan
menyalin logikanya sendiri - supaya tidak ada lagi dua tracker dengan dua
ruang ID berbeda di satu program.

Urutan per frame:
    1. model.predict()  - YOLO11n, letterbox ke ``imgsz`` (tanpa resize distort)
    2. ROIFilter        - buang deteksi di luar area jalan
    3. ObjectTracker    - SATU tracker, menghasilkan ``track_id``
    4. VehicleCounter   - dual-line, memakai ``track_id`` dari langkah 3
"""

import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO

from utils.paths import (
    ensure_src_on_path,
    get_class_names,
    get_model_path,
    load_config,
)
from utils.roi_filter import ROIBoundary, ROIConfig, ROIFilter
from utils.tracker import ObjectTracker
from utils.counter import VehicleCounter

ensure_src_on_path()

DEFAULT_COLORS = {
    "motor": (255, 0, 0),
    "mobil": (0, 255, 0),
    "bus": (0, 0, 255),
    "truk": (255, 255, 0),
}
DEFAULT_COLOR = (200, 200, 200)


class VehiclePipeline:
    """Pipeline lengkap deteksi kendaraan: deteksi -> ROI -> track -> hitung."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.config = config if config is not None else load_config()
        model_cfg = self.config.get("model", {})

        self.model_path = get_model_path(self.config, explicit=model_path)
        self.device = device or model_cfg.get("device", "cpu")
        self.imgsz = int(model_cfg.get("input_size", 416))
        self.iou_threshold = float(model_cfg.get("iou_threshold", 0.45))
        self.conf_threshold = float(model_cfg.get("confidence_threshold", 0.5))
        # Inферен harus dijalankan pada ambang paling rendah yang dipakai
        # tracker; kalau tidak, deteksi 0.25-0.50 tidak pernah sampai ke
        # ObjectTracker dan rescue track yang hilang tidak pernah terjadi.
        self.low_conf_threshold = float(
            model_cfg.get("low_conf_threshold", min(0.25, self.conf_threshold))
        )
        if self.low_conf_threshold > self.conf_threshold:
            self.low_conf_threshold = self.conf_threshold

        self.class_names = get_class_names(self.config)
        self.class_colors: Dict[str, tuple] = {}
        for name in self.class_names.values():
            self.class_colors[name] = DEFAULT_COLORS.get(name, DEFAULT_COLOR)
        for name, color in (self.config.get("visualization", {})
                            .get("color_map") or {}).items():
            self.class_colors[str(name)] = tuple(color)

        self._allowed_classes = self.config.get("detection", {}).get("classes")
        if self._allowed_classes is not None:
            self._allowed_classes = {int(c) for c in self._allowed_classes}

        self.model = YOLO(self.model_path)

        # ROI
        roi_cfg = self.config.get("roi", {})
        reference = roi_cfg.get("reference_resolution", {}) or {}
        boundary = ROIBoundary.from_config(
            roi_cfg.get("boundary", {}) or {}, reference=reference
        )
        self.roi_filter = ROIFilter(
            ROIConfig(
                enabled=bool(roi_cfg.get("enabled", True)),
                boundary=boundary,
                min_bbox_height=int(roi_cfg.get("min_bbox_height", 20)),
            )
        )

        # Tracker
        track_cfg = self.config.get("tracking", {})
        self.tracker = ObjectTracker(
            max_age=int(track_cfg.get("max_age", 30)),
            min_hits=int(track_cfg.get("min_hits", 3)),
            max_distance=float(track_cfg.get("max_distance", 80.0)),
            low_conf_threshold=self.low_conf_threshold,
        )

        # Counter
        count_cfg = self.config.get("counting", {})
        self.counter = VehicleCounter(
            line1_position=float(count_cfg.get("line1_position", 0.48)),
            line2_position=float(count_cfg.get("line2_position", 0.75)),
            direction=str(count_cfg.get("direction", "both")),
            min_track_length=int(count_cfg.get("min_track_length", 3)),
            min_displacement=float(count_cfg.get("min_displacement", 25.0)),
        )

        self.frame_number = 0
        self.total_detections = 0
        # Baris track per frame. Hanya bertambah lewat process_frame();
        # process_image() sengaja TIDAK menyentuh karena tidak ada tracking
        # di sana (lihat catatan di method itu).
        self.total_tracked = 0
        # Deteksi setelah ROI, terpisah dari total_tracked supaya ringkasan
        # mode batch tidak melaporkan angka yang menyesatkan.
        self.total_after_roi = 0
        self.fps = 0.0
        self._last_timestamp: Optional[float] = None
        self._fps_history: List[float] = []

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset tracker, counter, dan counter internal. Untuk stream baru."""
        self.tracker.reset()
        self.counter.reset()
        self.frame_number = 0
        self.total_detections = 0
        self.total_tracked = 0
        self.total_after_roi = 0
        self.fps = 0.0
        self._last_timestamp = None
        self._fps_history.clear()

    # ------------------------------------------------------------------
    # Deteksi
    # ------------------------------------------------------------------

    def detect(self, frame) -> List[dict]:
        """
        Deteksi mentah dari model, TANPA ROI dan TANPA tracking.

        ``imgsz`` dipakai apa adanya supaya Ultralytics melakukan letterbox
        (pad + resize proporsional), bukan resize distort ke 416x416 yang
        membuat kendaraan kecil jadi tidak terbaca.
        """
        results = self.model.predict(
            frame,
            conf=self.low_conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[dict] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.class_names:
                    # kelas di luar mapping config (mis. "person" dari model
                    # yang salah dimuat) - buang, jangan dilabeli diam-diam.
                    continue
                if self._allowed_classes is not None and cls_id not in self._allowed_classes:
                    continue
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.class_names[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": [float(c) for c in box.xyxy[0].tolist()],
                })
        return detections

    # ------------------------------------------------------------------
    # Gambar tunggal (tanpa tracking)
    # ------------------------------------------------------------------

    def process_image(self, frame) -> Tuple[Any, List[dict], dict]:
        """
        Proses SATU gambar: deteksi + ROI + gambar, tanpa tracking.

        Batch processing gambar tidak boleh lewat ``process_frame()``.
        ``ObjectTracker`` hanya mengonfirmasi track setelah ``min_hits``
        frame berturut-turut (default 3), jadi satu gambar tidak pernah
        menghasilkan track sama sekali - batch akan menulis 0 box ke
        semua gambar padahal modelnya mendeteksi kendaraan.

        Karena itu di sini tracker dilewati, dan tiap deteksi diberi
        ``track_id`` sementara (``-1``) supaya bisa memakai fungsi
        ``draw()`` yang sama. Menghitung kendaraan dari satu gambar
        tidak bermakna, jadi counter tidak disentuh.

        ``total_tracked`` juga tidak ditambah. Dulu variabel itu naik di sini,
        sehingga ringkasan mode batch melaporkan "baris track" padahal tidak
        ada tracking sama sekali - angkanya sama dengan jumlah deteksi, dan
        bisa salah dibaca sebagai jumlah kendaraan. Yang dijumlahkan di sini
        hanya ``total_after_roi``. ``frame_number`` tetap naik supaya
        ``total_frames`` pada ringkasan mencerminkan banyaknya gambar yang
        diproses, bukan 0.

        Returns:
            (gambar_teranotasi, deteksi, info)
        """
        if frame is None or not getattr(frame, "size", 0):
            raise ValueError("frame kosong")

        self.frame_number += 1
        raw = self.detect(frame)
        filtered = self.roi_filter.filter_detections(raw, frame.shape)

        for d in filtered:
            d["track_id"] = -1
            d["counted"] = False
            d["direction"] = None

        self.total_detections += len(raw)
        self.total_after_roi += len(filtered)

        info = {
            "detections_count": len(raw),
            "after_roi_count": len(filtered),
            "tracked_count": 0,
            "counted_count": 0,
        }
        annotated = self.draw(frame, filtered, draw_roi=bool(
            self.config.get("roi", {}).get("draw_roi", True)))
        return annotated, filtered, info

    # ------------------------------------------------------------------
    # Frame penuh
    # ------------------------------------------------------------------

    def process_frame(
        self, frame, timestamp: Optional[float] = None
    ) -> Tuple[Any, List[dict], dict]:
        """
        Proses satu frame.

        Returns:
            (frame_teranotasi, tracked, info)

        ``tracked`` adalah list dict dari ``ObjectTracker.update()``; tiap item
        sudah punya ``track_id``, ``class_name``, ``bbox``, ``counted``, dan
        ``direction``.
        """
        if frame is None or not getattr(frame, "size", 0):
            raise ValueError("frame kosong")

        self.frame_number += 1
        frame_height = frame.shape[0]

        raw = self.detect(frame)
        raw_count = len(raw)

        filtered = self.roi_filter.filter_detections(raw, frame.shape)
        after_roi = len(filtered)

        tracked = self.tracker.update(filtered)
        self.counter.update(tracked, frame_height)

        for t in tracked:
            t["counted"] = self.counter.is_counted(t["track_id"])
            t["direction"] = self.counter.direction_of(t["track_id"])

        self.total_detections += raw_count
        self.total_tracked += len(tracked)
        self._update_fps(timestamp)

        info = {
            "frame_number": self.frame_number,
            "timestamp": timestamp,
            "detections_count": raw_count,
            "after_roi_count": after_roi,
            "tracked_count": len(tracked),
            "counted_count": sum(1 for t in tracked if t["counted"]),
            "fps": self.fps,
            "total_count": self.counter.total_count,
        }

        annotated = self.draw(frame, tracked, draw_roi=bool(
            self.config.get("roi", {}).get("draw_roi", True)))
        return annotated, tracked, info

    def _update_fps(self, timestamp: Optional[float]) -> None:
        """Hitung FPS eksponensial bergerak dari timestamp frame."""
        if timestamp is None:
            self.fps = 0.0
            return
        if self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if dt > 0:
                instant = 1.0 / dt
                self._fps_history.append(instant)
                if len(self._fps_history) > 30:
                    self._fps_history.pop(0)
                self.fps = sum(self._fps_history) / len(self._fps_history)
        self._last_timestamp = timestamp

    # ------------------------------------------------------------------
    # Visualisasi
    # ------------------------------------------------------------------

    def _color_for(self, class_name: str):
        return self.class_colors.get(class_name, DEFAULT_COLOR)

    def draw(self, frame, tracked: List[dict], draw_roi: bool = True):
        """Gambar ROI, bbox ber-ID, dan HUD hitungan. Frame tidak dimodifikasi."""
        canvas = frame.copy()

        if draw_roi:
            canvas = self.roi_filter.draw_roi(canvas)
        if bool(self.config.get("counting", {}).get("draw_lines", True)):
            self._draw_counting_lines(canvas)

        vis_cfg = self.config.get("visualization", {})
        show_labels = bool(vis_cfg.get("show_labels", True))
        show_id = bool(self.config.get("tracking", {}).get("show_track_id", True))
        font_scale = float(vis_cfg.get("font_scale", 0.6))
        thickness = int(vis_cfg.get("font_thickness", 2))
        line_thickness = int(self.config.get("detection", {}).get("line_thickness", 2))

        for t in tracked:
            x1, y1, x2, y2 = (int(c) for c in t["bbox"])
            class_name = t["class_name"]
            color = self._color_for(class_name)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_thickness)

            if show_labels:
                parts = []
                if show_id:
                    parts.append(f"#{t['track_id']}")
                parts.append(class_name)
                if bool(self.config.get("detection", {}).get("show_conf", True)):
                    parts.append(f"{t['confidence']:.2f}")
                if t.get("counted"):
                    parts.append(f"[{str(t.get('direction') or 'hit').upper()}]")
                label = " ".join(parts)

                # y1-6 bisa keluar frame saat bbox menyentuh tepi atas
                ty = y1 - 6 if y1 - 6 > 12 else y1 + 16
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                              font_scale, thickness)
                ty = max(ty, th + 4)
                cv2.rectangle(canvas, (x1, ty - th - 6), (x1 + tw + 6, ty + 2),
                              color, -1)
                cv2.putText(canvas, label, (x1 + 3, ty - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                            thickness, cv2.LINE_AA)

        self._draw_hud(canvas, tracked)
        return canvas

    def _draw_counting_lines(self, canvas) -> None:
        """
        Gambar dua garis hitung. Posisi memakai FRAME yang sedang diproses,
        bukan koordinat ROI reference, supaya selalu selaras dengan
        VehicleCounter yang juga memakai frame_height.
        """
        h, w = canvas.shape[:2]
        line1_y = int(h * self.counter.line1_position)
        line2_y = int(h * self.counter.line2_position)

        cv2.line(canvas, (0, line1_y), (w, line1_y), (255, 100, 0), 2)
        cv2.putText(canvas, "GARIS 1", (10, max(20, line1_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1, cv2.LINE_AA)

        cv2.line(canvas, (0, line2_y), (w, line2_y), (0, 255, 255), 2)
        cv2.putText(canvas, "GARIS 2", (10, min(h - 10, line2_y + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    def _draw_hud(self, canvas, tracked: List[dict]) -> None:
        vis_cfg = self.config.get("visualization", {})
        summary = self.counter.get_count_summary()

        lines: List[str] = []
        if vis_cfg.get("show_fps", True):
            lines.append(f"FPS: {self.fps:.1f}")
        if vis_cfg.get("show_count", True):
            lines.append(f"TERHITUNG: {self.counter.total_count}")
            for cls, n in sorted(summary["by_class"].items()):
                lines.append(f"  {cls}: {n}")
            for direction in ("down", "up"):
                per_cls = summary["by_direction"].get(direction) or {}
                if per_cls:
                    label = "MASUK" if direction == "down" else "KELUAR"
                    total = sum(per_cls.values())
                    lines.append(f"  {label}: {total}")

        if not lines:
            return

        font_scale = 0.5
        thickness = 1
        pad = 8
        sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                 for t in lines]
        line_h = max(s[1] for s in sizes) + 8
        box_w = max(s[0] for s in sizes) + pad * 2
        box_h = line_h * len(lines) + pad

        h, w = canvas.shape[:2]
        x0, y0 = w - box_w - 10, 10
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        for i, text in enumerate(lines):
            cv2.putText(canvas, text, (x0 + pad, y0 + pad + line_h * (i + 1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Ringkasan
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Ringkasan hitungan run ini, format yang dipakai finalize_session()."""
        return {
            "total_frames": self.frame_number,
            "total_detections": self.total_detections,
            "total_tracked_rows": self.total_tracked,
            "total_after_roi_rows": self.total_after_roi,
            "total_counted": self.counter.total_count,
            "avg_fps": self.fps,
            "by_class": dict(self.counter.get_count_summary()["by_class"]),
            "by_direction": {
                k: dict(v) for k, v in self.counter.get_count_summary()["by_direction"].items()
            },
        }

    def new_counts(self, before_total: int) -> int:
        """Jumlah kendaraan yang terhitung sejak frame terakhir."""
        return self.counter.total_count - before_total

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def process_video(
        self,
        source,
        output_path: Optional[str] = None,
        show: bool = False,
        max_frames: Optional[int] = None,
        save_db: bool = False,
    ):
        """
        Proses video/webcam dan opsional simpan video + database.

        Args:
            source: path video, URL, atau "0" untuk webcam.
            output_path: path video hasil (None = tidak menyimpan).
            show: tampilkan preview jendela.
            max_frames: batas frame (untuk uji cepat).
            save_db: simpan hasil ke database.
        """
        from utils.database import DetectionDatabase

        source = 0 if source in ("0", 0) else source
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Gagal membuka sumber: {source}")

        db: Optional[DetectionDatabase] = None
        session_id = None
        if save_db:
            db = DetectionDatabase()
            kind = "webcam" if source == 0 else "video"
            session_id = db.start_session(f"pipeline-{kind}", kind, str(source))

        writer = None
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if not src_fps or src_fps != src_fps or src_fps <= 1:  # NaN / 0 / sentinel
                src_fps = 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     src_fps, (width, height))

        processed = 0
        wall_start = time.perf_counter()
        try:
            while True:
                if max_frames is not None and processed >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break

                # Pakai jam dinding, bukan POS_MSEC. POS_MSEC mengikuti
                # timeline video, jadi 1/dt hanya mengulang FPS metadata dan
                # tidak pernah menunjukkan kecepatan pemrosesan sebenarnya.
                # Jam dinding juga benar untuk webcam/CCTV (POS_MSEC = 0).
                timestamp = time.perf_counter() - wall_start

                annotated, tracked, info = self.process_frame(frame, timestamp=timestamp)

                if writer is not None:
                    writer.write(annotated)

                if db is not None:
                    db.save_detections_batch(session_id, [
                        {
                            "vehicle_id": f"track_{t['track_id']}",
                            "class_id": t["class_id"],
                            "class_name": t["class_name"],
                            "confidence": t["confidence"],
                            "bbox": t["bbox"],
                            "frame_number": info["frame_number"],
                            "timestamp": info["timestamp"],
                            "direction": t.get("direction"),
                            "counted": bool(t.get("counted")),
                        }
                        for t in tracked
                    ])
                    db.save_frame_stats(
                        session_id,
                        info["frame_number"], info["detections_count"],
                        info["after_roi_count"], info["tracked_count"],
                        info["counted_count"], info["fps"], info["timestamp"],
                    )

                processed += 1
                if show:
                    cv2.imshow("Pipeline", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show:
                cv2.destroyAllWindows()
            if db is not None:
                db.finalize_session(session_id, self.summary())
                db.close()

        return self.summary()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Deteksi + tracking + hitung kendaraan")
    parser.add_argument("--source", default="0", help="path video, URL, atau 0 untuk webcam")
    parser.add_argument("--output", default=None, help="simpan video hasil ke path ini")
    parser.add_argument("--show", action="store_true", help="tampilkan preview")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--db", action="store_true", help="simpan hasil ke database")
    parser.add_argument("--model", default=None, help="override path bobot")
    args = parser.parse_args()

    print("=== PIPELINE DETEKSI KENDARAAN ===")
    pipeline = VehiclePipeline(model_path=args.model)
    print(f"Model : {pipeline.model_path}")
    print(f"Device: {pipeline.device}  imgsz={pipeline.imgsz}  "
          f"conf>={pipeline.low_conf_threshold} (high {pipeline.conf_threshold})")

    summary = pipeline.process_video(args.source, output_path=args.output,
                                      show=args.show, max_frames=args.max_frames,
                                      save_db=args.db)
    print("\n--- RINGKASAN ---")
    print(f"Frame diproses     : {summary['total_frames']}")
    print(f"Total terhitung    : {summary['total_counted']}")
    print(f"Per kelas          : {summary['by_class']}")
    print(f"Per arah           : {summary['by_direction']}")


if __name__ == "__main__":
    main()
