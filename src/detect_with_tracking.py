"""Vehicle Detection + ROI Filtering + Object Tracking Pipeline
Optimized for CPU (Intel i3-1115G4, 8GB RAM)
"""

import os
import sys
import cv2
import yaml
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from ultralytics import YOLO
from utils.roi_filter import ROIFilter, ROIConfig, ROIBoundary
from utils.tracker import ObjectTracker
from utils.counter import VehicleCounter


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class VehicleDetectionPipeline:
    """Full pipeline: YOLOv8 Detection + ROI Filter + Object Tracker + Counter."""

    CLASS_NAMES = {0: "motor", 1: "mobil", 2: "bus", 3: "truk"}
    COLORS = {
        "motor": (255, 0, 0),
        "mobil": (0, 255, 0),
        "bus": (0, 0, 255),
        "truk": (255, 255, 0),
    }

    def __init__(self, config: dict, model_path: str = None):
        self.config = config
        model_cfg = config["model"]
        roi_cfg = config.get("roi", {})
        track_cfg = config.get("tracking", {})
        count_cfg = config.get("counting", {})

        # Load YOLO model
        if model_path is None:
            model_path = "models/yolov8n_vehicle/weights/best.pt"
            if not os.path.exists(model_path):
                model_path = model_cfg["architecture"]

        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)

        # ROI Filter
        boundary_cfg = roi_cfg.get("boundary", {})
        boundary = ROIBoundary(
            top_left=tuple(boundary_cfg.get("top_left", [150, 120])),
            top_right=tuple(boundary_cfg.get("top_right", [490, 120])),
            bottom_left=tuple(boundary_cfg.get("bottom_left", [0, 416])),
            bottom_right=tuple(boundary_cfg.get("bottom_right", [640, 416])),
            max_distance_m=roi_cfg.get("max_distance_m", 20.0),
        )
        self.roi_config = ROIConfig(
            enabled=roi_cfg.get("enabled", True),
            boundary=boundary,
            max_distance_m=roi_cfg.get("max_distance_m", 20.0),
            min_bbox_height=roi_cfg.get("min_bbox_height", 20),
        )
        self.roi_filter = ROIFilter(self.roi_config)
        self.show_roi = roi_cfg.get("draw_roi", True)

        # Object Tracker
        self.tracker = ObjectTracker(
            max_age=track_cfg.get("max_age", 30),
            min_hits=track_cfg.get("min_hits", 3),
            max_distance=track_cfg.get("max_distance", 80.0),
            track_buffer=track_cfg.get("track_buffer", 50),
        )
        self.show_track_id = track_cfg.get("show_track_id", True)
        self.show_velocity = track_cfg.get("show_velocity", False)

        # Vehicle Counter (line crossing)
        self.counter = VehicleCounter(
            line_position=count_cfg.get("line_position", 0.5),
            direction=count_cfg.get("direction", "both"),
            min_track_length=count_cfg.get("min_track_length", 5),
            max_lost_frames=count_cfg.get("max_lost_frames", 30),
        )

        # FPS tracking
        self.fps_history = []
        self.total_detections = 0
        self.total_filtered = 0

    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame. Returns (annotated_frame, stats)."""
        start = time.time()
        input_size = self.config["model"]["input_size"]
        conf_thresh = self.config["model"]["confidence_threshold"]
        iou_thresh = self.config["model"]["iou_threshold"]

        # Resize for inference
        frame_resized = cv2.resize(frame, (input_size, input_size))

        # YOLO inference
        results = self.model(
            frame_resized,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
        )

        # Scale factors
        scale_x = frame.shape[1] / input_size
        scale_y = frame.shape[0] / input_size

        # Parse detections
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.CLASS_NAMES:
                    continue
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                # Scale back to original
                xyxy[0] *= scale_x
                xyxy[1] *= scale_y
                xyxy[2] *= scale_x
                xyxy[3] *= scale_y

                detections.append({
                    "class_id": cls_id,
                    "class_name": self.CLASS_NAMES[cls_id],
                    "confidence": conf,
                    "bbox": xyxy,
                })

        self.total_detections += len(detections)

        # ROI filtering
        filtered = self.roi_filter.filter_detections(detections, frame.shape)
        self.total_filtered += len(filtered)

        # Object tracking
        tracked = self.tracker.update(filtered)

        # Update counter (for line crossing)
        counter_input = [
            {
                "class_id": t["class_id"],
                "class_name": t["class_name"],
                "confidence": t["confidence"],
                "bbox": t["bbox"],
            }
            for t in tracked
        ]
        self.counter.update(counter_input, frame.shape[0])

        # FPS
        inference_time = time.time() - start
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
        avg_fps = np.mean(self.fps_history[-30:])

        # Draw results
        result_frame = frame.copy()

        # Draw ROI
        if self.show_roi:
            result_frame = self.roi_filter.draw_roi(result_frame)

        # Draw counting line
        line_y = int(frame.shape[0] * self.counter.line_position)
        cv2.line(result_frame, (0, line_y), (frame.shape[1], line_y),
                (0, 255, 255), 2)
        cv2.putText(result_frame, "COUNTING LINE", (10, line_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw tracked objects
        for t in tracked:
            x1, y1, x2, y2 = [int(c) for c in t["bbox"]]
            cls_name = t["class_name"]
            color = self.COLORS.get(cls_name, (0, 255, 0))

            # Bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # Label
            track_id = t["track_id"]
            label_parts = [f"ID:{track_id} {cls_name}"]
            if "distance_m" in t:
                label_parts.append(f"{t['distance_m']}m")
            label_parts.append(f"{t['confidence']:.2f}")
            label = " ".join(label_parts)

            cv2.putText(result_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Velocity arrow
            if self.show_velocity and "velocity" in t:
                vx, vy = t["velocity"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                end_x = int(cx + vx * 3)
                end_y = int(cy + vy * 3)
                cv2.arrowedLine(result_frame, (cx, cy), (end_x, end_y),
                              (255, 255, 255), 2)

        # HUD
        self._draw_hud(result_frame, tracked, avg_fps)

        stats = {
            "fps": avg_fps,
            "total_detections": len(detections),
            "after_roi": len(filtered),
            "tracked": len(tracked),
            "counted": self.counter.total_count,
        }

        return result_frame, stats

    def _draw_hud(self, frame, tracked, fps):
        """Draw heads-up display on frame."""
        h, w = frame.shape[:2]

        # Background panel
        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (w, 90), (100, 100, 100), 1)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Vehicle counts
        counts = {}
        for t in tracked:
            name = t["class_name"]
            counts[name] = counts.get(name, 0) + 1

        y_offset = 55
        for name in ["motor", "mobil", "bus", "truk"]:
            c = counts.get(name, 0)
            color = self.COLORS[name]
            cv2.putText(frame, f"{name}: {c}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 20

        # Total counted (line crossing)
        cv2.putText(frame, f"Counted: {self.counter.total_count}",
                   (w - 180, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (0, 255, 255), 2)

        # Active tracks
        cv2.putText(frame, f"Active: {len(tracked)}",
                   (w - 180, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 1)

    def process_video(self, source, output_path=None, show=False, max_frames=None):
        """Process video file or webcam."""
        is_webcam = source in ["0", 0]
        cap = cv2.VideoCapture(0 if is_webcam else source)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {source}")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

        print(f"\n[INFO] Source: {'Webcam' if is_webcam else source}")
        print(f"[INFO] Resolution: {width}x{height}")
        print(f"[INFO] ROI: {'ON' if self.roi_config.enabled else 'OFF'} "
              f"(max {self.roi_config.max_distance_m}m)")
        print(f"[INFO] Tracking: {'ON' if self.tracker else 'OFF'}")
        print(f"[INFO] Press 'q' to quit\n")

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_src, (width, height))

        self.counter.reset()
        self.tracker.reset()
        self.fps_history = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if max_frames and frame_count > max_frames:
                break

            result_frame, stats = self.process_frame(frame)

            if writer:
                writer.write(result_frame)

            if show:
                cv2.imshow("Vehicle Detection + ROI + Tracking", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_count % 30 == 0:
                print(f"  Frame {frame_count} | FPS: {stats['fps']:.1f} | "
                      f"Det: {stats['total_detections']} -> ROI: {stats['after_roi']} | "
                      f"Tracked: {stats['tracked']} | Counted: {stats['counted']}")

        # Summary
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"  Frames processed: {frame_count}")
        print(f"  Average FPS: {np.mean(self.fps_history):.1f}")
        print(f"  Total detections: {self.total_detections}")
        print(f"  After ROI filter: {self.total_filtered}")
        print(f"  Final count: {self.counter.total_count}")
        print(f"  Count by class: {dict(self.counter.class_counts)}")
        print(f"{'='*60}")

        return {
            "frames": frame_count,
            "fps": np.mean(self.fps_history),
            "total_detections": self.total_detections,
            "total_filtered": self.total_filtered,
            "count": self.counter.total_count,
            "by_class": dict(self.counter.class_counts),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Vehicle Detection + ROI + Tracking Pipeline"
    )
    parser.add_argument("--source", type=str, default="0",
                       help="Video file path or '0' for webcam")
    parser.add_argument("--model", type=str, default=None,
                       help="Path to YOLOv8 .pt model")
    parser.add_argument("--output", type=str, default=None,
                       help="Output video path")
    parser.add_argument("--show", action="store_true",
                       help="Show preview window")
    parser.add_argument("--conf", type=float, default=None,
                       help="Override confidence threshold")
    parser.add_argument("--roi-max-dist", type=float, default=None,
                       help="Override ROI max distance (meters)")
    parser.add_argument("--no-roi", action="store_true",
                       help="Disable ROI filtering")
    parser.add_argument("--no-track", action="store_true",
                       help="Disable object tracking")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Max frames to process")
    args = parser.parse_args()

    config = load_config()

    # CLI overrides
    if args.conf:
        config["model"]["confidence_threshold"] = args.conf
    if args.roi_max_dist:
        config["roi"]["max_distance_m"] = args.roi_max_dist
    if args.no_roi:
        config["roi"]["enabled"] = False
    if args.no_track:
        config["tracking"]["enabled"] = False

    pipeline = VehicleDetectionPipeline(config, model_path=args.model)
    pipeline.process_video(
        source=args.source,
        output_path=args.output,
        show=args.show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
