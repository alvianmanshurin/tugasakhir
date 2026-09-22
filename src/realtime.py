"""
Deteksi real-time
"""
import os
import cv2
import yaml
import time
import sys
import numpy as np
sys.path.insert(0, 'src')
from ultralytics import YOLO
from utils.counter import VehicleCounter


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class LightweightRealtimeDetector:
    """Lightweight real-time vehicle detector for CPU."""

    def __init__(self, config):
        self.config = config
        self.model_cfg = config["model"]
        self.count_cfg = config["counting"]
        self.class_names = config["dataset"]["names"]

        # Load model
        model_path = "models/vehicle_detection/weights/best.pt"
        if not os.path.exists(model_path):
            model_path = self.model_cfg["architecture"]

        print(f"[INFO] Loading model: {model_path}")
        self.model = YOLO(model_path)

        # Counter (dual-line)
        self.counter = VehicleCounter(
            line1_position=self.count_cfg.get("line1_position", 0.48),
            line2_position=self.count_cfg.get("line2_position", 0.75),
            direction=self.count_cfg["direction"],
            min_track_length=self.count_cfg.get("min_track_length", 3),
            max_lost_frames=self.count_cfg["max_lost_frames"],
        )

        # Colors
        self.colors = {
            "motor": (255, 0, 0),
            "mobil": (0, 255, 0),
            "bus": (0, 0, 255),
            "truk": (255, 255, 0),
        }

        # FPS tracking
        self.fps_history = []

    def process(self, source=None, output_path=None, show=False):
        """Process video or webcam."""
        if source is None:
            source = "0"

        # Open source
        if source == "0" or source == 0:
            cap = cv2.VideoCapture(0)
            is_webcam = True
        else:
            cap = cv2.VideoCapture(source)
            is_webcam = False

        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {source}")
            return

        # Get properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30

        print(f"\n[INFO] Source: {'Webcam' if is_webcam else source}")
        print(f"[INFO] Resolution: {width}x{height}")
        print(f"[INFO] Press 'q' to quit")

        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_src, (width, height))

        self.counter.reset()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            start = time.time()

            # Resize for faster inference
            input_size = self.model_cfg["input_size"]
            frame_resized = cv2.resize(frame, (input_size, input_size))

            # Inference
            results = self.model(
                frame_resized,
                conf=self.model_cfg["confidence_threshold"],
                iou=self.model_cfg["iou_threshold"],
                verbose=False,
            )

            # Process detections
            detections = []
            counts = {name: 0 for name in self.class_names.values()}

            # Scale factor
            scale_x = frame.shape[1] / input_size
            scale_y = frame.shape[0] / input_size

            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        xyxy = box.xyxy[0].tolist()

                        # Scale back to original size
                        xyxy[0] *= scale_x
                        xyxy[1] *= scale_y
                        xyxy[2] *= scale_x
                        xyxy[3] *= scale_y

                        if cls_id < len(self.class_names):
                            cls_name = self.class_names[cls_id]
                            detections.append({
                                "class_id": cls_id,
                                "class_name": cls_name,
                                "confidence": conf,
                                "bbox": xyxy,
                            })
                            counts[cls_name] += 1

            # Update counter
            self.counter.update(detections, frame.shape[0])

            # Draw
            frame_result = self._draw_frame(frame, detections, counts)

            # FPS
            inference_time = time.time() - start
            fps = 1.0 / inference_time if inference_time > 0 else 0
            self.fps_history.append(fps)
            avg_fps = np.mean(self.fps_history[-30:])

            # Draw FPS
            cv2.putText(frame_result, f"FPS: {avg_fps:.1f}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Draw count
            total = sum(counts.values())
            cv2.putText(frame_result, f"Vehicles: {total}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Write
            if writer:
                writer.write(frame_result)

            # Show
            if show:
                cv2.imshow("Vehicle Detection - ITERA", frame_result)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Progress
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count} | FPS: {avg_fps:.1f} | Vehicles: {total}")

        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        # Summary
        print(f"\n{'='*50}")
        print(f"PROCESSED {frame_count} frames")
        print(f"Average FPS: {np.mean(self.fps_history):.1f}")
        print(f"Total counted: {self.counter.total_count}")
        print(f"{'='*50}")

    def _draw_frame(self, frame, detections, counts):
        """Draw detections on frame."""
        result = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            color = self.colors.get(det["class_name"], (0, 255, 0))

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(result, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return result


def realtime_detection():
    """Deteksi real-time menggunakan webcam"""
    model = YOLO('models/vehicle_detection/weights/best.pt')
    CLASS_NAMES = {0: 'motor', 1: 'mobil', 2: 'bus', 3: 'truk'}
    COLORS = {'motor': (255, 0, 0), 'mobil': (0, 255, 0), 'bus': (0, 0, 255), 'truk': (255, 255, 0)}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak dapat membuka webcam")
        return

    print("Tekan 'q' untuk keluar")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model(frame, conf=0.35, iou=0.45, verbose=False)
        fps = 1.0 / (time.time() - start)

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    if cls_id in CLASS_NAMES:
                        cls_name = CLASS_NAMES[cls_id]
                        x1, y1, x2, y2 = [int(c) for c in xyxy]
                        color = COLORS[cls_name]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f'{cls_name} {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Real-time Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print("=== DETEKSI REAL-TIME ===")
    realtime_detection()


if __name__ == "__main__":
    main()
