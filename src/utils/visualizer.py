"""Visualization utilities for Vehicle Detection System"""

import cv2
import numpy as np


class Visualizer:
    """Draw detection results, counts, and FPS on frames."""

    def __init__(self, config):
        self.config = config
        self.viz_cfg = config.get("visualization", {})
        self.color_map = self.viz_cfg.get(
            "color_map",
            {
                "motor": [255, 0, 0],
                "mobil": [0, 255, 0],
                "bus": [0, 0, 255],
                "truk": [255, 255, 0],
            },
        )
        self.font_scale = self.viz_cfg.get("font_scale", 0.6)
        self.font_thickness = self.viz_cfg.get("font_thickness", 2)
        self.show_fps = self.viz_cfg.get("show_fps", True)
        self.show_count = self.viz_cfg.get("show_count", True)

    def _get_color(self, class_name):
        """Get color for a class."""
        return self.color_map.get(class_name, [0, 255, 0])

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on frame."""
        result = frame.copy()

        for det in detections:
            bbox = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            color = self._get_color(class_name)

            x1, y1, x2, y2 = [int(c) for c in bbox]

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
            )
            cv2.rectangle(result, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

            # Draw label text
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                self.font_thickness,
                cv2.LINE_AA,
            )

        return result

    def draw_counting_line(self, frame, line_position):
        """Draw the counting line on the frame."""
        h, w = frame.shape[:2]
        line_y = int(h * line_position)

        # Draw dashed line
        dash_len = 20
        for x in range(0, w, dash_len * 2):
            cv2.line(
                frame,
                (x, line_y),
                (min(x + dash_len, w), line_y),
                (0, 255, 255),
                2,
            )

        # Draw label
        cv2.putText(
            frame,
            "COUNTING LINE",
            (10, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return frame

    def draw_count(self, frame, counts, total):
        """Draw vehicle count display on frame."""
        h, w = frame.shape[:2]

        # Create background panel
        panel_h = 120
        panel_w = 250
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)  # Dark gray background

        # Draw title
        cv2.putText(
            panel,
            "VEHICLE COUNT",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw separator line
        cv2.line(panel, (10, 35), (panel_w - 10, 35), (100, 100, 100), 1)

        # Draw counts
        y_offset = 55
        for class_name, count in counts.items():
            color = self._get_color(class_name)
            cv2.putText(
                panel,
                f"{class_name}:",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                panel,
                str(count),
                (panel_w - 50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_offset += 22

        # Draw total
        cv2.line(panel, (10, y_offset - 5), (panel_w - 10, y_offset - 5), (100, 100, 100), 1)
        cv2.putText(
            panel,
            "TOTAL:",
            (10, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            str(total),
            (panel_w - 50, y_offset + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Place panel on frame
        result = frame.copy()
        result[10 : 10 + panel_h, 10 : 10 + panel_w] = panel

        return result

    def draw_fps(self, frame, fps):
        """Draw FPS display on frame."""
        if not self.show_fps:
            return frame

        h, w = frame.shape[:2]
        fps_text = f"FPS: {fps:.1f}"

        # Background
        (text_w, text_h), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame, (w - text_w - 20, 10), (w - 10, 10 + text_h + 10), (40, 40, 40), -1
        )

        # Text color based on FPS
        if fps >= 30:
            color = (0, 255, 0)  # Green
        elif fps >= 15:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        cv2.putText(
            frame,
            fps_text,
            (w - text_w - 15, 10 + text_h + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

        return frame

    def draw_frame_info(self, frame, frame_count, total_counted):
        """Draw frame count and total counted vehicles."""
        h, w = frame.shape[:2]

        info_text = f"Frame: {frame_count} | Total: {total_counted}"
        (text_w, text_h), _ = cv2.getTextSize(
            info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # Background
        cv2.rectangle(
            frame,
            (10, h - 35),
            (10 + text_w + 10, h - 10),
            (40, 40, 40),
            -1,
        )

        # Text
        cv2.putText(
            frame,
            info_text,
            (15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return frame

    def draw_comparison(self, frame, yolo_count, manual_count):
        """Draw comparison between YOLO and manual count."""
        h, w = frame.shape[:2]

        text = f"YOLO: {yolo_count} | Manual: {manual_count}"
        diff = abs(yolo_count - manual_count)
        accuracy = (
            min(yolo_count, manual_count) / max(yolo_count, manual_count) * 100
            if max(yolo_count, manual_count) > 0
            else 100
        )

        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Background
        cv2.rectangle(
            frame,
            (w // 2 - text_w // 2 - 10, h - 50),
            (w // 2 + text_w // 2 + 10, h - 10),
            (40, 40, 40),
            -1,
        )

        # Main text
        cv2.putText(
            frame,
            text,
            (w // 2 - text_w // 2, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Accuracy
        acc_color = (0, 255, 0) if accuracy >= 90 else (0, 255, 255)
        acc_text = f"Accuracy: {accuracy:.1f}%"
        cv2.putText(
            frame,
            acc_text,
            (w // 2 - 50, h - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            acc_color,
            1,
            cv2.LINE_AA,
        )

        return frame
