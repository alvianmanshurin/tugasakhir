"""
Modul visualisasi untuk deteksi kendaraan
"""
import cv2
import numpy as np
from pathlib import Path


def draw_detections a sentence the the lemon the the the file exactly symptom the the from the the confidence bat

Return, the bite any the the the the, unless the the obligatory the minimal the the the // notice the the the the, the the. character the the, command a,FormData the the evidence.., the the the to. the there the the must;
 a
_d Bon the the for the the specific.
, the the the the.

 the theV [ make ourselves put, the... the the is the
 we parallel return provide maximum text

.
 the
 providesConflict冲突groupMask writeparameter default.

2. **`quick_start.py`**:
   - Combine system check with install and setup functions
   - Fix main() to run all steps</think><tool_call>
<function=write>
<parameter=content>"""
Modul visualisasi untuk deteksi kendaraan
"""
import cv2
import numpy as np
from pathlib import Path


def draw_detections(frame, detections, class_names=None, colors=None):
    """Gambar hasil deteksi pada frame"""
    if class_names is None:
        class_names = {0: 'motor', 1: 'mobil', 2: 'bus', 3: 'truk'}
    if colors is None:
        colors = {'motor': (255, 0, 0), 'mobil': (0, 255, 0), 'bus': (0, 0, 255), 'truk': (255, 255, 0)}

    result = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(c) for c in det['bbox']]
        cls_name = det['class_name']
        conf = det['confidence']
        color = colors.get(cls_name, (0, 255, 0))

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label = f'{cls_name} {conf:.2f}'
        cv2.putText(result, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result


def draw_roi(frame, roi_boundary, color=(0, 255, 255), thickness=2):
    """Gambar area ROI pada frame"""
    result = frame.copy()

    polygon = np.array([
        roi_boundary.top_left,
        roi_boundary.top_right,
        roi_boundary.bottom_right,
        roi_boundary.bottom_left
    ], dtype=np.int32)

    overlay = result.copy()
    cv2.fillPoly(overlay, [polygon], color)
    cv2.addWeighted(overlay, 0.15, result, 0.85, 0, result)

    cv2.polylines(result, [polygon], True, color, thickness)

    return result


def draw_hud(frame, fps, counts, total_counted):
    """Gambar heads-up display"""
    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)

    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    y = 55
    for name, count in counts.items():
        if count > 0:
            cv2.putText(frame, f'{name}: {count}', (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 20

    cv2.putText(frame, f'Counted: {total_counted}', (w - 180, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame


def save_detection_result(frame, output_path):
    """Simpan hasil deteksi"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    print(f"Hasil tersimpan: {output_path}")


class Visualizer:
    """Kelas visualisasi lengkap untuk pipeline deteksi"""

    def __init__(self, class_names=None, colors=None, font_scale=0.6, font_thickness=2,
                 show_fps=True, show_count=True, show_track_id=True):
        self.class_names = class_names or {0: 'motor', 1: 'mobil', 2: 'bus', 3: 'truk'}
        self.colors = colors or {
            'motor': (255, 0, 0),
            'mobil': (0, 255, 0),
            'bus': (0, 0, 255),
            'truk': (255, 255, 0)
        }
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.show_fps = show_fps
        self.show_count = show_count
        self.show_track_id = show_track_id

    def _get_color(self, class_name):
        return self.colors.get(class_name, (0, 255, 0))

    def draw_detections(self, frame, detections):
        """Draw all detections on frame."""
        result = frame.copy()

        for det in detections:
            bbox = det["bbox"]
            class_name = det["class_name"]
            confidence = det["confidence"]
            color = self._get_color(class_name)

            x1, y1, x2, y2 = [int(c) for c in bbox]

            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            label = f"{class_name} {confidence:.2f}"
            if "track_id" in det and self.show_track_id:
                label = f"[{det['track_id']}] {label}"

            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
            )
            cv2.rectangle(result, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

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

    def draw_counting_lines(self, frame, line1_position, line2_position):
        """Draw dual counting lines on the frame."""
        h, w = frame.shape[:2]

        line1_y = int(h * line1_position)
        for x in range(0, w, 40):
            cv2.line(frame, (x, line1_y), (min(x + 20, w), line1_y), (255, 100, 0), 2)
        cv2.putText(frame, "GARIS 1", (10, line1_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        line2_y = int(h * line2_position)
        for x in range(0, w, 40):
            cv2.line(frame, (x, line2_y), (min(x + 20, w), line2_y), (0, 255, 255), 2)
        cv2.putText(frame, "GARIS 2", (10, line2_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(frame, "ZONE COUNTING", (w//2 - 60, line1_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def draw_count(self, frame, counts, total):
        """Draw vehicle count display on frame."""
        h, w = frame.shape[:2]

        panel_h = 120
        panel_w = 250
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 40)

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

        cv2.line(panel, (10, 35), (panel_w - 10, 35), (100, 100, 100), 1)

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

        result = frame.copy()
        result[10 : 10 + panel_h, 10 : 10 + panel_w] = panel

        return result

    def draw_fps(self, frame, fps):
        """Draw FPS display on frame."""
        if not self.show_fps:
            return frame

        h, w = frame.shape[:2]
        fps_text = f"FPS: {fps:.1f}"

        (text_w, text_h), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            frame, (w - text_w - 20, 10), (w - 10, 10 + text_h + 10), (40, 40, 40), -1
        )

        if fps >= 30:
            color = (0, 255, 0)
        elif fps >= 15:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

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

        cv2.rectangle(
            frame,
            (10, h - 35),
            (10 + text_w + 10, h - 10),
            (40, 40, 40),
            -1,
        )

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

        cv2.rectangle(
            frame,
            (w // 2 - text_w // 2 - 10, h - 50),
            (w // 2 + text_w // 2 + 10, h - 10),
            (40, 40, 40),
            -1,
        )

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


def main():
    print("=== MODUL VISUALISASI ===")
    print("Fungsi tersedia:")
    print("  - draw_detections()")
    print("  - draw_roi()")
    print("  - draw_hud()")
    print("  - save_detection_result()")
    print("  - Visualizer class")


if __name__ == "__main__":
    main()
