"""
Script uji coba deteksi dengan ROI final (tanpa jarak, dengan ID)
"""
import cv2
import sys
sys.path.insert(0, 'src')
from ultralytics import YOLO
from utils.roi_filter import ROIFilter, ROIConfig, ROIBoundary
from utils.tracker import ObjectTracker

model = YOLO('runs/detect/models/vehicle_detection/weights/best.pt')
CLASS_NAMES = {0: 'motor', 1: 'mobil', 2: 'bus', 3: 'truk'}
COLORS = {'motor': (255, 0, 0), 'mobil': (0, 255, 0), 'bus': (0, 0, 255), 'truk': (255, 255, 0)}

boundary = ROIBoundary()
roi_config = ROIConfig(enabled=True, boundary=boundary)
roi_filter = ROIFilter(roi_config)
tracker = ObjectTracker(max_age=30, min_hits=1, max_distance=80.0)

img_path = sys.argv[1] if len(sys.argv) > 1 else 'data/annotated/images/val/merged_00013.jpg'
img = cv2.imread(img_path)
print(f'Gambar: {img_path}')
print(f'ROI: top_left={boundary.top_left}, top_right={boundary.top_right}')
print(f'    bottom_left={boundary.bottom_left}, bottom_right={boundary.bottom_right}')

results = model(img, conf=0.35, iou=0.45, verbose=False)

detections = []
for r in results:
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            if cls_id in CLASS_NAMES:
                detections.append({
                    'class_id': cls_id,
                    'class_name': CLASS_NAMES[cls_id],
                    'confidence': conf,
                    'bbox': xyxy,
                })

filtered = roi_filter.filter_detections(detections, img.shape)
tracked = tracker.update(filtered)

result_img = roi_filter.draw_roi(img.copy())
for t in tracked:
    x1, y1, x2, y2 = [int(c) for c in t['bbox']]
    cls_name = t['class_name']
    track_id = t['track_id']
    color = COLORS[cls_name]
    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
    label = f'ID:{track_id} {cls_name}'
    cv2.putText(result_img, label, (x1, y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite('data/test_roi_new_coords.jpg', result_img)

print(f'\n=== HASIL ===')
print(f'Deteksi awal: {len(detections)}')
print(f'Dalam ROI: {len(filtered)}')
for t in tracked:
    print(f'  ID:{t["track_id"]} | {t["class_name"]}: {t["confidence"]:.2f}')
print(f'Hasil: data/test_roi_new_coords.jpg')
