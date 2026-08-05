"""
Script uji coba deteksi dengan ROI filter (tanpa jarak, dengan ID)
"""
import cv2
import sys
sys.path.insert(0, 'src')
from ultralytics import YOLO
from utils.roi_filter import ROIFilter, ROIConfig, ROIBoundary
from utils.tracker import ObjectTracker

model = YOLO('models/yolov8n_vehicle/weights/best.pt')
CLASS_NAMES = {0: 'motor', 1: 'mobil', 2: 'bus', 3: 'truk'}
COLORS = {'motor': (255, 0, 0), 'mobil': (0, 255, 0), 'bus': (0, 0, 255), 'truk': (255, 255, 0)}

boundary = ROIBoundary()
roi_config = ROIConfig(enabled=True, boundary=boundary)
roi_filter = ROIFilter(roi_config)
tracker = ObjectTracker(max_age=30, min_hits=1, max_distance=80.0)

img_path = sys.argv[1] if len(sys.argv) > 1 else 'data/annotated/images/val/merged_00010.jpg'
img = cv2.imread(img_path)
print(f'Gambar: {img_path} ({img.shape[1]}x{img.shape[0]})')

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

print(f'\nDeteksi awal: {len(detections)} kendaraan')

filtered = roi_filter.filter_detections(detections, img.shape)
print(f'Setelah ROI (20m): {len(filtered)} kendaraan')

tracked = tracker.update(filtered)

result_img = roi_filter.draw_roi(img.copy())

counts = {'motor': 0, 'mobil': 0, 'bus': 0, 'truk': 0}
for t in tracked:
    x1, y1, x2, y2 = [int(c) for c in t['bbox']]
    cls_name = t['class_name']
    track_id = t['track_id']
    color = COLORS[cls_name]
    counts[cls_name] += 1
    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
    label = f'ID:{track_id} {cls_name}'
    cv2.putText(result_img, label, (x1, y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.rectangle(result_img, (0, 0), (450, 110), (0, 0, 0), -1)
cv2.putText(result_img, f'ROI 20m | Deteksi: {len(detections)} | Dalam ROI: {len(filtered)}',
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
y = 60
for name in ['motor', 'mobil', 'bus', 'truk']:
    if counts[name] > 0:
        cv2.putText(result_img, f'{name}: {counts[name]}', (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[name], 2)
        y += 25

output_path = 'data/test_detection_with_roi.jpg'
cv2.imwrite(output_path, result_img)

print(f'\n=== RINCIAN DALAM ROI ===')
for t in tracked:
    print(f'  ID:{t["track_id"]} | {t["class_name"]}: {t["confidence"]:.2f}')
print(f'\nHasil tersimpan: {output_path}')
