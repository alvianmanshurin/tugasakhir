"""
Script uji coba deteksi kendaraan
"""
import cv2
import sys
sys.path.insert(0, 'src')
from ultralytics import YOLO

CLASS_NAMES = {0: 'motor', 1: 'mobil', 2: 'bus', 3: 'truk'}
COLORS = {'motor': (255, 0, 0), 'mobil': (0, 255, 0), 'bus': (0, 0, 255), 'truk': (255, 255, 0)}

model = YOLO('runs/detect/models/vehicle_detection/weights/best.pt')

img_path = sys.argv[1] if len(sys.argv) > 1 else 'data/annotated/images/val/merged_00010.jpg'
img = cv2.imread(img_path)
print(f'Gambar: {img_path} ({img.shape[1]}x{img.shape[0]})')

results = model(img, conf=0.35, iou=0.45, verbose=False)

counts = {'motor': 0, 'mobil': 0, 'bus': 0, 'truk': 0}
result_img = img.copy()

for r in results:
    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            if cls_id in CLASS_NAMES:
                cls_name = CLASS_NAMES[cls_id]
                counts[cls_name] += 1
                x1, y1, x2, y2 = [int(c) for c in xyxy]
                color = COLORS[cls_name]
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                label = f'{cls_name} {conf:.2f}'
                cv2.putText(result_img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

total = sum(counts.values())
print(f'\n=== HASIL DETEKSI ===')
print(f'Total kendaraan: {total}')
for name, count in counts.items():
    if count > 0:
        print(f'  {name}: {count}')

output_path = 'data/test_detection_result.jpg'
cv2.imwrite(output_path, result_img)
print(f'\nHasil tersimpan: {output_path}')
