"""
Script untuk batch processing gambar
"""
import os
import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

def batch_process(input_dir, output_dir, model_path="models/yolov8n_vehicle/weights/best.pt"):
    """Proses batch gambar"""
    model = YOLO(model_path)
    CLASS_NAMES = {0: 'motor', 1: 'mobil', 2: 'bus', 3: 'truk'}
    COLORS = {'motor': (255, 0, 0), 'mobil': (0, 255, 0), 'bus': (0, 0, 255), 'truk': (255, 255, 0)}
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    
    print(f"Memproses {len(images)} gambar...")
    
    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        results = model(img, conf=0.35, iou=0.45, verbose=False)
        
        result_img = img.copy()
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
                        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
                        label = f'{cls_name} {conf:.2f}'
                        cv2.putText(result_img, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        output_path = output_dir / f"detected_{img_path.name}"
        cv2.imwrite(str(output_path), result_img)
        
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{len(images)}")
    
    print(f"Selesai! Hasil tersimpan di: {output_dir}")

def main():
    print("=== BATCH PROCESSING ===")
    # Tambahkan kode sesuai kebutuhan

if __name__ == "__main__":
    main()
