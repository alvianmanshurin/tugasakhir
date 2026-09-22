"""
Script monitoring real-time
"""
import cv2
import time
import sys
sys.path.insert(0, 'src')
from ultralytics import YOLO

def monitor_webcam():
    """Monitor webcam secara real-time"""
    model = YOLO('models/yolov8n_vehicle/weights/best.pt')
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
        
        counts = {}
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    if cls_id in CLASS_NAMES:
                        cls_name = CLASS_NAMES[cls_id]
                        counts[cls_name] = counts.get(cls_name, 0) + 1
                        x1, y1, x2, y2 = [int(c) for c in xyxy]
                        color = COLORS[cls_name]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f'{cls_name} {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Monitoring', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("=== MONITORING REAL-TIME ===")
    monitor_webcam()

if __name__ == "__main__":
    main()
