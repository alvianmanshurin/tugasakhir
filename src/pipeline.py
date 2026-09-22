"""
Pipeline lengkap untuk deteksi kendaraan
"""
import cv2
import sys
sys.path.insert(0, 'src')
from ultralytics import YOLO
from utils.roi_filter import ROIFilter, ROIConfig, ROIBoundary
from utils.tracker import ObjectTracker
from utils.counter import VehicleCounter

class VehiclePipeline:
    """Pipeline lengkap deteksi kendaraan"""
    
    def __init__(self):
        self.model = YOLO('models/yolov8n_vehicle/weights/best.pt')
        self.CLASS_NAMES = {0: 'motor', 1: 'mobil', 2: 'bus', 3: 'truk'}
        self.COLORS = {'motor': (255, 0, 0), 'mobil': (0, 255, 0), 'bus': (0, 0, 255), 'truk': (255, 255, 0)}
        
        boundary = ROIBoundary()
        roi_config = ROIConfig(enabled=True, boundary=boundary, max_distance_m=20.0)
        self.roi_filter = ROIFilter(roi_config)
        self.tracker = ObjectTracker(max_age=30, min_hits=1, max_distance=80.0)
        self.counter = VehicleCounter()
    
    def process_frame(self, frame):
        """Proses satu frame"""
        results = self.model(frame, conf=0.35, iou=0.45, verbose=False)
        
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    if cls_id in self.CLASS_NAMES:
                        detections.append({
                            'class_id': cls_id,
                            'class_name': self.CLASS_NAMES[cls_id],
                            'confidence': conf,
                            'bbox': xyxy,
                        })
        
        filtered = self.roi_filter.filter_detections(detections, frame.shape)
        tracked = self.tracker.update(filtered)
        
        result_frame = self.roi_filter.draw_roi(frame.copy())
        for t in tracked:
            x1, y1, x2, y2 = [int(c) for c in t['bbox']]
            cls_name = t['class_name']
            track_id = t['track_id']
            color = self.COLORS[cls_name]
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            label = f'ID:{track_id} {cls_name}'
            cv2.putText(result_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result_frame, tracked
    
    def process_video(self, source, output_path=None, show=False):
        """Proses video"""
        cap = cv2.VideoCapture(0 if source == "0" else source)
        
        if not cap.isOpened():
            print(f"Gagal membuka: {source}")
            return
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result_frame, tracked = self.process_frame(frame)
            
            if writer:
                writer.write(result_frame)
            
            if show:
                cv2.imshow('Pipeline', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

def main():
    print("=== PIPELINE DETEKSI KENDARAAN ===")
    pipeline = VehiclePipeline()
    pipeline.process_video("0", show=True)

if __name__ == "__main__":
    main()
