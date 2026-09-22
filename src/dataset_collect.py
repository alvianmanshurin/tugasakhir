"""
Script untuk pengumpulan dataset
"""
import os
import cv2
from pathlib import Path
from datetime import datetime

def collect_from_webcam(output_dir, num_frames=100):
    """Mengumpulkan gambar dari webcam"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak dapat membuka webcam")
        return
    
    print(f"Mengumpulkan {num_frames} gambar dari webcam...")
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"webcam_{timestamp}_{i:05d}.jpg"
        cv2.imwrite(str(output_dir / filename), frame)
        
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{num_frames}")
    
    cap.release()
    print(f"Selesai! Gambar tersimpan di: {output_dir}")

def main():
    print("=== PENGUMPULAN DATASET ===")
    # Tambahkan kode sesuai kebutuhan

if __name__ == "__main__":
    main()
