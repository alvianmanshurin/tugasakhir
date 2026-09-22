"""
Helper untuk anotasi gambar
"""
import os
import json
from pathlib import Path

def create_labelme_template(image_path, output_path):
    """Membuat template LabelMe JSON"""
    template = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": 0,
        "imageWidth": 0
    }
    
    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)
    
    print(f"Template dibuat: {output_path}")

def convert_labelme_to_yolo(json_path, output_path, class_mapping):
    """Konversi LabelMe ke format YOLO"""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    image_w = data["imageWidth"]
    image_h = data["imageHeight"]
    
    yolo_lines = []
    for shape in data["shapes"]:
        label = shape["label"]
        if label not in class_mapping:
            continue
        
        cls_id = class_mapping[label]
        points = shape["points"]
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        x_center = (x_min + x_max) / 2 / image_w
        y_center = (y_min + y_max) / 2 / image_h
        width = (x_max - x_min) / image_w
        height = (y_max - y_min) / image_h
        
        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    with open(output_path, "w") as f:
        f.write("\n".join(yolo_lines))

def main():
    print("=== HELPER ANOTASI ===")
    # Tambahkan kode sesuai kebutuhan

if __name__ == "__main__":
    main()
