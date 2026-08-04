"""Auto-annotation script using pretrained YOLOv8 (COCO weights)
Maps COCO classes to project classes:
  COCO: car(2), motorcycle(3), bus(5), truck(7)
  Project: motor(0), mobil(1), bus(2), truk(3)
"""

import os
import cv2
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO

# COCO class ID -> Project class ID
COCO_TO_PROJECT = {
    3: 0,   # motorcycle -> motor
    2: 1,   # car -> mobil
    5: 2,   # bus -> bus
    7: 3,   # truck -> truk
}

PROJECT_NAMES = {0: "motor", 1: "mobil", 2: "bus", 3: "truk"}


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def auto_annotate(
    image_dir,
    label_dir,
    model_name="yolov8n.pt",
    conf_threshold=0.35,
    iou_threshold=0.45,
    img_size=640,
):
    """Auto-annotate images using pretrained YOLOv8."""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    # Gather image files
    image_files = sorted(
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )

    if not image_files:
        print(f"[ERROR] No images found in {image_dir}")
        return

    print(f"[INFO] Found {len(image_files)} images in {image_dir}")
    print(f"[INFO] Loading model: {model_name}")

    model = YOLO(model_name)

    total_boxes = 0
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    skipped = 0

    for i, img_path in enumerate(image_files):
        results = model.predict(
            source=str(img_path),
            imgsz=img_size,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        label_lines = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            img_h, img_w = r.orig_shape

            for box in r.boxes:
                coco_cls = int(box.cls[0])
                if coco_cls not in COCO_TO_PROJECT:
                    continue

                proj_cls = COCO_TO_PROJECT[coco_cls]

                # Convert to YOLO normalized format
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                # Clamp values
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                w = max(0.0, min(1.0, w))
                h = max(0.0, min(1.0, h))

                label_lines.append(f"{proj_cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
                class_counts[proj_cls] += 1
                total_boxes += 1

        # Write label file
        lbl_file = label_dir / (img_path.stem + ".txt")
        with open(lbl_file, "w") as f:
            f.write("\n".join(label_lines))

        if not label_lines:
            skipped += 1

        # Progress
        if (i + 1) % 50 == 0 or (i + 1) == len(image_files):
            print(f"  [{i+1}/{len(image_files)}] processed... ({total_boxes} boxes so far)")

    print("\n" + "=" * 60)
    print("AUTO-ANNOTATION COMPLETE")
    print("=" * 60)
    print(f"  Total images:  {len(image_files)}")
    print(f"  Total boxes:   {total_boxes}")
    print(f"  Empty images:  {skipped}")
    print(f"\n  Per-class breakdown:")
    for cls_id, name in PROJECT_NAMES.items():
        print(f"    {cls_id} ({name}): {class_counts[cls_id]}")
    print(f"\n  Labels saved to: {label_dir}")


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate images with YOLOv8 pretrained")
    parser.add_argument("--image-dir", type=str, required=True, help="Image directory")
    parser.add_argument("--label-dir", type=str, required=True, help="Label output directory")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold (default: 0.35)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold (default: 0.45)")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size (default: 640)")
    args = parser.parse_args()

    auto_annotate(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        model_name=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
