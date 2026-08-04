"""Annotation Helper Tool for Vehicle Detection Dataset"""

import os
import cv2
import yaml
import argparse
import json
from pathlib import Path


def load_config(config_path="config/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class AnnotationHelper:
    """Interactive annotation helper for vehicle detection."""

    def __init__(self, config):
        self.config = config
        self.class_names = config["dataset"]["names"]
        self.current_class = 0
        self.bboxes = []
        self.drawing = False
        self.start_point = None
        self.current_image = None
        self.current_image_path = None

    def create_annotation_template(self, image_dir, output_dir):
        """Create template annotation files for all images."""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

        created = 0
        for img_file in image_files:
            lbl_file = output_dir / (img_file.stem + ".txt")
            if not lbl_file.exists():
                lbl_file.touch()
                created += 1

        print(f"[OK] Created {created} template annotation files")
        print(f"[INFO] Use LabelImg or this tool to add annotations")

    def export_to_labelme(self, yolo_dir, image_dir, output_dir):
        """Export YOLO annotations to LabelMe format for review."""
        import json

        yolo_dir = Path(yolo_dir)
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        label_files = list(yolo_dir.glob("*.txt"))
        exported = 0

        for lbl_file in label_files:
            img_file = image_dir / (lbl_file.stem + ".jpg")
            if not img_file.exists():
                img_file = image_dir / (lbl_file.stem + ".png")
            if not img_file.exists():
                continue

            # Read image to get dimensions
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w = img.shape[:2]

            # Read YOLO annotations
            shapes = []
            with open(lbl_file, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    cls_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    bbox_w = float(parts[3]) * w
                    bbox_h = float(parts[4]) * h

                    # Convert to corners
                    x_min = x_center - bbox_w / 2
                    y_min = y_center - bbox_h / 2
                    x_max = x_center + bbox_w / 2
                    y_max = y_center + bbox_h / 2

                    label = self.class_names.get(cls_id, f"class_{cls_id}")

                    shapes.append({
                        "label": label,
                        "points": [[x_min, y_min], [x_max, y_max]],
                        "shape_type": "rectangle",
                        "flags": {}
                    })

            # Create LabelMe JSON
            labelme_data = {
                "version": "4.5.6",
                "flags": {},
                "shapes": shapes,
                "imagePath": img_file.name,
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w
            }

            output_file = output_dir / (lbl_file.stem + ".json")
            with open(output_file, "w") as f:
                json.dump(labelme_data, f, indent=2)

            exported += 1

        print(f"[OK] Exported {exported} annotations to LabelMe format")

    def visualize_annotations(self, image_dir, label_dir, num_images=10):
        """Visualize annotations on images."""
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        image_files = image_files[:num_images]

        colors = [
            (255, 0, 0),    # motor - Red
            (0, 255, 0),    # mobil - Green
            (0, 0, 255),    # bus - Blue
            (255, 255, 0),  # truk - Yellow
        ]

        for img_file in image_files:
            img = cv2.imread(str(img_file))
            if img is None:
                continue

            lbl_file = label_dir / (img_file.stem + ".txt")
            if lbl_file.exists():
                with open(lbl_file, "r") as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        cls_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        bbox_w = float(parts[3])
                        bbox_h = float(parts[4])

                        h, w = img.shape[:2]
                        x1 = int((x_center - bbox_w / 2) * w)
                        y1 = int((y_center - bbox_h / 2) * h)
                        x2 = int((x_center + bbox_w / 2) * w)
                        y2 = int((y_center + bbox_h / 2) * h)

                        color = colors[cls_id % len(colors)]
                        label = self.class_names.get(cls_id, f"class_{cls_id}")

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Show image
            cv2.imshow("Annotations", img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

    def count_annotations(self, label_dir):
        """Count total annotations per class."""
        label_dir = Path(label_dir)
        label_files = list(label_dir.glob("*.txt"))

        total_counts = {name: 0 for name in self.class_names.values()}
        total_files = 0
        empty_files = 0

        for lbl_file in label_files:
            has_annotations = False
            with open(lbl_file, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id in self.class_names:
                            total_counts[self.class_names[cls_id]] += 1
                            has_annotations = True

            total_files += 1
            if not has_annotations:
                empty_files += 1

        print("\n" + "=" * 60)
        print("ANNOTATION STATISTICS")
        print("=" * 60)
        print(f"\nTotal label files: {total_files}")
        print(f"Empty files: {empty_files}")
        print(f"Files with annotations: {total_files - empty_files}")
        print(f"\nAnnotations per class:")
        for name, count in total_counts.items():
            print(f"  {name}: {count}")
        print(f"\nTotal annotations: {sum(total_counts.values())}")

        return total_counts


def main():
    parser = argparse.ArgumentParser(description="Annotation Helper Tool")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--action", type=str, required=True,
                       choices=["template", "export-labelme", "visualize", "count"],
                       help="Action to perform")
    parser.add_argument("--image-dir", type=str, help="Image directory")
    parser.add_argument("--label-dir", type=str, help="Label directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--num", type=int, default=10, help="Number of images")
    args = parser.parse_args()

    config = load_config(args.config)
    helper = AnnotationHelper(config)

    if args.action == "template":
        helper.create_annotation_template(args.image_dir, args.label_dir)
    elif args.action == "export-labelme":
        helper.export_to_labelme(args.label_dir, args.image_dir, args.output)
    elif args.action == "visualize":
        helper.visualize_annotations(args.image_dir, args.label_dir, args.num)
    elif args.action == "count":
        helper.count_annotations(args.label_dir)


if __name__ == "__main__":
    main()
