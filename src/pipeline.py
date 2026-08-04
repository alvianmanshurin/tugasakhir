"""Complete Pipeline for Vehicle Detection System
Automates: Extract → Split → Annotate → Train → Evaluate
Optimized for Intel i3-1115G4, 8GB RAM
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class Pipeline:
    """Complete pipeline for vehicle detection."""

    def __init__(self, config):
        self.config = config
        self.video_source = config["dataset"]["video_source"]

    def show_status(self):
        """Show current project status."""
        print("\n" + "=" * 70)
        print("PROJECT STATUS - Vehicle Detection System")
        print("=" * 70)

        # Check videos
        video_dir = Path(self.video_source)
        videos = list(video_dir.glob("*.MOV")) if video_dir.exists() else []
        print(f"\n[VIDEOS] {len(videos)} files in {self.video_source}")
        for v in videos:
            print(f"  - {v.name}")

        # Check raw frames
        raw_dir = Path("data/raw")
        raw_frames = list(raw_dir.glob("*.jpg")) if raw_dir.exists() else []
        print(f"\n[RAW FRAMES] {len(raw_frames)} images in data/raw/")

        # Check train/val
        train_imgs = list(Path("data/annotated/images/train").glob("*.jpg"))
        val_imgs = list(Path("data/annotated/images/val").glob("*.jpg"))
        train_lbls = list(Path("data/annotated/labels/train").glob("*.txt"))
        val_lbls = list(Path("data/annotated/labels/val").glob("*.txt"))

        print(f"\n[DATASET]")
        print(f"  Train images: {len(train_imgs)}")
        print(f"  Train labels: {len(train_lbls)}")
        print(f"  Val images:   {len(val_imgs)}")
        print(f"  Val labels:   {len(val_lbls)}")

        # Check annotated (non-empty labels)
        annotated = 0
        for lbl in train_lbls + val_lbls:
            if lbl.stat().st_size > 0:
                annotated += 1
        print(f"  Annotated:    {annotated}")

        # Check model
        model_path = Path("models/vehicle_detection/weights/best.pt")
        print(f"\n[MODEL]")
        print(f"  Best model: {'EXISTS' if model_path.exists() else 'NOT FOUND'}")

        # Check outputs
        det_dir = Path("outputs/detections")
        detections = list(det_dir.glob("*.jpg")) if det_dir.exists() else []
        print(f"\n[OUTPUTS]")
        print(f"  Detections: {len(detections)}")

        print("\n" + "=" * 70)

    def step1_extract(self, interval=30, max_frames=500):
        """Step 1: Extract frames from videos."""
        print("\n" + "=" * 70)
        print("STEP 1: EXTRACT FRAMES FROM VIDEOS")
        print("=" * 70)

        from src.extract_frames import VideoFrameExtractor
        extractor = VideoFrameExtractor(self.config)
        extractor.extract_all_videos(
            interval=interval,
            max_frames_per_video=max_frames,
        )

    def step2_split(self, train_ratio=0.8):
        """Step 2: Split to train/val."""
        print("\n" + "=" * 70)
        print("STEP 2: SPLIT TO TRAIN/VAL")
        print("=" * 70)

        from src.extract_frames import VideoFrameExtractor
        extractor = VideoFrameExtractor(self.config)
        extractor.split_to_trainval(train_ratio=train_ratio)

    def step3_annotate_guide(self):
        """Step 3: Show annotation guide."""
        print("\n" + "=" * 70)
        print("STEP 3: ANNOTATION")
        print("=" * 70)

        print("""
[INSTRUCTIONS]

1. Install LabelImg:
   python setup_labelimg.py --action install

2. Launch LabelImg:
   python setup_labelimg.py --action launch

3. In LabelImg:
   - Click "Open Dir" -> select data/annotated/images/train
   - Click "Change Save Dir" -> select data/annotated/labels/train
   - Make sure "YOLO" format is selected (bottom bar)
   - Press "W" to create bounding box
   - Draw around vehicle and select class:
     * motor (class 0)
     * mobil (class 1)
     * bus (class 2)
     * truk (class 3)
   - Press "D" for next image
   - Press Ctrl+S to save

4. Repeat for val set:
   - Open Dir -> data/annotated/images/val
   - Change Save Dir -> data/annotated/labels/val

5. After annotation, validate:
   python src/dataset_prepare.py --action validate
""")

    def step4_train(self, quick=False):
        """Step 4: Train model."""
        print("\n" + "=" * 70)
        print("STEP 4: TRAIN MODEL")
        print("=" * 70)

        if quick:
            os.system("python src/train.py --quick")
        else:
            os.system("python src/train.py")

    def step5_evaluate(self):
        """Step 5: Evaluate model."""
        print("\n" + "=" * 70)
        print("STEP 5: EVALUATE MODEL")
        print("=" * 70)

        os.system("python src/evaluate.py")

    def step6_detect(self, source="data/raw"):
        """Step 6: Run detection."""
        print("\n" + "=" * 70)
        print("STEP 6: DETECTION")
        print("=" * 70)

        os.system(f"python src/detect.py --source {source}")

    def run_full_pipeline(self, quick=False):
        """Run complete pipeline."""
        print("\n" + "=" * 70)
        print("COMPLETE PIPELINE - Vehicle Detection")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        start = time.time()

        # Step 1: Extract
        self.step1_extract(interval=30, max_frames=500)

        # Step 2: Split
        self.step2_split()

        # Step 3: Annotation guide
        self.step3_annotate_guide()

        print("\n" + "=" * 70)
        print("PIPELINE PAUSED")
        print("=" * 70)
        print("""
Next steps require manual annotation:

1. Annotate images with LabelImg:
   python setup_labelimg.py --action launch

2. After annotation, continue with:
   python src/pipeline.py --step train

Or run steps individually:
   python src/pipeline.py --step extract
   python src/pipeline.py --step split
   python src/pipeline.py --step train
   python src/pipeline.py --step evaluate
   python src/pipeline.py --step detect
""")

    def run_from_train(self, quick=False):
        """Run from training step (after annotation)."""
        print("\n" + "=" * 70)
        print("RUNNING FROM TRAINING STEP")
        print("=" * 70)

        # Validate first
        print("\n[STEP 0] Validating dataset...")
        os.system("python src/dataset_prepare.py --action validate")

        # Train
        self.step4_train(quick=quick)

        # Evaluate
        self.step5_evaluate()

        elapsed = time.time() - start
        print(f"\n[INFO] Total time: {elapsed/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(
        description="Complete Pipeline for Vehicle Detection"
    )
    parser.add_argument("--action", type=str, default="status",
                       choices=["status", "full", "extract", "split", "annotate",
                               "train", "evaluate", "detect", "from-train"],
                       help="Pipeline action")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode (10 epochs)")
    parser.add_argument("--interval", type=int, default=30,
                       help="Frame extraction interval")
    parser.add_argument("--max-frames", type=int, default=500,
                       help="Max frames per video")
    args = parser.parse_args()

    config = load_config()
    pipeline = Pipeline(config)

    if args.action == "status":
        pipeline.show_status()

    elif args.action == "full":
        pipeline.run_full_pipeline(quick=args.quick)

    elif args.action == "extract":
        pipeline.step1_extract(interval=args.interval, max_frames=args.max_frames)

    elif args.action == "split":
        pipeline.step2_split()

    elif args.action == "annotate":
        pipeline.step3_annotate_guide()

    elif args.action == "train":
        pipeline.step4_train(quick=args.quick)

    elif args.action == "evaluate":
        pipeline.step5_evaluate()

    elif args.action == "detect":
        pipeline.step6_detect()

    elif args.action == "from-train":
        pipeline.run_from_train(quick=args.quick)


if __name__ == "__main__":
    main()
