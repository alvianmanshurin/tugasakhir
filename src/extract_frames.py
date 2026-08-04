"""Video Frame Extraction for Vehicle Detection Dataset
Extract frames dari video KIRI/TENGAH di gerbang masuk Kampus ITERA
"""

import os
import cv2
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class VideoFrameExtractor:
    """Extract frames dari video untuk dataset."""

    def __init__(self, config):
        self.config = config
        self.video_source = config["dataset"]["video_source"]
        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_videos(self):
        """List semua video yang tersedia."""
        video_dir = Path(self.video_source)
        if not video_dir.exists():
            print(f"[ERROR] Video directory not found: {video_dir}")
            return []

        videos = list(video_dir.glob("*.MOV")) + list(video_dir.glob("*.mp4"))
        print(f"\n[INFO] Found {len(videos)} videos in: {video_dir}")
        for v in videos:
            size_mb = v.stat().st_size / (1024*1024)
            print(f"  - {v.name} ({size_mb:.1f} MB)")

        return videos

    def get_video_info(self, video_path):
        """Get video information."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0

        cap.release()
        return info

    def extract_frames(self, video_path, output_dir=None, interval=30,
                       max_frames=None, resize=None):
        """
        Extract frames dari video.

        Args:
            video_path: Path ke video
            output_dir: Output directory
            interval: Extract every N frames
            max_frames: Maximum frames to extract
            resize: Resize tuple (width, height) or None
        """
        video_path = Path(video_path)
        if output_dir:
            out_dir = Path(output_dir)
        else:
            # Create subdirectory based on video name
            out_dir = self.output_dir / video_path.stem

        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"EXTRACTING FRAMES")
        print(f"{'='*60}")
        print(f"Video: {video_path.name}")

        # Get video info
        info = self.get_video_info(video_path)
        if info:
            print(f"Resolution: {info['width']}x{info['height']}")
            print(f"FPS: {info['fps']:.1f}")
            print(f"Duration: {info['duration']:.1f}s")
            print(f"Total frames: {info['total_frames']}")

        print(f"Extract every: {interval} frames")
        print(f"Output: {out_dir}")

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {video_path}")
            return []

        frame_count = 0
        extracted = 0
        extracted_files = []
        start_time = time.time()

        # Count existing files to continue numbering
        existing = len(list(out_dir.glob("*.jpg")))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                # Resize if specified
                if resize:
                    frame = cv2.resize(frame, resize)

                # Save frame
                filename = f"{video_path.stem}_{existing + extracted:05d}.jpg"
                save_path = out_dir / filename
                cv2.imwrite(str(save_path), frame)
                extracted_files.append(save_path)
                extracted += 1

                # Progress
                if extracted % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_extract = extracted / elapsed if elapsed > 0 else 0
                    print(f"  Extracted: {extracted} frames | FPS: {fps_extract:.1f}")

                # Check max frames
                if max_frames and extracted >= max_frames:
                    print(f"  Reached max frames: {max_frames}")
                    break

            frame_count += 1

        cap.release()
        elapsed = time.time() - start_time

        print(f"\n[OK] Extracted {extracted} frames to: {out_dir}")
        print(f"[OK] Time: {elapsed:.1f}s")

        return extracted_files

    def extract_all_videos(self, interval=30, max_frames_per_video=500, resize=None):
        """Extract frames dari semua video."""
        videos = self.list_videos()

        if not videos:
            print("[ERROR] No videos found")
            return

        print(f"\n[INFO] Processing {len(videos)} videos...")
        print(f"[INFO] Interval: every {interval} frames")
        print(f"[INFO] Max frames per video: {max_frames_per_video}")

        total_extracted = 0

        for video in videos:
            print(f"\n{'='*60}")
            print(f"Processing: {video.name}")
            print(f"{'='*60}")

            files = self.extract_frames(
                video,
                interval=interval,
                max_frames=max_frames_per_video,
                resize=resize,
            )
            total_extracted += len(files)

        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total videos: {len(videos)}")
        print(f"Total frames extracted: {total_extracted}")
        print(f"Output directory: {self.output_dir}")

        # Create summary
        self._create_summary(videos, total_extracted)

    def _create_summary(self, videos, total_frames):
        """Create extraction summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "videos": [v.name for v in videos],
            "total_frames": total_frames,
            "output_dir": str(self.output_dir),
        }

        summary_path = self.output_dir / "extraction_summary.yaml"
        with open(summary_path, "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        print(f"\n[INFO] Summary saved: {summary_path}")

    def split_to_trainval(self, source_dir=None, train_ratio=0.8):
        """Split extracted frames ke train/val."""
        import random
        import shutil

        source = Path(source_dir) if source_dir else self.output_dir
        images = list(source.glob("*.jpg"))

        if not images:
            print(f"[ERROR] No images found in: {source}")
            return

        print(f"\n[SPLIT] Total images: {len(images)}")
        print(f"[SPLIT] Train ratio: {train_ratio}")

        # Shuffle
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)

        train_files = images[:split_idx]
        val_files = images[split_idx:]

        # Directories
        train_img = Path("data/annotated/images/train")
        val_img = Path("data/annotated/images/val")
        train_lbl = Path("data/annotated/labels/train")
        val_lbl = Path("data/annotated/labels/val")

        for d in [train_img, val_img, train_lbl, val_lbl]:
            d.mkdir(parents=True, exist_ok=True)

        # Copy
        for f in train_files:
            shutil.copy2(f, train_img / f.name)
            (train_lbl / (f.stem + ".txt")).touch()

        for f in val_files:
            shutil.copy2(f, val_img / f.name)
            (val_lbl / (f.stem + ".txt")).touch()

        print(f"[OK] Train: {len(train_files)} images -> {train_img}")
        print(f"[OK] Val: {len(val_files)} images -> {val_img}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video for vehicle detection dataset"
    )
    parser.add_argument("--action", type=str, default="extract",
                       choices=["list", "info", "extract", "extract-all", "split"],
                       help="Action to perform")
    parser.add_argument("--video", type=str, help="Video file path")
    parser.add_argument("--interval", type=int, default=30,
                       help="Extract every N frames (default: 30)")
    parser.add_argument("--max-frames", type=int, default=500,
                       help="Max frames per video")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--resize", type=int, nargs=2, default=None,
                       metavar=("WIDTH", "HEIGHT"), help="Resize frames")
    parser.add_argument("--ratio", type=float, default=0.8,
                       help="Train split ratio")
    args = parser.parse_args()

    config = load_config()
    extractor = VideoFrameExtractor(config)

    if args.action == "list":
        extractor.list_videos()

    elif args.action == "info":
        if args.video:
            info = extractor.get_video_info(args.video)
            if info:
                print(f"\nVideo Info:")
                for k, v in info.items():
                    print(f"  {k}: {v}")
        else:
            print("[ERROR] Please specify --video")

    elif args.action == "extract":
        if args.video:
            extractor.extract_frames(
                args.video,
                output_dir=args.output,
                interval=args.interval,
                max_frames=args.max_frames,
                resize=tuple(args.resize) if args.resize else None,
            )
        else:
            print("[ERROR] Please specify --video")

    elif args.action == "extract-all":
        extractor.extract_all_videos(
            interval=args.interval,
            max_frames_per_video=args.max_frames,
            resize=tuple(args.resize) if args.resize else None,
        )

    elif args.action == "split":
        extractor.split_to_trainval(args.output, args.ratio)


if __name__ == "__main__":
    main()
