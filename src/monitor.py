"""Training Monitor - Real-time training progress display
Optimized for low-end laptop
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime


class TrainingMonitor:
    """Monitor training progress from log files."""

    def __init__(self, log_dir="models/vehicle_detection"):
        self.log_dir = Path(log_dir)
        self.results_file = self.log_dir / "results.csv"

    def watch(self, refresh=5):
        """Watch training progress in real-time."""
        print("=" * 60)
        print("TRAINING MONITOR")
        print("Press Ctrl+C to stop")
        print("=" * 60)

        if not self.results_file.exists():
            print(f"\n[INFO] Waiting for training to start...")
            print(f"[INFO] Looking for: {self.results_file}")

            while not self.results_file.exists():
                time.sleep(2)

            print("[INFO] Training started!")

        last_line = 0
        try:
            while True:
                if self.results_file.exists():
                    with open(self.results_file, "r") as f:
                        lines = f.readlines()

                    if len(lines) > last_line:
                        # Clear screen
                        os.system('cls' if os.name == 'nt' else 'clear')

                        print("=" * 60)
                        print("TRAINING MONITOR - UPT K3L ITERA")
                        print(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
                        print("=" * 60)

                        # Parse and display latest results
                        for line in lines[last_line:]:
                            if line.strip() and not line.startswith("epoch"):
                                parts = line.strip().split(",")
                                if len(parts) >= 8:
                                    try:
                                        epoch = parts[0].strip()
                                        box_loss = parts[1].strip()
                                        cls_loss = parts[2].strip()
                                        dfl_loss = parts[3].strip()
                                        mAP50 = parts[5].strip()
                                        mAP5095 = parts[6].strip()

                                        print(f"\nEpoch: {epoch}")
                                        print(f"  Box Loss:    {box_loss}")
                                        print(f"  Cls Loss:    {cls_loss}")
                                        print(f"  DFL Loss:    {dfl_loss}")
                                        print(f"  mAP50:       {mAP50}")
                                        print(f"  mAP50-95:    {mAP5095}")
                                    except:
                                        pass

                        last_line = len(lines)

                        # Progress bar
                        print("\n" + "-" * 60)
                        print("Watching for updates... (Ctrl+C to stop)")

                time.sleep(refresh)

        except KeyboardInterrupt:
            print("\n\n[INFO] Monitor stopped.")

    def show_summary(self):
        """Show training summary."""
        if not self.results_file.exists():
            print("[ERROR] No training results found.")
            return

        with open(self.results_file, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            print("[INFO] No training data yet.")
            return

        print("=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)

        # Parse all results
        results = []
        for line in lines[1:]:
            if line.strip():
                parts = line.strip().split(",")
                if len(parts) >= 7:
                    results.append({
                        "epoch": int(parts[0]),
                        "box_loss": float(parts[1]),
                        "cls_loss": float(parts[2]),
                        "dfl_loss": float(parts[3]),
                        "mAP50": float(parts[5]),
                        "mAP5095": float(parts[6]),
                    })

        if not results:
            print("[INFO] No valid results found.")
            return

        # Find best
        best = max(results, key=lambda x: x["mAP50"])
        latest = results[-1]

        print(f"\nTotal epochs completed: {len(results)}")
        print(f"\nLatest (Epoch {latest['epoch']}):")
        print(f"  mAP50:       {latest['mAP50']:.4f}")
        print(f"  mAP50-95:    {latest['mAP5095']:.4f}")

        print(f"\nBest (Epoch {best['epoch']}):")
        print(f"  mAP50:       {best['mAP50']:.4f}")
        print(f"  mAP50-95:    {best['mAP5095']:.4f}")

        print(f"\nTraining history:")
        print(f"  {'Epoch':<8} {'mAP50':<12} {'mAP50-95':<12}")
        print(f"  {'-'*32}")
        for r in results[::max(1, len(results)//10)]:  # Show ~10 samples
            print(f"  {r['epoch']:<8} {r['mAP50']:<12.4f} {r['mAP5095']:<12.4f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Training Monitor")
    parser.add_argument("--action", type=str, default="watch",
                       choices=["watch", "summary"])
    parser.add_argument("--log-dir", type=str, default="models/vehicle_detection")
    parser.add_argument("--refresh", type=int, default=5)
    args = parser.parse_args()

    monitor = TrainingMonitor(args.log_dir)

    if args.action == "watch":
        monitor.watch(args.refresh)
    else:
        monitor.show_summary()


if __name__ == "__main__":
    main()
