"""Simple GUI for Vehicle Detection System
Optimized for Intel i3-1115G4, 8GB RAM
"""

import os
import sys
import yaml
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import cv2
from pathlib import Path
from PIL import Image, ImageTk


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class VehicleDetectionGUI:
    """GUI Application for Vehicle Detection System."""

    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Detection System - UPT K3L ITERA")
        self.root.geometry("900x700")
        self.root.configure(bg="#2b2b2b")

        # Config
        self.config = load_config()
        self.class_names = self.config["dataset"]["names"]

        # Variables
        self.source_path = tk.StringVar()
        self.model_path = tk.StringVar(value="models/vehicle_detection/weights/best.pt")
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.is_processing = False

        # Colors
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#4a9eff"
        self.success_color = "#4caf50"
        self.warning_color = "#ff9800"

        self._create_widgets()

    def _create_widgets(self):
        """Create GUI widgets."""
        # Title
        title_frame = tk.Frame(self.root, bg=self.accent_color, height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        tk.Label(title_frame, text="VEHICLE DETECTION SYSTEM",
                font=("Arial", 16, "bold"), bg=self.accent_color,
                fg="white").pack(pady=15)

        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg="#3c3c3c", width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Model section
        tk.Label(left_panel, text="MODEL", font=("Arial", 10, "bold"),
                bg="#3c3c3c", fg=self.accent_color).pack(pady=(10, 5), anchor=tk.W, padx=10)

        tk.Label(left_panel, text="Model Path:", bg="#3c3c3c",
                fg=self.fg_color).pack(anchor=tk.W, padx=10)
        model_frame = tk.Frame(left_panel, bg="#3c3c3c")
        model_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Entry(model_frame, textvariable=self.model_path, width=25).pack(side=tk.LEFT)
        tk.Button(model_frame, text="Browse", command=self._browse_model).pack(side=tk.LEFT, padx=5)

        tk.Label(left_panel, text=f"Confidence:", bg="#3c3c3c",
                fg=self.fg_color).pack(anchor=tk.W, padx=10)
        conf_scale = tk.Scale(left_panel, from_=0.1, to=1.0, resolution=0.05,
                             variable=self.conf_threshold, orient=tk.HORIZONTAL,
                             bg="#3c3c3c", fg=self.fg_color, troughcolor="#4a4a4a")
        conf_scale.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Source section
        tk.Label(left_panel, text="SOURCE", font=("Arial", 10, "bold"),
                bg="#3c3c3c", fg=self.accent_color).pack(pady=(10, 5), anchor=tk.W, padx=10)

        tk.Label(left_panel, text="Image/Video Path:", bg="#3c3c3c",
                fg=self.fg_color).pack(anchor=tk.W, padx=10)
        source_frame = tk.Frame(left_panel, bg="#3c3c3c")
        source_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Entry(source_frame, textvariable=self.source_path, width=25).pack(side=tk.LEFT)
        tk.Button(source_frame, text="Browse", command=self._browse_source).pack(side=tk.LEFT, padx=5)

        # Buttons
        btn_frame = tk.Frame(left_panel, bg="#3c3c3c")
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        self.detect_btn = tk.Button(btn_frame, text="DETECT",
                                   command=self._run_detection,
                                   bg=self.accent_color, fg="white",
                                   font=("Arial", 10, "bold"), height=2)
        self.detect_btn.pack(fill=tk.X, pady=(0, 5))

        self.webcam_btn = tk.Button(btn_frame, text="WEBCAM",
                                   command=self._run_webcam,
                                   bg=self.warning_color, fg="white",
                                   font=("Arial", 10, "bold"), height=2)
        self.webcam_btn.pack(fill=tk.X, pady=(0, 5))

        self.stop_btn = tk.Button(btn_frame, text="STOP",
                                 command=self._stop_processing,
                                 bg="#f44336", fg="white",
                                 font=("Arial", 10, "bold"), height=2,
                                 state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X)

        # Info section
        tk.Label(left_panel, text="INFO", font=("Arial", 10, "bold"),
                bg="#3c3c3c", fg=self.accent_color).pack(pady=(15, 5), anchor=tk.W, padx=10)

        self.info_text = tk.Text(left_panel, height=8, bg="#2b2b2b",
                                fg=self.fg_color, font=("Consolas", 9))
        self.info_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        self._update_info("System ready.\nSelect source and click DETECT.")

        # Right panel - Display
        right_panel = tk.Frame(main_frame, bg="#3c3c3c")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right_panel, text="DISPLAY", font=("Arial", 10, "bold"),
                bg="#3c3c3c", fg=self.accent_color).pack(pady=(10, 5))

        self.display_label = tk.Label(right_panel, bg="#1e1e1e",
                                     text="No image/video loaded",
                                     fg="#666666", font=("Arial", 12))
        self.display_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Results section
        tk.Label(right_panel, text="RESULTS", font=("Arial", 10, "bold"),
                bg="#3c3c3c", fg=self.accent_color).pack(anchor=tk.W, padx=10)

        self.results_text = scrolledtext.ScrolledText(right_panel, height=8,
                                                     bg="#1e1e1e", fg=self.fg_color,
                                                     font=("Consolas", 9))
        self.results_text.pack(fill=tk.X, padx=10, pady=(0, 10))

        # Status bar
        status_frame = tk.Frame(self.root, bg="#1e1e1e", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)

        self.status_label = tk.Label(status_frame, text="Ready",
                                    bg="#1e1e1e", fg=self.fg_color,
                                    font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, padx=10)

        self.fps_label = tk.Label(status_frame, text="FPS: --",
                                 bg="#1e1e1e", fg=self.success_color,
                                 font=("Arial", 9))
        self.fps_label.pack(side=tk.RIGHT, padx=10)

    def _browse_model(self):
        """Browse model file."""
        path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)

    def _browse_source(self):
        """Browse source file/directory."""
        path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("Video files", "*.mp4 *.avi *.mov"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.source_path.set(path)

    def _update_info(self, text):
        """Update info text."""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)

    def _update_results(self, text):
        """Update results text."""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)

    def _update_status(self, text):
        """Update status bar."""
        self.status_label.config(text=text)

    def _run_detection(self):
        """Run detection in thread."""
        source = self.source_path.get()
        if not source or not os.path.exists(source):
            messagebox.showerror("Error", "Please select a valid source.")
            return

        self.is_processing = True
        self.detect_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._update_status("Processing...")

        thread = threading.Thread(target=self._detect_thread, args=(source,))
        thread.daemon = True
        thread.start()

    def _detect_thread(self, source):
        """Detection thread."""
        try:
            from ultralytics import YOLO
            import time

            # Load model
            model_path = self.model_path.get()
            self._update_info(f"Loading model:\n{model_path}\n\n")
            model = YOLO(model_path)

            source_path = Path(source)

            if source_path.is_file():
                # Single image
                img = cv2.imread(source)
                if img is None:
                    self._update_status("Error: Cannot read image")
                    return

                start = time.time()
                results = model(img, conf=self.conf_threshold.get(),
                              imgsz=self.config["model"]["input_size"],
                              verbose=False)
                inference_time = time.time() - start

                # Process results
                counts = {name: 0 for name in self.class_names.values()}
                detections = []

                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].tolist()

                            if cls_id < len(self.class_names):
                                cls_name = self.class_names[cls_id]
                                counts[cls_name] += 1
                                detections.append({
                                    "class": cls_name,
                                    "confidence": conf
                                })

                # Draw on image
                img_result = img.copy()
                colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]

                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            xyxy = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = [int(c) for c in xyxy]
                            color = colors[cls_id % len(colors)]
                            cv2.rectangle(img_result, (x1,y1), (x2,y2), color, 2)

                # Display
                self._display_image(img_result)

                # Results
                total = sum(counts.values())
                fps = 1/inference_time if inference_time > 0 else 0
                self.fps_label.config(text=f"FPS: {fps:.1f}")

                result_text = f"Detection Results\n{'='*30}\n"
                result_text += f"Total: {total} vehicles\n\n"
                for name, count in counts.items():
                    if count > 0:
                        result_text += f"{name}: {count}\n"
                result_text += f"\nInference: {inference_time*1000:.0f}ms"
                result_text += f"\nFPS: {fps:.1f}"

                self._update_results(result_text)
                self._update_status("Detection complete")

            elif source_path.is_dir():
                # Directory
                exts = ["*.jpg", "*.jpeg", "*.png"]
                images = []
                for ext in exts:
                    images.extend(source_path.glob(ext))

                self._update_info(f"Processing {len(images)} images...\n\n")

                total_counts = {name: 0 for name in self.class_names.values()}
                total_vehicles = 0

                for idx, img_path in enumerate(sorted(images), 1):
                    if not self.is_processing:
                        break

                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    results = model(img, conf=self.conf_threshold.get(),
                                  imgsz=self.config["model"]["input_size"],
                                  verbose=False)

                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                cls_id = int(box.cls[0])
                                if cls_id < len(self.class_names):
                                    total_counts[self.class_names[cls_id]] += 1
                                    total_vehicles += 1

                    if idx % 5 == 0:
                        self._update_info(f"Processing: {idx}/{len(images)}\n"
                                         f"Vehicles found: {total_vehicles}")

                result_text = f"Batch Results\n{'='*30}\n"
                result_text += f"Images: {len(images)}\n"
                result_text += f"Total vehicles: {total_vehicles}\n\n"
                for name, count in total_counts.items():
                    if count > 0:
                        result_text += f"{name}: {count}\n"

                self._update_results(result_text)
                self._update_status("Batch processing complete")

        except Exception as e:
            self._update_status(f"Error: {str(e)}")
            messagebox.showerror("Error", str(e))

        finally:
            self.is_processing = False
            self.detect_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _run_webcam(self):
        """Run webcam detection."""
        self.is_processing = True
        self.detect_btn.config(state=tk.DISABLED)
        self.webcam_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._update_status("Webcam active...")

        thread = threading.Thread(target=self._webcam_thread)
        thread.daemon = True
        thread.start()

    def _webcam_thread(self):
        """Webcam thread."""
        try:
            from ultralytics import YOLO
            import time

            model = YOLO(self.model_path.get())
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                self._update_status("Error: Cannot open webcam")
                return

            self._update_info("Webcam active\nPress 'q' in video window to stop\n")

            fps_history = []

            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break

                start = time.time()
                results = model(frame, conf=self.conf_threshold.get(),
                              imgsz=self.config["model"]["input_size"],
                              verbose=False)
                inference_time = time.time() - start

                # Draw
                colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
                counts = {name: 0 for name in self.class_names.values()}

                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = [int(c) for c in xyxy]

                            if cls_id < len(self.class_names):
                                cls_name = self.class_names[cls_id]
                                counts[cls_name] += 1
                                color = colors[cls_id % len(colors)]
                                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                                label = f"{cls_name} {conf:.2f}"
                                cv2.putText(frame, label, (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # FPS
                fps = 1/inference_time if inference_time > 0 else 0
                fps_history.append(fps)
                avg_fps = sum(fps_history[-30:]) / min(len(fps_history), 30)

                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                total = sum(counts.values())
                cv2.putText(frame, f"Vehicles: {total}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                self._display_cv_image(frame)
                self.fps_label.config(text=f"FPS: {avg_fps:.1f}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            self._update_status("Webcam stopped")

        except Exception as e:
            self._update_status(f"Error: {str(e)}")

        finally:
            self.is_processing = False
            self.detect_btn.config(state=tk.NORMAL)
            self.webcam_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _display_image(self, cv_img):
        """Display OpenCV image in label."""
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # Resize to fit
        display_w = self.display_label.winfo_width()
        display_h = self.display_label.winfo_height()
        if display_w > 1 and display_h > 1:
            pil_img.thumbnail((display_w, display_h), Image.Resampling.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil_img)
        self.display_label.config(image=tk_img, text="")
        self.display_label.image = tk_img

    def _display_cv_image(self, cv_img):
        """Display OpenCV image (thread-safe)."""
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize((640, 480), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)

        self.root.after(0, lambda: self._update_display_label(tk_img))

    def _update_display_label(self, tk_img):
        """Update display label (main thread)."""
        self.display_label.config(image=tk_img, text="")
        self.display_label.image = tk_img

    def _stop_processing(self):
        """Stop processing."""
        self.is_processing = False
        self._update_status("Stopping...")


def main():
    root = tk.Tk()
    app = VehicleDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
