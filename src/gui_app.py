"""
GUI aplikasi deteksi kendaraan (Tkinter).

Aturan threading yang dipegang file ini:

  * Widget, StringVar/DoubleVar, PhotoImage, dan messagebox HANYA boleh
    disentuh di main thread (thread yang menjalankan ``root.mainloop()``).
  * Worker thread hanya boleh: membuat pipeline, membaca frame, menjalankan
    inferensi, menulis ke database, dan menaruh frame ke ``queue.Queue``.
  * Main thread Retire queue lewat ``root.after()``.

Versi sebelumnya melanggar aturan ini: ``self.conf_threshold.get()``,
``self.model_path.get()``, ``self.fps_label.config()``, dan
``messagebox.showerror()`` dipanggil dari worker thread. Tkinter tidak
thread-safe - pemanggilannya bisa membuat GUI hang atau crash tanpa
jejak. Selain itu, penghitung naïf menambah ``counts[cls] += 1`` di dalam
loop frame, sehingga satu kendaraan yang terlihat 200 frame dilaporkan
sebagai "200 kendaraan".
"""

import argparse
import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

import cv2
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import VehiclePipeline
from query_db import export_csv
from utils.database import DetectionDatabase
from utils.paths import DB_PATH, OUTPUT_DIR, get_model_path, load_config

# Diisi CLI; None = pakai config/config.yaml
_CONFIG_OVERRIDE: Optional[str] = None

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class VehicleDetectionGUI:
    """GUI aplikasi deteksi kendaraan."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Vehicle Detection System - UPT K3L ITERA")
        self.root.geometry("1180x760")
        self.root.configure(bg="#2b2b2b")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- Config (dibaca sekali di main thread) -----------------------
        self.config = load_config(_CONFIG_OVERRIDE)
        model_cfg = self.config.get("model", {})
        self.class_names = dict(self.config.get("dataset", {}).get("names") or
                                {0: "motor", 1: "mobil", 2: "bus", 3: "truk"})

        # --- Variables Tk (hanya disentuh di main thread) ----------------
        self.source_path = tk.StringVar()
        try:
            default_model = get_model_path(self.config)
        except FileNotFoundError:
            default_model = model_cfg.get("best_weights", "")
        self.model_path = tk.StringVar(value=default_model)
        self.conf_threshold = tk.DoubleVar(
            value=float(model_cfg.get("confidence_threshold", 0.5))
        )
        self.rtsp_url = tk.StringVar()
        self.save_to_db = tk.BooleanVar(value=False)
        self.is_processing = False

        # --- State antar thread ------------------------------------------
        self._frame_queue: "queue.Queue" = queue.Queue(maxsize=2)
        self._pending: dict = {}
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        # Disimpan sebagai atribut supaya PhotoImage tidak di-GC oleh Tk
        self._tk_img = None

        # --- Warna --------------------------------------------------------
        self.bg_color = "#2b2b2b"
        self.fg_color = "#ffffff"
        self.accent_color = "#4a9eff"
        self.success_color = "#4caf50"
        self.warning_color = "#ff9800"

        self._create_widgets()
        self._drain_queue()  # mulai pompa queue

    # ------------------------------------------------------------------
    # Widgets
    # ------------------------------------------------------------------

    def _create_widgets(self) -> None:
        title_frame = tk.Frame(self.root, bg=self.accent_color, height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        tk.Label(title_frame, text="VEHICLE DETECTION SYSTEM",
                 font=("Arial", 16, "bold"), bg=self.accent_color,
                 fg="white").pack(pady=15)

        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = tk.Frame(main_frame, bg="#3c3c3c", width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        right_panel = tk.Frame(main_frame, bg="#3c3c3c")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Panel kiri: model & threshold -------------------------------
        tk.Label(left_panel, text="MODEL", font=("Arial", 10, "bold"),
                 bg="#3c3c3c", fg=self.accent_color).pack(pady=(10, 5),
                                                          anchor=tk.W, padx=10)
        tk.Label(left_panel, text="Model Path:", bg="#3c3c3c",
                 fg=self.fg_color).pack(anchor=tk.W, padx=10)
        model_frame = tk.Frame(left_panel, bg="#3c3c3c")
        model_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Entry(model_frame, textvariable=self.model_path).pack(side=tk.LEFT,
                                                                 fill=tk.X,
                                                                 expand=True)
        tk.Button(model_frame, text="Browse",
                  command=self._browse_model).pack(side=tk.LEFT, padx=5)

        tk.Label(left_panel, text="Confidence (track baru):", bg="#3c3c3c",
                 fg=self.fg_color).pack(anchor=tk.W, padx=10)
        tk.Scale(left_panel, from_=0.1, to=1.0, resolution=0.05,
                 variable=self.conf_threshold, orient=tk.HORIZONTAL,
                 bg="#3c3c3c", fg=self.fg_color, troughcolor="#4a4a4a",
                 highlightthickness=0).pack(fill=tk.X, padx=10, pady=(0, 2))
        tk.Label(left_panel, text="< 0.25 = confidence rendah hanya untuk "
                                  "menolong track yang hilang",
                 bg="#3c3c3c", fg="#9a9a9a", font=("Arial", 8),
                 wraplength=290, justify=tk.LEFT).pack(anchor=tk.W, padx=10,
                                                        pady=(0, 10))

        # --- Panel kiri: sumber ------------------------------------------
        tk.Label(left_panel, text="SOURCE", font=("Arial", 10, "bold"),
                 bg="#3c3c3c", fg=self.accent_color).pack(pady=(10, 5),
                                                          anchor=tk.W, padx=10)
        tk.Label(left_panel, text="Image / Video / Folder:", bg="#3c3c3c",
                 fg=self.fg_color).pack(anchor=tk.W, padx=10)
        source_frame = tk.Frame(left_panel, bg="#3c3c3c")
        source_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Entry(source_frame, textvariable=self.source_path).pack(side=tk.LEFT,
                                                                   fill=tk.X,
                                                                   expand=True)
        tk.Button(source_frame, text="Browse",
                  command=self._browse_source).pack(side=tk.LEFT, padx=5)

        tk.Checkbutton(left_panel, text="Simpan hasil ke database",
                       variable=self.save_to_db, bg="#3c3c3c", fg=self.fg_color,
                       selectcolor="#3c3c3c", activebackground="#3c3c3c",
                       activeforeground=self.fg_color,
                       font=("Arial", 9)).pack(anchor=tk.W, padx=10, pady=(5, 10))

        # --- Panel kiri: CCTV --------------------------------------------
        tk.Label(left_panel, text="CCTV / RTSP", font=("Arial", 10, "bold"),
                 bg="#3c3c3c", fg=self.warning_color).pack(pady=(10, 5),
                                                           anchor=tk.W, padx=10)
        tk.Label(left_panel, text="RTSP URL:", bg="#3c3c3c",
                 fg=self.fg_color).pack(anchor=tk.W, padx=10)
        rtsp_frame = tk.Frame(left_panel, bg="#3c3c3c")
        rtsp_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        tk.Entry(rtsp_frame, textvariable=self.rtsp_url).pack(side=tk.LEFT,
                                                              fill=tk.X,
                                                              expand=True)
        self.cctv_btn = tk.Button(left_panel, text="CCTV", command=self._run_cctv,
                                  bg="#ff5722", fg="white",
                                  font=("Arial", 10, "bold"))
        self.cctv_btn.pack(fill=tk.X, padx=10, pady=(0, 10))

        # --- Panel kiri: tombol aksi ------------------------------------
        btn_frame = tk.Frame(left_panel, bg="#3c3c3c")
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        self.detect_btn = tk.Button(btn_frame, text="DETECT",
                                    command=self._run_detection, bg=self.accent_color,
                                    fg="white", font=("Arial", 10, "bold"), height=2)
        self.detect_btn.pack(fill=tk.X, pady=(0, 5))
        self.webcam_btn = tk.Button(btn_frame, text="WEBCAM",
                                    command=self._run_webcam, bg=self.warning_color,
                                    fg="white", font=("Arial", 10, "bold"), height=2)
        self.webcam_btn.pack(fill=tk.X, pady=(0, 5))
        self.stop_btn = tk.Button(btn_frame, text="STOP",
                                  command=self._stop_processing, bg="#f44336",
                                  fg="white", font=("Arial", 10, "bold"), height=2,
                                  state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X)
        self.database_btn = tk.Button(btn_frame, text="DATABASE",
                                      command=self._open_db_tab,
                                      bg=self.success_color, fg="white",
                                      font=("Arial", 10, "bold"), height=2)
        self.database_btn.pack(fill=tk.X, pady=(5, 0))

        # --- Panel kiri: info --------------------------------------------
        tk.Label(left_panel, text="INFO", font=("Arial", 10, "bold"),
                 bg="#3c3c3c", fg=self.accent_color).pack(pady=(15, 5),
                                                          anchor=tk.W, padx=10)
        self.info_text = tk.Text(left_panel, height=10, bg="#2b2b2b", fg=self.fg_color,
                                 font=("Consolas", 9), wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # --- Panel kanan: notebook (tab Deteksi & Database) --------------
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 10))
        self._setup_ttk_style()

        self.detect_tab = tk.Frame(self.notebook, bg="#3c3c3c")
        self.db_tab = tk.Frame(self.notebook, bg="#3c3c3c")
        self.notebook.add(self.detect_tab, text="  Deteksi  ")
        self.notebook.add(self.db_tab, text="  Database  ")

        # --- Tab Deteksi: display + hasil hitungan ------------------------
        tk.Label(self.detect_tab, text="DISPLAY", font=("Arial", 10, "bold"),
                 bg="#3c3c3c", fg=self.accent_color).pack(pady=(10, 5))
        self.display_label = tk.Label(self.detect_tab, bg="#1e1e1e",
                                     text="No image/video loaded", fg="#666666",
                                     font=("Arial", 12))
        self.display_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        tk.Label(self.detect_tab, text="HASIL PENGHITUNGAN",
                 font=("Arial", 10, "bold"),
                 bg="#3c3c3c", fg=self.accent_color).pack(anchor=tk.W, padx=10)
        self.results_text = scrolledtext.ScrolledText(self.detect_tab, height=10,
                                                     bg="#1e1e1e", fg=self.fg_color,
                                                     font=("Consolas", 9))
        self.results_text.pack(fill=tk.X, padx=10, pady=(0, 10))

        # --- Tab Database -------------------------------------------------
        self._create_db_tab()
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        # --- Status bar ----------------------------------------------------
        status_frame = tk.Frame(self.root, bg="#1e1e1e", height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        self.status_label = tk.Label(status_frame, text="Ready", bg="#1e1e1e",
                                     fg=self.fg_color, font=("Arial", 9))
        self.status_label.pack(side=tk.LEFT, padx=10)
        self.fps_label = tk.Label(status_frame, text="FPS: --", bg="#1e1e1e",
                                  fg=self.success_color, font=("Arial", 9))
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        self.count_label = tk.Label(status_frame, text="Kendaraan: 0", bg="#1e1e1e",
                                    fg=self.warning_color, font=("Arial", 9, "bold"))
        self.count_label.pack(side=tk.RIGHT, padx=10)

        self._update_info("Siap.\nPilih sumber, lalu klik DETECT.\n\n"
                          "Penghitungan memakai ID track: satu kendaraan yang "
                          "terlihat di banyak frame tetap dihitung 1 kali.")

    # ------------------------------------------------------------------
    # Dialog
    # ------------------------------------------------------------------

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Pilih Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")])
        if path:
            self.model_path.set(path)

    def _browse_source(self) -> None:
        path = filedialog.askopenfilename(
            title="Pilih Sumber",
            filetypes=[("Media", " ".join(f"*{e}" for e in sorted(VIDEO_EXTS | IMAGE_EXTS))),
                       ("Video", " ".join(f"*{e}" for e in sorted(VIDEO_EXTS))),
                       ("Image", " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))),
                       ("All files", "*.*")])
        if path:
            self.source_path.set(path)

    # ------------------------------------------------------------------
    # Tab Database (MAIN THREAD)
    #
    # Baca database saja lewat thread kecil tidak perlu - tabel kecil dan
    # SQLite WAL, jadi baca singkat di main thread aman (worker tetap
    # boleh menulis di saat yang sama).
    # ------------------------------------------------------------------

    def _setup_ttk_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=self.bg_color, borderwidth=0)
        style.configure("TNotebook.Tab", background="#3c3c3c",
                        foreground="#9a9a9a", padding=[14, 7],
                        font=("Arial", 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", self.accent_color)],
                  foreground=[("selected", "white")])
        style.configure("Treeview", background="#1e1e1e",
                        fieldbackground="#1e1e1e", foreground="#ffffff",
                        rowheight=22, font=("Consolas", 9), borderwidth=0)
        style.configure("Treeview.Heading", background="#4a4a4a",
                        foreground="#ffffff", font=("Arial", 9, "bold"),
                        relief="flat")
        style.map("Treeview",
                  background=[("selected", self.accent_color)],
                  foreground=[("selected", "white")])
        style.configure("Vertical.TScrollbar", background="#4a4a4a",
                        troughcolor="#2b2b2b", bordercolor="#3c3c3c",
                        arrowcolor="#ffffff")

    def _create_db_tab(self) -> None:
        """Isi tab Database: daftar sesi + preview tabel + ekspor CSV."""
        bar = tk.Frame(self.db_tab, bg="#3c3c3c")
        bar.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(bar, text="DATABASE DETEKSI", font=("Arial", 10, "bold"),
                 bg="#3c3c3c", fg=self.accent_color).pack(side=tk.LEFT)
        self.db_status = tk.Label(bar, text="", bg="#3c3c3c", fg="#9a9a9a",
                                  font=("Arial", 8))
        self.db_status.pack(side=tk.RIGHT, padx=4)

        btns = tk.Frame(self.db_tab, bg="#3c3c3c")
        btns.pack(fill=tk.X, padx=8, pady=(0, 6))
        tk.Button(btns, text="Muat Ulang", command=self._db_refresh
                  ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(btns, text="Ekspor CSV", command=self._db_export_csv,
                  bg=self.success_color, fg="white",
                  font=("Arial", 9, "bold")).pack(side=tk.LEFT)
        tk.Label(btns, text="Tabel:", bg="#3c3c3c",
                 fg=self.fg_color).pack(side=tk.LEFT, padx=(14, 4))
        self.db_table_var = tk.StringVar(value="detections")
        self.db_table_box = ttk.Combobox(
            btns, textvariable=self.db_table_var, state="readonly", width=22,
            values=("detections", "frame_stats", "counting_summary",
                    "vehicle_accumulation", "sessions"))
        self.db_table_box.pack(side=tk.LEFT)
        self.db_table_box.bind("<<ComboboxSelected>>",
                               lambda _e: self._db_load_table())

        # --- Daftar sesi --------------------------------------------------
        tk.Label(self.db_tab, text="SESI", font=("Arial", 10, "bold"),
                 bg="#3c3c3c", fg=self.accent_color).pack(anchor=tk.W,
                                                          padx=10, pady=(4, 2))
        sess_frame = tk.Frame(self.db_tab, bg="#3c3c3c")
        sess_frame.pack(fill=tk.X, padx=8)
        sess_cols = ("id", "nama", "sumber", "waktu", "frame", "status")
        self.db_sessions = ttk.Treeview(
            sess_frame, columns=sess_cols, show="headings", height=5,
            selectmode="browse")
        for col, title, width in zip(sess_cols,
                                     ("ID", "Nama", "Sumber", "Waktu",
                                      "Frame", "Status"),
                                     (36, 200, 70, 150, 60, 90)):
            self.db_sessions.heading(col, text=title)
            self.db_sessions.column(col, width=width, stretch=col in ("nama",))
        sess_scroll = ttk.Scrollbar(sess_frame, orient=tk.VERTICAL,
                                    command=self.db_sessions.yview)
        self.db_sessions.configure(yscrollcommand=sess_scroll.set)
        self.db_sessions.pack(side=tk.LEFT, fill=tk.X, expand=True)
        sess_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.db_sessions.bind("<<TreeviewSelect>>", self._db_on_session_select)

        # --- Preview tabel ------------------------------------------------
        head = tk.Frame(self.db_tab, bg="#3c3c3c")
        head.pack(fill=tk.X, padx=10, pady=(8, 2))
        tk.Label(head, text="ISI TABEL", font=("Arial", 10, "bold"),
                 bg="#3c3c3c", fg=self.accent_color).pack(side=tk.LEFT)
        self.db_row_info = tk.Label(head, text="", bg="#3c3c3c",
                                    fg="#9a9a9a", font=("Arial", 8))
        self.db_row_info.pack(side=tk.RIGHT)

        prev_frame = tk.Frame(self.db_tab, bg="#3c3c3c")
        prev_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.db_preview = ttk.Treeview(prev_frame, show="headings",
                                       selectmode="browse")
        prev_v = ttk.Scrollbar(prev_frame, orient=tk.VERTICAL,
                               command=self.db_preview.yview)
        prev_h = ttk.Scrollbar(prev_frame, orient=tk.HORIZONTAL,
                               command=self.db_preview.xview)
        self.db_preview.configure(yscrollcommand=prev_v.set,
                                  xscrollcommand=prev_h.set)
        self.db_preview.grid(row=0, column=0, sticky="nsew")
        prev_v.grid(row=0, column=1, sticky="ns")
        prev_h.grid(row=1, column=0, sticky="ew")
        prev_frame.rowconfigure(0, weight=1)
        prev_frame.columnconfigure(0, weight=1)

        self._db_refresh()

    # --- aksi tab database ------------------------------------------------

    def _open_db_tab(self) -> None:
        if self.notebook.select() == str(self.db_tab):
            self._db_refresh()
        else:  # memilih tab memicu <<NotebookTabChanged>> -> refresh
            self.notebook.select(self.db_tab)

    def _on_tab_changed(self, _event=None) -> None:
        try:
            if self.notebook.select() == str(self.db_tab):
                self._db_refresh()
        except tk.TclError:
            pass

    def _db_refresh(self) -> None:
        """Muat ulang daftar sesi + isi tabel preview."""
        prev = self._db_selected_session()
        self._db_loaded_key = None
        self.db_sessions.delete(*self.db_sessions.get_children())
        if not Path(DB_PATH).is_file():
            self.db_status.config(text=f"Belum ada database: {DB_PATH}")
            self.db_row_info.config(text="")
            self._clear_preview("(database belum ada)")
            return

        db = DetectionDatabase()
        try:
            sessions = db.get_sessions()
        finally:
            db.close()

        ids = []
        for s in sessions:
            sid = int(s["id"])
            ids.append(sid)
            self.db_sessions.insert("", tk.END, iid=str(sid), values=(
                sid, s["session_name"], s["source_type"], s["start_time"],
                s["total_frames"], s["status"]))
        self.db_status.config(
            text=f"{len(sessions)} sesi | {Path(DB_PATH).name}")
        if not ids:
            self.db_row_info.config(text="")
            self._clear_preview("(belum ada sesi)")
            return

        target = str(prev) if prev in ids else str(ids[0])
        self.db_sessions.selection_set(target)
        self.db_sessions.see(target)
        self._db_load_table()

    def _db_selected_session(self) -> Optional[int]:
        sel = self.db_sessions.selection()
        if not sel:
            return None
        try:
            return int(self.db_sessions.item(sel[0], "values")[0])
        except (ValueError, tk.TclError):
            return None

    def _db_on_session_select(self, _event=None) -> None:
        self._db_load_table()

    def _clear_preview(self, note: str = "") -> None:
        self.db_preview.delete(*self.db_preview.get_children())
        self.db_preview["columns"] = ()
        self.db_row_info.config(text=note)

    def _db_load_table(self, force: bool = False) -> None:
        """
        Tampilkan isi tabel (difilter sesi terpilih) di treeview.

        Dipanggil juga dari event seleksi yang bisa terpicu dua kali
        (pemilihan baris + refresh eksplisit), jadi isi yang sama tidak
        dimuat ulang kecuali ``force=True``.
        """
        table = self.db_table_var.get()
        session_id = self._db_selected_session()
        key = (table, session_id)
        if not force and key == getattr(self, "_db_loaded_key", None):
            return
        if not Path(DB_PATH).is_file():
            self._db_loaded_key = key
            self._clear_preview("(database belum ada)")
            return

        db = DetectionDatabase()
        try:
            if table not in [r[0] for r in db.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")]:
                self._db_loaded_key = key
                self._clear_preview(f"(tabel {table} tidak ada)")
                return
            cols = [c[1] for c in db.conn.execute(
                f'PRAGMA table_info("{table}")')]
            where, params = [], []
            if session_id is not None:
                if "session_id" in cols:
                    where.append("session_id = ?")
                    params.append(session_id)
                elif table == "sessions":
                    where.append("id = ?")
                    params.append(session_id)
            where_sql = (" WHERE " + " AND ".join(where)) if where else ""
            total = db.conn.execute(
                f'SELECT COUNT(*) FROM "{table}"{where_sql}',
                params).fetchone()[0]
            cursor = db.conn.execute(
                f'SELECT * FROM "{table}"{where_sql} ORDER BY rowid LIMIT 500',
                params)
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description]
        finally:
            db.close()

        self._db_loaded_key = key
        self.db_preview.delete(*self.db_preview.get_children())
        self.db_preview["columns"] = cols
        for i, col in enumerate(cols):
            self.db_preview.heading(col, text=col)
            width = max(70, min(220, 10 + 8 * max(
                len(col), *(len(str(r[i])) for r in rows[:50]))))
            self.db_preview.column(col, width=width, stretch=False)
        for row in rows:
            self.db_preview.insert("", tk.END, values=[
                "" if v is None else v for v in row])
        self.db_row_info.config(
            text=(f"{len(rows)} dari {total} baris"
                  + (f" (sesi #{session_id})" if session_id is not None else "")
                  + (" - menampilkan 500 pertama" if total > len(rows) else "")))

    def _db_export_csv(self) -> None:
        """Simpan tabel (opsional per sesi) ke CSV via dialog simpan."""
        table = self.db_table_var.get()
        session_id = self._db_selected_session()
        default_name = (f"export_{table}"
                        + (f"_session{session_id}" if session_id is not None
                           else "") + ".csv")
        path = filedialog.asksaveasfilename(
            title="Ekspor CSV", defaultextension=".csv",
            initialdir=str(OUTPUT_DIR), initialfile=default_name,
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        db = DetectionDatabase()
        try:
            rc = export_csv(db, session_id=session_id, table=table,
                            output_path=path)
        finally:
            db.close()
        if rc == 0:
            self.db_status.config(text=f"Tersimpan: {path}")
            messagebox.showinfo("Ekspor CSV", f"Berhasil disimpan ke:\n{path}")
        else:
            messagebox.showerror("Ekspor CSV",
                                 f"Gagal mengekspor tabel '{table}'. "
                                 "Lihat pesan di terminal.")

    # ------------------------------------------------------------------
    # Ponteks run: dibaca di MAIN THREAD sebelum worker dibuat
    # ------------------------------------------------------------------

    def _make_run_context(self) -> dict:
        """
        Kumpulkan semua nilai Tk/config yang dibutuhkan worker.

        Dipanggil dari main thread, jadi aman membaca StringVar/DoubleVar.
        """
        conf = float(self.conf_threshold.get())
        model_cfg = self.config.setdefault("model", {})
        # Ambang low tidak boleh lebih tinggi dari ambang high.
        model_cfg["low_conf_threshold"] = min(model_cfg.get("low_conf_threshold", 0.25), conf)
        model_cfg["confidence_threshold"] = conf
        return {
            "config": self.config,
            "model_path": self.model_path.get().strip() or None,
            "save_db": bool(self.save_to_db.get()),
        }

    def _set_running(self, running: bool) -> None:
        self.is_processing = running
        state = tk.DISABLED if running else tk.NORMAL
        for btn in (self.detect_btn, self.webcam_btn, self.cctv_btn):
            btn.config(state=state)
        self.stop_btn.config(state=tk.NORMAL if running else tk.DISABLED)

    def _spawn(self, target_name: str, **kwargs) -> None:
        self._set_running(True)
        self._update_status("Memproses...")
        ctx = self._make_run_context()
        self._worker = threading.Thread(
            target=getattr(self, target_name), args=(ctx,), kwargs=kwargs, daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Loop video/webcam/cctv (WORKER THREAD)
    # ------------------------------------------------------------------

    def _stream_worker(self, ctx: dict, kind: str, source, save_video: bool = False) -> None:
        """
        Loop utama untuk video / webcam / CCTV.

        Tidak boleh menyentuh widget. Semua hasil dikirim lewat
        ``_frame_queue`` dan ditulis ke database.
        """
        pipeline = None
        cap = None
        writer = None
        db = None
        session_id = None
        output_path = None
        try:
            pipeline = VehiclePipeline(config=ctx["config"], model_path=ctx["model_path"])
            pipeline.reset()

            cap = cv2.VideoCapture(0 if source in ("0", 0) else source)
            if not cap.isOpened():
                raise RuntimeError(f"Tidak dapat membuka sumber: {source}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            if save_video and kind == "video":
                stem = Path(source).stem
                output_path = str(Path(source).parent / f"{stem}_output.mp4")
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps, (width, height))

            if ctx["save_db"]:
                db = DetectionDatabase()
                session_id = db.start_session(
                    session_name=f"{Path(str(source)).stem or kind}_{int(time.time())}",
                    source_type=kind, source_path=str(source))

            self._push_info(
                f"Sumber    : {source}\n"
                f"Resolusi  : {width}x{height}\n"
                f"Model     : {Path(pipeline.model_path).name}\n"
                f"Ambang    : low {pipeline.low_conf_threshold} / "
                f"high {pipeline.conf_threshold}\n"
                f"ROI       : {'ON' if pipeline.roi_filter.config.enabled else 'OFF'} "
                f"({pipeline.roi_filter.coverage_ratio((height, width, 3)):.1%} dari frame)\n"
                f"Tracker   : min_hits={pipeline.tracker.min_hits}, "
                f"max_age={pipeline.tracker.max_age}\n"
                f"Garis     : {pipeline.counter.line1_position} / "
                f"{pipeline.counter.line2_position}\n"
                f"Database  : {'AKTIF' if db else 'tidak aktif'}\n"
                f"Total frame: {total_frames or '?'}\n\nMemproses..."
            )

            frame_no = 0
            wall_start = time.perf_counter()
            reconnect = kind == "cctv"

            while self.is_processing:
                ok, frame = cap.read()
                if not ok:
                    if reconnect and self.is_processing:
                        self._push_info("Stream terputus. Mencoba sambung ulang...")
                        time.sleep(1.0)
                        cap.release()
                        cap = cv2.VideoCapture(source)
                        continue
                    break

                frame_no += 1
                annotated, tracked, info = pipeline.process_frame(
                    frame, timestamp=time.perf_counter() - wall_start)

                if writer is not None:
                    writer.write(annotated)

                if db is not None:
                    db.save_detections_batch(session_id, [
                        {
                            "vehicle_id": f"track_{t['track_id']}",
                            "class_id": t["class_id"], "class_name": t["class_name"],
                            "confidence": t["confidence"], "bbox": t["bbox"],
                            "frame_number": info["frame_number"],
                            "timestamp": info["timestamp"],
                            "direction": t.get("direction"),
                            "counted": bool(t.get("counted")),
                        } for t in tracked
                    ])
                    db.save_frame_stats(
                        session_id, info["frame_number"], info["detections_count"],
                        info["after_roi_count"], info["tracked_count"],
                        info["counted_count"], info["fps"], info["timestamp"])

                self._queue_frame(annotated, info, pipeline)

                if frame_no % 30 == 0:
                    progress = (f"{frame_no / total_frames * 100:.0f}%"
                                if total_frames else "-")
                    self._push_info(
                        f"Frame     : {frame_no}"
                        f"{f'/{total_frames}' if total_frames else ''} ({progress})\n"
                        f"FPS       : {info['fps']:.1f}\n"
                        f"Output YOLO: {info['detections_count']}  "
                        f"setelah ROI: {info['after_roi_count']}  "
                        f"dilacak: {info['tracked_count']}\n"
                        f"KENDARAAN TERHITUNG: {info['total_count']} "
                        f"(unik per track)\n"
                        f"Per kelas : {pipeline.counter.get_count_summary()['by_class']}"
                    )

            summary = pipeline.summary()
            self._push_done(
                f"Video Selesai\n{'=' * 32}\n"
                f"Frame diproses : {summary['total_frames']}\n"
                f"FPS rerata     : {summary['avg_fps']:.1f}\n"
                f"KENDARAAN     : {summary['total_counted']}\n\n"
                f"Per kelas:\n" + _fmt_counts(summary["by_class"])
                + f"\nMASUK (down):\n" + _fmt_counts(summary["by_direction"].get("down", {}))
                + f"\nKELUAR (up):\n" + _fmt_counts(summary["by_direction"].get("up", {}))
                + (f"\nVideo output : {output_path}" if output_path else "")
                + (f"\nDatabase     : session #{session_id} tersimpan" if db else "")
            )

        except Exception as exc:  # noqa: BLE001 - dikembalikan ke GUI
            self._push_error(str(exc))
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if db is not None and session_id is not None and pipeline is not None:
                try:
                    db.finalize_session(session_id, pipeline.summary())
                finally:
                    db.close()
            self._push_stopped()

    # ------------------------------------------------------------------
    # Image tunggal & folder (WORKER THREAD)
    # ------------------------------------------------------------------

    def _image_worker(self, ctx: dict, source: str) -> None:
        try:
            pipeline = VehiclePipeline(config=ctx["config"], model_path=ctx["model_path"])
            img = cv2.imread(source)
            if img is None:
                raise RuntimeError(f"Gambar tidak terbaca: {source}")

            start = time.perf_counter()
            annotated, tracked, info = pipeline.process_frame(img, timestamp=0.0)
            elapsed = time.perf_counter() - start

            self._push_info(
                f"File      : {Path(source).name}\n"
                f"Resolusi  : {img.shape[1]}x{img.shape[0]}\n"
                f"Waktu     : {elapsed * 1000:.0f} ms\n"
                f"Setelah ROI: {info['after_roi_count']} dari "
                f"{info['detections_count']} deteksi\n"
                f"Dilacak   : {info['tracked_count']}\n\n"
                "CATATAN: pada satu gambar tidak ada lintasan, jadi penghitungan "
                "dual-line tidak bisa menghasilkan angka. Penghitungan hanya "
                "berfungsi untuk video/webcam/CCTV."
            )
            self._queue_frame(annotated, info, pipeline)
            self._push_done(
                f"Deteksi Selesai\n{'=' * 32}\n"
                f"File       : {Path(source).name}\n"
                f"Waktu      : {elapsed * 1000:.0f} ms ({1 / elapsed:.1f} FPS)\n"
                f"Deteksi    : {info['detections_count']}\n"
                f"Setelah ROI: {info['after_roi_count']}\n"
                f"Dilacak    : {info['tracked_count']}\n"
                + _fmt_object_list(tracked)
            )
        except Exception as exc:  # noqa: BLE001
            self._push_error(str(exc))
        finally:
            self._push_stopped()

    def _directory_worker(self, ctx: dict, source: str) -> None:
        try:
            pipeline = VehiclePipeline(config=ctx["config"], model_path=ctx["model_path"])
            files = sorted(
                p for p in Path(source).iterdir()
                if p.suffix.lower() in IMAGE_EXTS
            )
            if not files:
                raise RuntimeError(f"Tidak ada gambar di: {source}")

            self._push_info(f"Mengolah {len(files)} gambar dari {Path(source).name}...")
            totals: dict = {}
            total_rows = 0
            for i, path in enumerate(files, 1):
                if not self.is_processing:
                    break
                img = cv2.imread(str(path))
                if img is None:
                    continue
                _, tracked, info = pipeline.process_frame(img, timestamp=0.0)
                total_rows += info["tracked_count"]
                for t in tracked:
                    totals[t["class_name"]] = totals.get(t["class_name"], 0) + 1
                if i % 5 == 0 or i == len(files):
                    self._push_info(
                        f"Progres   : {i}/{len(files)}\n"
                        f"Kendaraan terdeteksi (per track): {total_rows}\n"
                        f"Per kelas : {totals}"
                    )
            self._push_done(
                f"Batch Selesai\n{'=' * 32}\n"
                f"Gambar      : {len(files)}\n"
                f"Kendaraan   : {total_rows} (jumlah track, BUKAN kendaraan unik)\n"
                f"Per kelas   :\n" + _fmt_counts(totals)
                + "\nSetiap gambar dihitung terpisah - tidak ada lintasan, "
                  "jadi tidak ada penghitungan dual-line."
            )
        except Exception as exc:  # noqa: BLE001
            self._push_error(str(exc))
        finally:
            self._push_stopped()

    # ------------------------------------------------------------------
    # Handler tombol (MAIN THREAD)
    # ------------------------------------------------------------------

    def _run_detection(self) -> None:
        source = self.source_path.get().strip()
        if not source:
            messagebox.showerror("Error", "Pilih file gambar, video, atau folder dulu.")
            return
        path = Path(source)
        if path.is_dir():
            self._spawn("_directory_worker", source=source)
        elif path.suffix.lower() in VIDEO_EXTS:
            self._spawn("_stream_worker", kind="video", source=source, save_video=True)
        elif path.suffix.lower() in IMAGE_EXTS:
            self._spawn("_image_worker", source=source)
        else:
            messagebox.showerror("Error", f"Format tidak didukung: {path.suffix}")

    def _run_webcam(self) -> None:
        self._spawn("_stream_worker", kind="webcam", source=0)

    def _run_cctv(self) -> None:
        url = self.rtsp_url.get().strip()
        if not url:
            messagebox.showerror("Error", "Masukkan RTSP URL.")
            return
        self._spawn("_stream_worker", kind="cctv", source=url)

    def _stop_processing(self) -> None:
        self.is_processing = False
        self._update_status("Menghentikan...")

    def _on_close(self) -> None:
        self.is_processing = False
        self.root.destroy()

    # ------------------------------------------------------------------
    # Jembatan worker -> main thread
    # ------------------------------------------------------------------

    def _queue_frame(self, frame, info: dict, pipeline) -> None:
        """Masukkan frame terbaru ke queue; buang yang tertinggal (no backlog)."""
        payload = {
            "kind": "frame",
            "frame": frame,
            "info": info,
            "counts": pipeline.counter.get_count_summary(),
        }
        try:
            self._frame_queue.put_nowait(payload)
        except queue.Full:
            try:  # buang frame tertua, simpan yang paling baru
                self._frame_queue.get_nowait()
                self._frame_queue.put_nowait(payload)
            except (queue.Empty, queue.Full):
                pass

    def _push_info(self, text: str) -> None:
        self._pending["info"] = text

    def _push_done(self, text: str) -> None:
        self._pending["done"] = text

    def _push_error(self, text: str) -> None:
        self._pending["error"] = text

    def _push_stopped(self) -> None:
        self._pending["stopped"] = True

    def _drain_queue(self) -> None:
        """
        Dipanggil berulang dari main thread oleh ``root.after``.

        Satu-satunya tempat yang menyentuh widget. Dijadwalkan ulang supaya
        tidak membeku kalau worker sedang sibuk.
        """
        try:
            while True:
                payload = self._frame_queue.get_nowait()
                if payload.get("kind") == "frame":
                    self._render_frame(payload)
        except queue.Empty:
            pass
        except Exception:  # noqa: BLE001
            pass

        if "info" in self._pending:
            self._update_info(self._pending.pop("info"))
        if "done" in self._pending:
            self._update_results(self._pending.pop("done"))
            self._update_status("Selesai")
            # Sesi baru pasti tercatat - muat ulang kalau tab Database aktif.
            if self.notebook.select() == str(self.db_tab):
                self._db_refresh()
        if "error" in self._pending:
            err = self._pending.pop("error")
            self._update_info(f"ERROR:\n{err}")
            self._update_status(f"Error: {err}")
            messagebox.showerror("Error", err)
        if self._pending.pop("stopped", False):
            self._set_running(False)

        self.root.after(30, self._drain_queue)

    def _render_frame(self, payload: dict) -> None:
        """Konversi BGR -> PhotoImage di main thread, lalu tampilkan."""
        frame = payload["frame"]
        info = payload["info"]
        counts = payload["counts"]

        display_w = self.display_label.winfo_width()
        display_h = self.display_label.winfo_height()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        if display_w > 1 and display_h > 1:
            pil_img.thumbnail((display_w, display_h), Image.Resampling.LANCZOS)

        self._tk_img = ImageTk.PhotoImage(pil_img)
        self.display_label.config(image=self._tk_img, text="")

        self.fps_label.config(text=f"FPS: {info['fps']:.1f}")
        # AngkaKendaraan yang ditampilkan adalah hitungan per track,
        # bukan jumlah deteksi di frame ini.
        self.count_label.config(text=f"Kendaraan: {counts['total']}")

    def _update_status(self, text: str) -> None:
        self.status_label.config(text=text)

    def _update_info(self, text: str) -> None:
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)

    def _update_results(self, text: str) -> None:
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)


def _fmt_counts(counts: dict) -> str:
    if not counts:
        return "  (tidak ada)\n"
    return "".join(f"  {name:6s}: {n}\n" for name, n in sorted(counts.items()))


def _fmt_object_list(tracked) -> str:
    if not tracked:
        return "\nTidak ada kendaraan terdeteksi.\n"
    lines = ["\nObjek terdeteksi:\n"]
    for t in tracked:
        lines.append(f"  #{t['track_id']} {t['class_name']} "
                     f"conf={t['confidence']:.2f} hits={t['hits']}\n")
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GUI deteksi kendaraan (Tkinter)")
    parser.add_argument("--config", default=None,
                        help="Path config YAML (default: config/config.yaml)")
    parser.add_argument("--source", default=None,
                        help="Isi field sumber otomatis (gambar/video/RTSP)")
    parser.add_argument("--list-sources", action="store_true",
                        help="Tampilkan daftar video yang tersedia lalu keluar")
    args = parser.parse_args()

    if args.list_sources:
        from utils.paths import ROOT
        found = []
        for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            found.extend(sorted(ROOT.rglob(pattern)))
        print("=== VIDEO DI PROJECT ROOT ===")
        if not found:
            print("  (tidak ada video)")
        for p in found:
            print(f"  {p.relative_to(ROOT)}")
        return 0

    if args.config:
        global _CONFIG_OVERRIDE
        _CONFIG_OVERRIDE = args.config

    root = tk.Tk()
    gui = VehicleDetectionGUI(root)
    if args.source:
        gui.source_path.set(args.source)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
