"""Pemilih koordinat ROI interaktif (klik 4 titik pada frame asli).

Kenapa ada file ini: koordinat ROI yang disalin dari tool berbeda (player
zoom, screenshot, dsb) punya ruang piksel berbeda dan bisa melebihi tinggi
frame, sehingga ROI jadi meleset diam-diam. Dengan mengklik langsung pada
frame, koordinat selalu 1:1 dengan piksel video.

Urutan klik (sesuai urutan key di config):
    1. Kiri Atas   -> top_left
    2. Kanan Atas  -> top_right
    3. Kanan Bawah -> bottom_right
    4. Kiri Bawah  -> bottom_left

Tombol:
    klik kiri   tambah titik
    u           hapus titik terakhir
    r           reset semua titik
    s / Enter   simpan (cetak hasil; tulis config bila --apply)
    q / ESC     batal

Contoh:
    python src/roi_picker.py
    python src/roi_picker.py --source video.MOV --frame 5000
    python src/roi_picker.py --apply
    python src/roi_picker.py --lines            # klik posisi garis hitung
    python src/roi_picker.py --lines --apply
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.paths import CONFIG_PATH, load_config

# (key di config, label yang ditampilkan saat klik)
CORNERS: Sequence[Tuple[str, str]] = (
    ("top_left", "1 Kiri Atas (Depan/Jauh Kiri)"),
    ("top_right", "2 Kanan Atas (Depan/Jauh Kanan)"),
    ("bottom_right", "3 Kanan Bawah (Dekat/Kanan)"),
    ("bottom_left", "4 Kiri Bawah (Dekat/Kiri)"),
)

MAX_DISPLAY_W = 1500
MAX_DISPLAY_H = 850

COLOR_POINT = (0, 0, 255)
COLOR_NEXT = (0, 255, 255)
COLOR_POLY = (0, 255, 0)
COLOR_OLD = (200, 200, 200)
COLOR_LABEL = (255, 255, 255)


class PickerState:
    def __init__(self, scale: float, old_polygon: Optional[List[Tuple[int, int]]] = None):
        self.scale = scale
        self.points: List[Tuple[int, int]] = []
        self.old_polygon = old_polygon or []
        self.done = False

    def add(self, x: int, y: int) -> None:
        if len(self.points) < len(CORNERS) and not self.done:
            self.points.append((x, y))

    def undo(self) -> None:
        if not self.done and self.points:
            self.points.pop()

    def reset(self) -> None:
        self.points.clear()
        self.done = False


def load_frame(source: str, frame_index: int):
    """Ambil satu frame. ``source`` boleh gambar atau video."""
    suffix = Path(source).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = cv2.imread(source)
        if image is None:
            raise FileNotFoundError(f"Gagal membaca gambar: {source}")
        return image

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka sumber: {source}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if frame_index > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Gagal membaca frame {frame_index} dari {source}")
            return frame
        if total > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Gagal membaca frame dari {source}")
        return frame
    finally:
        cap.release()


def compute_scale(width: int, height: int) -> float:
    return min(1.0, MAX_DISPLAY_W / width, MAX_DISPLAY_H / height)


def make_canvas(frame, scale: float, state: PickerState):
    if scale != 1.0:
        canvas = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        canvas = frame.copy()

    h, w = canvas.shape[:2]

    if len(state.old_polygon) == 4:
        pts = np.array([(x * scale, y * scale) for x, y in state.old_polygon], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, COLOR_OLD, 1, cv2.LINE_AA)

    pts_px = [(int(x * scale), int(y * scale)) for x, y in state.points]
    for i, (px, py) in enumerate(pts_px):
        label = CORNERS[i][1]
        cv2.circle(canvas, (px, py), 6, COLOR_POINT, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (px + 10, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1, cv2.LINE_AA)

    if pts_px:
        arr = np.array(pts_px, dtype=np.int32)
        cv2.polylines(canvas, [arr], len(pts_px) == 4, COLOR_POLY, 2, cv2.LINE_AA)

    next_idx = len(pts_px)
    if next_idx < len(CORNERS):
        cv2.putText(canvas, f"Klik: {CORNERS[next_idx][1]}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_NEXT, 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "Lengkap - tekan [s] simpan, [u] undo, [r] reset", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_POLY, 2, cv2.LINE_AA)

    cv2.putText(canvas, "klik=tambah  u=undo  r=reset  s=save  q=cancel",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1, cv2.LINE_AA)
    return canvas


def on_mouse(event, x, y, flags, param) -> None:
    state: PickerState = param
    if event == cv2.EVENT_LBUTTONDOWN:
        state.add(int(round(x / state.scale)), int(round(y / state.scale)))


def pick_points(frame, old_polygon: Optional[Sequence[Tuple[int, int]]] = None,
                window_name: str = "Pilih ROI (klik 4 titik)") -> Optional[List[Tuple[int, int]]]:
    """
    Loop interaktif klik 4 titik pada ``frame``.

    Modal: memblokir sampai pengguna menekan ``s``/Enter (hasil) atau
    ``q``/ESC (batal). Aman dipanggil dari main thread Tk karena jendela
    OpenCV punya message loop sendiri; jendela Tk hanya tidak direpaint
    selama picking berlangsung.

    Returns:
        4 titik dalam piksel ``frame``, atau ``None`` bila dibatalkan.
    """
    height, width = frame.shape[:2]
    scale = compute_scale(width, height)
    state = PickerState(scale, [tuple(p) for p in (old_polygon or [])])

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse, state)
    try:
        while True:
            cv2.imshow(window_name, make_canvas(frame, scale, state))
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                return None
            if key == ord("u"):
                state.undo()
            elif key == ord("r"):
                state.reset()
            elif key in (ord("s"), 13) and len(state.points) == 4:
                return list(state.points)
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            cv2.destroyAllWindows()


# --- pemilih garis hitung ------------------------------------------------
LINE_LABELS = ("Garis 1 (MASUK)", "Garis 2 (KELUAR)")
COLOR_LINE1 = (0, 0, 255)
COLOR_LINE2 = (0, 255, 0)


class LineState:
    """Posisi kedua garis hitung dalam piksel frame + riwayat undo."""

    def __init__(self, scale: float, height: int, ys: Sequence[float]):
        self.scale = scale
        self.height = height
        self.ys = [float(y) for y in ys]
        self.initial = list(self.ys)
        self.history: List[List[float]] = []
        self.active = 0

    def click(self, x: int, y: int) -> None:
        """Pindahkan garis yang paling dekat ke posisi klik."""
        frame_y = y / self.scale
        idx = min(range(len(self.ys)), key=lambda i: abs(self.ys[i] - frame_y))
        self.history.append(list(self.ys))
        self.ys[idx] = min(max(frame_y, 0.0), float(self.height - 1))
        self.active = idx

    def undo(self) -> None:
        if self.history:
            self.ys = self.history.pop()

    def reset(self) -> None:
        self.history.append(list(self.ys))
        self.ys = list(self.initial)


def make_line_canvas(frame, state: LineState,
                     old_polygon: Optional[Sequence[Tuple[int, int]]] = None):
    canvas = cv2.resize(frame, None, fx=state.scale, fy=state.scale,
                        interpolation=cv2.INTER_AREA) \
        if state.scale != 1.0 else frame.copy()
    h, w = canvas.shape[:2]

    if old_polygon and len(old_polygon) == 4:
        pts = np.array([(x * state.scale, y * state.scale)
                        for x, y in old_polygon], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, COLOR_OLD, 1, cv2.LINE_AA)

    for i, y in enumerate(state.ys):
        py = int(round(y * state.scale))
        py = max(1, min(h - 2, py))
        color = COLOR_LINE1 if i == 0 else COLOR_LINE2
        active = i == state.active
        cv2.line(canvas, (0, py), (w - 1, py), color, 3 if active else 2,
                 cv2.LINE_AA)
        text = f"{LINE_LABELS[i]}   {y / state.height:.4f}"
        cv2.putText(canvas, text, (15, max(20, py - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color if active else COLOR_LABEL, 2, cv2.LINE_AA)

    cv2.putText(canvas, f"Aktif: {LINE_LABELS[state.active]} - klik untuk memindahkan",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_NEXT, 2, cv2.LINE_AA)
    cv2.putText(canvas, "klik=pindahkan garis terdekat  u=undo  r=reset  "
                        "s=selesai  q=batal",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_LABEL, 1,
                cv2.LINE_AA)
    return canvas


def on_line_mouse(event, x, y, flags, param) -> None:
    state: LineState = param
    if event == cv2.EVENT_LBUTTONDOWN:
        state.click(int(x), int(y))


def pick_lines(frame, ratios: Sequence[float],
               old_polygon: Optional[Sequence[Tuple[int, int]]] = None,
               window_name: str = "Pilih Garis Hitung (klik)") \
        -> Optional[Tuple[float, float]]:
    """
    Loop interaktif untuk menentukan posisi dua garis hitung dengan klik.

    Sama seperti :func:`pick_points` tapi hanya butuh klik: klik kiri akan
    memindahkan garis yang paling dekat ke titik tersebut, sehingga
    penentuan garis memakai cara yang sama dengan penentuan sudut ROI.

    Modal: memblokir sampai ``s``/Enter (hasil) atau ``q``/ESC (batal).

    Returns:
        ``(line1_position, line2_position)`` sebagai rasio tinggi frame,
        atau ``None`` bila dibatalkan.
    """
    height = frame.shape[0]
    scale = compute_scale(frame.shape[1], height)
    state = LineState(scale, height, [r * height for r in ratios])

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_line_mouse, state)
    try:
        while True:
            cv2.imshow(window_name, make_line_canvas(frame, state, old_polygon))
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                return None
            if key == ord("u"):
                state.undo()
            elif key == ord("r"):
                state.reset()
            elif key in (ord("s"), 13):
                return round(state.ys[0] / height, 4), round(state.ys[1] / height, 4)
    finally:
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            cv2.destroyAllWindows()


def recommend_counting(boundary: Dict[str, Tuple[int, int]], ref_h: int,
                       min_displacement: float) -> Tuple[float, float]:
    """
    Saran posisi garis hitung dari geometri ROI.

    line1 = tengah bbox vertikal ROI.
    line2 = sedekat mungkin dengan tepi bawah ROI, tapi masih menyisakan
    ruang ``min_displacement`` agar pusat bbox bisa melewatinya sambil
    tetap berada di dalam poligon ROI.
    """
    ys = [p[1] for p in boundary.values()]
    y_top, y_bottom = min(ys), max(ys)
    line1 = (y_top + y_bottom) / 2.0 / ref_h

    bottom_min = min(boundary["bottom_left"][1], boundary["bottom_right"][1])
    line2 = max(0.0, (bottom_min - min_displacement) / ref_h)

    line1 = min(max(line1, 0.0), 1.0)
    line2 = min(max(line2, 0.0), 1.0)
    if line1 >= line2:
        line1 = max(0.0, line2 - 0.01)
    return round(line1, 4), round(line2, 4)


def _block_range(lines: List[str], block: str, indent: int) -> Optional[Tuple[int, int]]:
    """Indeks awal/akhir isi blok ``<indent>block:`` (di luar baris bloknya)."""
    head = re.compile(r"^" + " " * indent + re.escape(block) + r":\s*(#.*)?$")
    for i, line in enumerate(lines):
        if head.match(line):
            j = i + 1
            while j < len(lines):
                line2 = lines[j]
                stripped = line2.strip()
                if stripped and not stripped.startswith("#"):
                    cur = len(line2) - len(line2.lstrip())
                    if cur <= indent:
                        break
                j += 1
            return i + 1, j
    return None


def patch_config(path: Path, boundary: Dict[str, Tuple[int, int]],
                 reference: Optional[Tuple[int, int]] = None) -> None:
    """Ganti koordinat boundary (dan reference_resolution bila berbeda)
    langsung di teks config supaya komentar di file tetap utuh."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    region = _block_range(lines, "boundary", indent=2)
    if region is None:
        raise RuntimeError("Blok 'roi.boundary' tidak ditemukan di config")
    start, end = region

    seen = set()
    for i in range(start, end):
        for key in boundary:
            match = re.match(rf"^(\s*){key}:\s*\[[^\]]*\](.*)$", lines[i])
            if match:
                tail = match.group(2)
                lines[i] = f"{match.group(1)}{key}: [{boundary[key][0]}, {boundary[key][1]}]{tail}"
                if not tail.endswith("\n"):
                    lines[i] += "\n"
                seen.add(key)
    missing = set(boundary) - seen
    if missing:
        raise RuntimeError(f"Key boundary tidak ditemukan di config: {sorted(missing)}")

    if reference is not None:
        ref_region = _block_range(lines, "reference_resolution", indent=2)
        if ref_region:
            rs, re_ = ref_region
            for i in range(rs, re_):
                if re.match(r"^\s*width:\s*\d", lines[i]):
                    lines[i] = f"    width: {reference[0]}\n"
                elif re.match(r"^\s*height:\s*\d", lines[i]):
                    lines[i] = f"    height: {reference[1]}\n"

    path.write_text("".join(lines), encoding="utf-8")


def patch_counting_lines(path: Path, line1: float, line2: float) -> None:
    """Tulis ``line1_position``/``line2_position`` ke blok ``counting:``
    tanpa menyentuh komentar lain di config."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    region = _block_range(lines, "counting", indent=0)
    if region is None:
        raise RuntimeError("Blok 'counting' tidak ditemukan di config")
    start, end = region

    found = set()
    for i in range(start, end):
        for key, value in (("line1_position", line1), ("line2_position", line2)):
            match = re.match(rf"^(\s*){key}:\s*[\d.]+(.*)$", lines[i])
            if match:
                tail = match.group(2)
                lines[i] = f"{match.group(1)}{key}: {value}{tail}"
                if not lines[i].endswith("\n"):
                    lines[i] += "\n"
                found.add(key)
    missing = {"line1_position", "line2_position"} - found
    if missing:
        raise RuntimeError(f"Key counting tidak ditemukan di config: {sorted(missing)}")

    path.write_text("".join(lines), encoding="utf-8")


def _run_lines(config, config_path: Path, frame, old_polygon,
               frame_size: Tuple[int, int], ref_size: Tuple[int, int],
               apply: bool) -> int:
    """Mode ``--lines``: tentukan posisi garis hitung dengan klik."""
    width, height = frame_size
    ref_w, ref_h = ref_size
    counting = config.get("counting", {}) or {}
    ratios = (float(counting.get("line1_position", 0.6676)),
              float(counting.get("line2_position", 0.7139)))

    poly_frame = None
    if old_polygon and len(old_polygon) == 4:
        poly_frame = [(x * width / ref_w, y * height / ref_h)
                      for x, y in old_polygon]

    print(f"Ukuran frame : {width}x{height}  (reference {ref_w}x{ref_h})")
    print(f"Garis saat ini: line1={ratios[0]}  line2={ratios[1]}")
    print("klik = pindahkan garis terdekat, s = selesai, q = batal")

    result = pick_lines(frame, ratios, poly_frame)
    if result is None:
        print("Batal.")
        return 0

    line1, line2 = result
    print("\n=== HASIL GARIS HITUNG ===")
    print(f"  line1_position: {line1}   (y = {round(line1 * height)} px)")
    print(f"  line2_position: {line2}   (y = {round(line2 * height)} px)")
    if not 0.0 <= line1 < line2 <= 1.0:
        print("[ERROR] Tidak memenuhi 0 <= line1 < line2 <= 1 - hasil tidak ditulis.")
        return 1

    if apply:
        patch_counting_lines(config_path, line1, line2)
        print(f"[OK] {config_path} diperbarui.")
    else:
        print("\nJalankan ulang dengan --apply untuk menulis ke config.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pilih 4 titik sudut ROI dengan klik")
    parser.add_argument("--source", default=None,
                        help="path video/gambar (default: video pertama di config)")
    parser.add_argument("--frame", type=int, default=0,
                        help="index frame yang dibuka (0 = frame tengah)")
    parser.add_argument("--apply", action="store_true",
                        help="tulis hasil ke config/config.yaml")
    parser.add_argument("--lines", action="store_true",
                        help="pilih posisi garis hitung (line1/line2) dengan klik, "
                             "bukan sudut ROI")
    parser.add_argument("--config", default=None, help="path config.yaml alternatif")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else CONFIG_PATH
    config = load_config(config_path if args.config else None)

    source = args.source
    if not source:
        video_source = config.get("dataset", {}).get("video_source")
        video_files = config.get("dataset", {}).get("video_files") or []
        if not (video_source and video_files):
            print("[ERROR] config tidak punya dataset.video_source/video_files, "
                  "jadi --source wajib diisi.")
            return 1
        source = str(Path(video_source) / video_files[0])

    frame = load_frame(source, args.frame)
    height, width = frame.shape[:2]
    scale = compute_scale(width, height)

    roi_cfg = config.get("roi", {})
    ref = roi_cfg.get("reference_resolution") or {}
    ref_w = int(ref.get("width", width))
    ref_h = int(ref.get("height", height))
    old = roi_cfg.get("boundary") or {}
    old_polygon = [tuple(old[k]) for k, _ in CORNERS if k in old]

    # Boundary ditulis pada resolusi referensi; kalau frame ini berbeda,
    # hasil klik di-map dulu ke ruang referensi sebelum disimpan.
    map_x = ref_w / width
    map_y = ref_h / height

    if args.lines:
        return _run_lines(config, config_path, frame, old_polygon,
                          (width, height), (ref_w, ref_h), args.apply)

    print(f"Sumber      : {source}")
    print(f"Ukuran frame: {width}x{height}  (reference {ref_w}x{ref_h}, scale tampil {scale:.3f})")
    print("Urutan klik :", " -> ".join(label for _, label in CORNERS))

    points = pick_points(frame, old_polygon)
    if points is None:
        print("Batal.")
        return 0

    boundary = {
        key: (int(round(x * map_x)), int(round(y * map_y)))
        for (key, _), (x, y) in zip(CORNERS, points)
    }

    print("\n=== HASIL ROI ===")
    for key, label in CORNERS:
        print(f"  {label:32s} {key:14s} {boundary[key]}")

    import numpy as np  # noqa: F401  (dipakai make_canvas)
    poly = np.array([boundary[k] for k, _ in CORNERS], dtype="int32")
    area = abs(float(cv2.contourArea(poly)))
    if not cv2.isContourConvex(poly):
        print("[PERINGATAN] Poligon tidak cekung/berpotongan - urutan klik mungkin keliru.")
    print(f"  Luas: {area:.0f} px^2 ({area / (ref_w * ref_h):.1%} dari frame)")

    line1, line2 = recommend_counting(boundary, ref_h, float(
        config.get("counting", {}).get("min_displacement", 25.0)))
    print("\n=== SARAN POSISI GARIS HITUNG ===")
    print(f"  line1_position: {line1}   (tengah ROI)")
    print(f"  line2_position: {line2}   (batas aman counting)")

    yaml_text = (
        "roi:\n"
        "  boundary:\n"
        + "".join(f"    {k}: [{boundary[k][0]}, {boundary[k][1]}]\n" for k, _ in CORNERS)
    )
    print("\n=== SIAP DITEMPEL KE config/config.yaml ===")
    print(yaml_text)

    if args.apply:
        changed_ref = None
        if (ref_w, ref_h) != (width, height):
            changed_ref = (width, height)
        patch_config(config_path, boundary, changed_ref)
        print(f"[OK] {config_path} diperbarui.")
        if changed_ref:
            print(f"     reference_resolution diubah ke {width}x{height}.")
        print("     Catatan: line1_position/line2_position TIDAK diubah otomatis, "
              "tempel manual dari saran di atas bila perlu.")
    else:
        print("\nJalankan ulang dengan --apply untuk menulis ke config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
