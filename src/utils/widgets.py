"""Widget Tkinter tambahan untuk GUI.

Modul ini hanya dipakai oleh ``gui_app``; modul lain di ``utils`` tidak
mengimpor ``tkinter`` supaya tetap bisa dipakai dari konteks non-GUI.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from typing import Any, Callable, Optional, Sequence, Tuple

__all__ = ["RoundedButton"]

DEFAULT_FONT: Tuple[str, int, str] = ("Arial", 10, "bold")


def _parse_hex(color: str) -> Tuple[int, int, int]:
    a = color.lstrip("#")
    if len(a) == 3:
        a = "".join(ch * 2 for ch in a)
    if len(a) != 6:
        raise ValueError("warna bukan hex: %r" % color)
    return int(a[0:2], 16), int(a[2:4], 16), int(a[4:6], 16)


def to_hex(color: str, widget: tk.Misc) -> str:
    """Normalisasi warna apa pun (#hex, 'white', 'systemButtonFace') ke hex."""
    if isinstance(color, str) and color.startswith("#"):
        try:
            _parse_hex(color)
            return color.lower()
        except ValueError:
            pass
    try:
        r, g, b = widget.winfo_rgb(color)
    except tk.TclError:
        return "#000000"
    return "#{:02x}{:02x}{:02x}".format(r * 255 // 65535, g * 255 // 65535,
                                        b * 255 // 65535)


def _mix(color: str, other: str, ratio: float) -> str:
    """Campur dua warna hex, ``ratio`` = 0 -> color, 1 -> other."""
    ca = _parse_hex(color)
    cb = _parse_hex(other)
    out = [int(round(x + (y - x) * ratio)) for x, y in zip(ca, cb)]
    return "#{:02x}{:02x}{:02x}".format(*out)


class RoundedButton(tk.Canvas):
    """Tombol dengan sudut membulat.

    ``tk.Button`` tidak mendukung radius, jadi tombol digambar sendiri di
    atas ``Canvas``. API-nya sengaja dibuat serupa ``tk.Button`` supaya
    pemakaian lama tetap jalan::

        RoundedButton(parent, text="OK", command=cb, bg="#4caf50")
        btn.config(state=tk.DISABLED)      # abaikan klik
        btn.config(text="PAUSE", bg="#ff9800")
        btn["text"]                        # -> "OK"
    """

    def __init__(
        self,
        master: tk.Misc,
        text: str = "",
        command: Optional[Callable[[], Any]] = None,
        bg: str = "#4a9eff",
        fg: str = "white",
        disabled_bg: Optional[str] = None,
        radius: int = 10,
        height: int = 32,
        font: Sequence[Any] = DEFAULT_FONT,
        width: Optional[int] = None,
        padding: int = 14,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("highlightthickness", 0)
        kwargs.setdefault("bd", 0)
        kwargs.setdefault(
            "background",
            master.cget("background") if hasattr(master, "cget") else "#3c3c3c",
        )
        kwargs.pop("bg", None)
        kwargs.pop("fg", None)
        kwargs.pop("state", None)
        kwargs.pop("text", None)
        kwargs.pop("command", None)
        kwargs.pop("font", None)
        kwargs.pop("height", None)
        kwargs.pop("cursor", None)

        super().__init__(master, height=height, **kwargs)

        self._text = text
        self._command = command
        self._btn_bg = to_hex(bg, self)
        self._fg = to_hex(fg, self)
        self._disabled_bg = to_hex(disabled_bg, self) if disabled_bg else "#5a5a5a"
        self._disabled_fg = _mix(self._fg, "#9e9e9e", 0.75)
        self._radius = radius
        self._font = tuple(font)
        self._height = height
        self._state = tk.NORMAL
        self._hover = False
        self._pressed = False
        self._padding = padding

        self._font_obj = tkfont.Font(font=self._font)
        base = self._font_obj.measure(text) + 2 * padding if text else 2 * padding
        self.configure(width=width if width else max(base, 60))

        self.bind("<Configure>", lambda _e: self._redraw(), add="+")
        self.bind("<Enter>", lambda _e: self._on_enter(), add="+")
        self.bind("<Leave>", lambda _e: self._on_leave(), add="+")
        self.bind("<ButtonPress-1>", self._on_press, add="+")
        self.bind("<ButtonRelease-1>", self._on_release, add="+")
        self._redraw()

    # --- API serupa tk.Button --------------------------------------------
    def configure(self, cnf: Any = None, **kw: Any) -> Any:
        opts: dict = {}
        if cnf:
            if not isinstance(cnf, dict):
                raise TypeError("konfigurasi harus berupa dict")
            opts.update(cnf)
        opts.update(kw)

        for key in list(opts):
            value = opts.pop(key)
            if key == "text":
                self._text = str(value)
            elif key in ("command",):
                self._command = value
            elif key in ("bg", "background", "buttonbackground"):
                self._btn_bg = to_hex(value, self)
            elif key in ("disabledbackground", "disabled_bg"):
                self._disabled_bg = to_hex(value, self)
            elif key in ("fg", "foreground"):
                self._fg = to_hex(value, self)
                self._disabled_fg = _mix(self._fg, "#9e9e9e", 0.75)
            elif key in ("disabledforeground", "disabled_fg"):
                self._disabled_fg = to_hex(value, self)
            elif key == "state":
                self._state = str(value)
            elif key == "font":
                self._font = tuple(value) if isinstance(value, (list, tuple)) \
                    else (value,)
                try:
                    self._font_obj = tkfont.Font(font=self._font)
                except tk.TclError:
                    self._font = DEFAULT_FONT
                    self._font_obj = tkfont.Font(font=self._font)
            elif key in ("radius",):
                self._radius = int(value)
            elif key in ("padding", "padx"):
                self._padding = int(value)
            else:
                opts[key] = value

        if opts:
            result = super().configure(**opts)
        else:
            result = None

        self.configure_cursor()
        self._redraw()
        return result

    config = configure

    def cget(self, key: str) -> Any:
        if key == "text":
            return self._text
        if key in ("bg", "background", "buttonbackground"):
            return self._btn_bg
        if key in ("disabledbackground", "disabled_bg"):
            return self._disabled_bg
        if key in ("fg", "foreground"):
            return self._fg
        if key in ("disabledforeground", "disabled_fg"):
            return self._disabled_fg
        if key == "command":
            return self._command
        if key == "state":
            return self._state
        if key == "font":
            return self._font
        if key == "radius":
            return self._radius
        if key == "padding":
            return self._padding
        if key == "height":
            return self._height
        return super().cget(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.configure(**{key: value})

    def __getitem__(self, key: str) -> Any:
        return self.cget(key)

    def configure_cursor(self) -> None:
        super().configure(cursor="hand2" if self._state != tk.DISABLED else "")

    def invoke(self) -> Any:
        """Jalankan ``command`` (hanya bila tombol aktif)."""
        if self._state == tk.DISABLED:
            return None
        return self._command() if self._command else None

    # --- perilaku ----------------------------------------------------------
    def _on_enter(self, _event: Any = None) -> None:
        if self._state == tk.DISABLED:
            return
        self._hover = True
        self._redraw()

    def _on_leave(self, _event: Any = None) -> None:
        if self._hover or self._pressed:
            self._hover = False
            self._pressed = False
            self._redraw()

    def _on_press(self, _event: Any = None) -> None:
        if self._state == tk.DISABLED:
            return
        self._pressed = True
        self._redraw()

    def _on_release(self, event: Any = None) -> None:
        was_pressed = self._pressed
        self._pressed = False
        if event is not None:
            inside = (0 <= event.x < self.winfo_width()
                      and 0 <= event.y < self.winfo_height())
        else:
            inside = True
        if self._state == tk.DISABLED:
            self._redraw()
            return
        self._redraw()
        if was_pressed and inside:
            self.invoke()

    # --- gambar ------------------------------------------------------------
    def _fill(self) -> str:
        if self._state == tk.DISABLED:
            return self._disabled_bg
        if self._pressed:
            return _mix(self._btn_bg, "#000000", 0.22)
        if self._hover:
            return _mix(self._btn_bg, "#ffffff", 0.16)
        return self._btn_bg

    def _round_rect(self, x1: int, y1: int, x2: int, y2: int, r: float,
                    **kw: Any) -> int:
        points = [
            x1 + r, y1, x1 + r, y1, x2 - r, y1, x2 - r, y1, x2, y1,
            x2, y1 + r, x2, y1 + r, x2, y2 - r, x2, y2 - r, x2, y2,
            x2 - r, y2, x2 - r, y2, x1 + r, y2, x1 + r, y2, x1, y2,
            x1, y2 - r, x1, y2 - r, x1, y1 + r, x1, y1 + r, x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kw)

    def _redraw(self) -> None:
        try:
            self.delete("all")
            w = self.winfo_width()
            h = self.winfo_height()
        except tk.TclError:
            return
        if w <= 2 or h <= 2:
            return

        r = max(2, min(self._radius, h / 2.0, w / 2.0))
        self._round_rect(1, 1, w - 2, h - 2, r, fill=self._fill(), outline="")

        fill = self._fg if self._state != tk.DISABLED else self._disabled_fg
        self.create_text(w / 2.0, h / 2.0, text=self._text, fill=fill,
                         font=self._font)
