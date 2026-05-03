#!/usr/bin/env python3
"""
ETS2 Navigation & Mirrors Overlay

Extrae y muestra en ventana flotante siempre encima:
  - Panel GPS / Route Advisor (esquina inferior derecha)
  - Retrovisor izquierdo (esquina superior izquierda)
  - Retrovisor derecho (esquina superior derecha)
"""

import os
import threading
import time
import tkinter as tk
from tkinter import Label, Button

import cv2
import numpy as np
from PIL import Image, ImageTk

from ets2_capture import capture_window_quartz, capture_fallback

# -----------------------------------------------------------------------------
# CONFIGURACION
# -----------------------------------------------------------------------------
FPS_LIMIT = 30

# Regiones como fracciones del frame (0.0 - 1.0)
# Ajusta segun tu resolucion y preferencias
REGIONS = {
    "gps": {
        "left": 0.60, "top": 0.58,
        "width": 0.39, "height": 0.40,
    },
    "mirror_left": {
        "left": 0.02, "top": 0.06,
        "width": 0.20, "height": 0.30,
    },
    "mirror_right": {
        "left": 0.78, "top": 0.06,
        "width": 0.20, "height": 0.30,
    },
}

# Cuanto escalar la ventana de salida (para que no sea gigante)
SCALE = 1.0  # 1.0 = tamano original de las regiones

# Mostrar retrovisores?
SHOW_MIRRORS = True

# -----------------------------------------------------------------------------
# UTILIDADES
# -----------------------------------------------------------------------------
def get_region_frame(frame, region_spec):
    """Extrae una region rectangular de un frame."""
    h, w = frame.shape[:2]
    x = int(region_spec["left"] * w)
    y = int(region_spec["top"] * h)
    rw = int(region_spec["width"] * w)
    rh = int(region_spec["height"] * h)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = min(rw, w - x)
    rh = min(rh, h - y)
    return frame[y:y + rh, x:x + rw]


# -----------------------------------------------------------------------------
# OVERLAY TKINTER
# -----------------------------------------------------------------------------
class NavOverlayWindow:
    def __init__(self, win_info):
        self.win_info = win_info
        self._win_lock = threading.Lock()
        self._running = True

        self.root = tk.Tk()
        self.root.title("ETS2 Nav + Mirrors")
        self.root.attributes("-topmost", True)
        try:
            self.root.overrideredirect(True)
        except tk.TclError:
            # macOS tkinter bug: usar wm_attributes como fallback
            self.root.wm_attributes("-type", "splash")
        self.root.configure(bg="black", highlightthickness=0)

        # Boton X rojo
        self.btn_close = Button(
            self.root, text="X", command=self.on_close,
            bg="#cc0000", fg="white", font=("Arial", 8, "bold"),
            bd=0, padx=4, pady=1, cursor="hand2"
        )
        self.btn_close.place(x=0, y=0)

        # Labels para cada region
        self.lbl_gps = Label(self.root, bg="black", bd=0)
        self.lbl_gps.grid(row=1, column=0, columnspan=2, padx=2, pady=2)

        if SHOW_MIRRORS:
            self.lbl_left = Label(self.root, bg="black", bd=0)
            self.lbl_left.grid(row=0, column=0, padx=2, pady=2)

            self.lbl_right = Label(self.root, bg="black", bd=0)
            self.lbl_right.grid(row=0, column=1, padx=2, pady=2)

        # Info bar
        self.lbl_info = Label(
            self.root, text="Nav Overlay", fg="#00ff00", bg="black",
            font=("Courier", 8), bd=0, padx=4, pady=2,
        )
        self.lbl_info.grid(row=2, column=0, columnspan=2, sticky="w")

        # Arrastre
        self.root.bind("<Button-1>", self._start_move)
        self.root.bind("<B1-Motion>", self._on_move)
        self._drag_x = 0
        self._drag_y = 0

        self._lock = threading.Lock()
        self.gps_img = None
        self.left_img = None
        self.right_img = None
        self.fps = 0

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        self._schedule_update()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _start_move(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_move(self, event):
        x = self.root.winfo_x() + event.x - self._drag_x
        y = self.root.winfo_y() + event.y - self._drag_y
        self.root.geometry(f"+{x}+{y}")

    def _capture_loop(self):
        prev_time = time.time()
        frame_count = 0
        fps_display = 0

        while self._running:
            loop_start = time.time()

            with self._win_lock:
                info = dict(self.win_info)

            if info.get("id"):
                frame_bgr = capture_window_quartz(info["id"])
            else:
                frame_bgr = capture_fallback(info)

            if frame_bgr is None:
                time.sleep(0.2)
                continue

            # Extraer regiones
            gps_frame = get_region_frame(frame_bgr, REGIONS["gps"])
            left_frame = get_region_frame(frame_bgr, REGIONS["mirror_left"]) if SHOW_MIRRORS else None
            right_frame = get_region_frame(frame_bgr, REGIONS["mirror_right"]) if SHOW_MIRRORS else None

            # FPS
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps_display = frame_count
                frame_count = 0
                prev_time = now

            with self._lock:
                self.gps_img = gps_frame
                if SHOW_MIRRORS:
                    self.left_img = left_frame
                    self.right_img = right_frame
                self.fps = fps_display

            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / FPS_LIMIT) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _schedule_update(self):
        self.root.after(25, self._update_gui)

    def _pil_from_frame(self, frame):
        if frame is None or frame.size == 0:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _update_gui(self):
        if not self._running:
            return
        with self._lock:
            gps = self.gps_img.copy() if self.gps_img is not None else None
            left = self.left_img.copy() if SHOW_MIRRORS and self.left_img is not None else None
            right = self.right_img.copy() if SHOW_MIRRORS and self.right_img is not None else None
            fps = self.fps

        total_w = 0
        total_h = 0

        # GPS (principal)
        if gps is not None:
            if SCALE != 1.0:
                gps = cv2.resize(gps, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            gps_pil = self._pil_from_frame(gps)
            if gps_pil:
                gps_tk = ImageTk.PhotoImage(image=gps_pil)
                self.lbl_gps.imgtk = gps_tk
                self.lbl_gps.configure(image=gps_tk)
                total_w = max(total_w, gps_pil.width)
                total_h += gps_pil.height

        # Retrovisores
        if SHOW_MIRRORS:
            if left is not None:
                left = cv2.resize(left, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                left_pil = self._pil_from_frame(left)
                if left_pil:
                    left_tk = ImageTk.PhotoImage(image=left_pil)
                    self.lbl_left.imgtk = left_tk
                    self.lbl_left.configure(image=left_tk)

            if right is not None:
                right = cv2.resize(right, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                right_pil = self._pil_from_frame(right)
                if right_pil:
                    right_tk = ImageTk.PhotoImage(image=right_pil)
                    self.lbl_right.imgtk = right_tk
                    self.lbl_right.configure(image=right_tk)

        # Info bar y geometria
        self.lbl_info.configure(text=f"Nav FPS:{fps} | Arrastra para mover | X para cerrar")

        # Calcular geometria de la ventana
        gps_w = gps.shape[1] if gps is not None else 0
        gps_h = gps.shape[0] if gps is not None else 0
        if SHOW_MIRRORS and left is not None:
            # 2 columnas: mirrors arriba, gps abajo
            mirror_w = left.shape[1] if left is not None else 0
            total_w = max(gps_w, mirror_w * 2 + 20)
            total_h = gps_h + (left.shape[0] if left is not None else 0) + 20
        else:
            total_w = gps_w + 10
            total_h = gps_h + 20

        if total_w > 0 and total_h > 0:
            self.root.geometry(f"{total_w}x{total_h}")
            self.btn_close.place(x=total_w - 20, y=0)

        self._schedule_update()

    def on_close(self):
        self._running = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()
