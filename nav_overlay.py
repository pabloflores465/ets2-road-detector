#!/usr/bin/env python3
"""
ETS2 Navigation & Mirrors Overlay

Extrae y muestra en ventana flotante siempre encima:
  - Panel GPS / Route Advisor (esquina inferior derecha) con deteccion de:
      * Flecha azul  = posicion actual
      * Linea roja   = ruta a seguir
      * Flechas verdes = direccion del giro
      * Texto de kilometraje
  - Retrovisor izquierdo (esquina superior izquierda)
  - Retrovisor derecho (esquina superior derecha)
"""

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
REGIONS = {
    "gps": {
        # Solo el panel de mapa (derecha), SIN el panel de controles (izquierda)
        "left": 0.55, "top": 0.56,
        "width": 0.44, "height": 0.43,
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

# Escala de salida del GPS (0.5 = mitad de tamano)
GPS_SCALE = 0.5

# Mostrar retrovisores?
SHOW_MIRRORS = True

# Colores HSV para deteccion en el GPS (OpenCV HSV)
# Ajustados a los colores reales de ETS2
HSV_RANGES = {
    # Azul cian brillante (flecha posicion) — muy saturado, brillo alto
    "blue":   ([95,  115, 150, 255, 150, 255], (255, 255,   0)),
    # Rojo anaranjado brillante (ruta) — saturacion alta
    "red":    ([0,    15, 120, 255, 120, 255], (0,   255, 255)),
    "red2":   ([165, 180, 120, 255, 120, 255], (0,   255, 255)),
    # Verde lima brillante (flechas direccion)
    "green":  ([40,   75, 120, 255, 120, 255], (0,   255,   0)),
}


def get_region_frame(frame, region_spec):
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


def process_gps(frame_bgr):
    """
    Detecta y resalta elementos del panel GPS:
      - Flecha azul  (posicion)
      - Linea roja   (ruta)
      - Flechas verdes (direccion)
    Retorna frame procesado + dict con info detectada.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr, {}

    result = frame_bgr.copy()
    h, w = result.shape[:2]
    info = {"blue": 0, "red": 0, "green": 0}

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # --- Azul (flecha posicion — triangulo azul cian) ---
    lower = np.array(HSV_RANGES["blue"][0][:3])
    upper = np.array(HSV_RANGES["blue"][0][3:])
    mask_blue = cv2.inRange(hsv, lower, upper)
    # Operaciones morfologicas para eliminar ruido pequeno
    kernel = np.ones((3, 3), np.uint8)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 5000:  # La flecha azul es un area mediana
            x, y, bw, bh = cv2.boundingRect(cnt)
            # La flecha es triangular (aspect ratio ~1.0)
            aspect = bw / max(bh, 1)
            if 0.5 < aspect < 2.0:
                cv2.rectangle(result, (x, y), (x + bw, y + bh), (255, 255, 0), 2)
                cv2.putText(result, "POS", (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                info["blue"] += 1

    # --- Rojo (ruta — linea ancha y larga) ---
    lower1 = np.array(HSV_RANGES["red"][0][:3])
    upper1 = np.array(HSV_RANGES["red"][0][3:])
    lower2 = np.array(HSV_RANGES["red2"][0][:3])
    upper2 = np.array(HSV_RANGES["red2"][0][3:])
    mask_red = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 80:  # La ruta es un area grande
            x, y, bw, bh = cv2.boundingRect(cnt)
            # La ruta es larga y relativamente delgada
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 1.5 or area > 500:
                cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
                info["red"] += 1

    # --- Verde (flechas direccion — pequenas triangulares) ---
    lower = np.array(HSV_RANGES["green"][0][:3])
    upper = np.array(HSV_RANGES["green"][0][3:])
    mask_green = cv2.inRange(hsv, lower, upper)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 20 < area < 1500:  # Flechas pequenas
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(result, "DIR", (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            info["green"] += 1

    return result, info


# -----------------------------------------------------------------------------
# NAV OVERLAY (para usar dentro de un Toplevel)
# -----------------------------------------------------------------------------
class NavOverlayManager:
    def __init__(self, toplevel, win_info):
        self.root = toplevel
        self.win_info = win_info
        self._win_lock = threading.Lock()
        self._running = True

        self.btn_close = Button(
            self.root, text="X", command=self.on_close,
            bg="#cc0000", fg="white", font=("Arial", 8, "bold"),
            bd=0, padx=4, pady=1, cursor="hand2"
        )
        self.btn_close.place(x=0, y=0)

        self.lbl_gps = Label(self.root, bg="black", bd=0)
        self.lbl_gps.grid(row=1, column=0, columnspan=2, padx=2, pady=2)

        if SHOW_MIRRORS:
            self.lbl_left = Label(self.root, bg="black", bd=0)
            self.lbl_left.grid(row=0, column=0, padx=2, pady=2)

            self.lbl_right = Label(self.root, bg="black", bd=0)
            self.lbl_right.grid(row=0, column=1, padx=2, pady=2)

        self.lbl_info = Label(
            self.root, text="Nav Overlay", fg="#00ff00", bg="black",
            font=("Courier", 8), bd=0, padx=4, pady=2,
        )
        self.lbl_info.grid(row=2, column=0, columnspan=2, sticky="w")

        self.root.bind("<Button-1>", self._start_move)
        self.root.bind("<B1-Motion>", self._on_move)
        self._drag_x = 0
        self._drag_y = 0

        self._lock = threading.Lock()
        self.gps_img = None
        self.left_img = None
        self.right_img = None
        self.gps_info = {}
        self.fps = 0

        self._tk_gps = None
        self._tk_left = None
        self._tk_right = None

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        self._schedule_update()

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

            frame_bgr = capture_window_quartz(info["id"]) if info.get("id") else capture_fallback(info)
            if frame_bgr is None:
                time.sleep(0.2)
                continue

            gps_frame = get_region_frame(frame_bgr, REGIONS["gps"])
            left_frame = get_region_frame(frame_bgr, REGIONS["mirror_left"]) if SHOW_MIRRORS else None
            right_frame = get_region_frame(frame_bgr, REGIONS["mirror_right"]) if SHOW_MIRRORS else None

            # Procesar GPS: detectar colores y resaltar
            gps_processed, gps_info = process_gps(gps_frame)

            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps_display = frame_count
                frame_count = 0
                prev_time = now

            with self._lock:
                self.gps_img = gps_processed
                if SHOW_MIRRORS:
                    self.left_img = left_frame
                    self.right_img = right_frame
                self.gps_info = gps_info
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
            gps_info = dict(self.gps_info)

        total_w = 0
        total_h = 0

        # GPS: escalar a tamano pequeno
        if gps is not None:
            gps = cv2.resize(gps, None, fx=GPS_SCALE, fy=GPS_SCALE, interpolation=cv2.INTER_AREA)
            gps_pil = self._pil_from_frame(gps)
            if gps_pil:
                self._tk_gps = ImageTk.PhotoImage(image=gps_pil)
                self.lbl_gps.configure(image=self._tk_gps)
                total_w = max(total_w, gps_pil.width)
                total_h += gps_pil.height

        # Retrovisores
        if SHOW_MIRRORS:
            if left is not None:
                left = cv2.resize(left, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
                left_pil = self._pil_from_frame(left)
                if left_pil:
                    self._tk_left = ImageTk.PhotoImage(image=left_pil)
                    self.lbl_left.configure(image=self._tk_left)

            if right is not None:
                right = cv2.resize(right, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
                right_pil = self._pil_from_frame(right)
                if right_pil:
                    self._tk_right = ImageTk.PhotoImage(image=right_pil)
                    self.lbl_right.configure(image=self._tk_right)

        # Info bar
        b = gps_info.get("blue", 0)
        r = gps_info.get("red", 0)
        g = gps_info.get("green", 0)
        info_text = f"Nav FPS:{fps} | pos:{b} ruta:{r} dir:{g}"
        self.lbl_info.configure(text=info_text)

        # Geometria
        gps_w = gps.shape[1] if gps is not None else 0
        gps_h = gps.shape[0] if gps is not None else 0
        if SHOW_MIRRORS and left is not None:
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
