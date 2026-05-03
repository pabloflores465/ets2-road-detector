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
        # Solo el panel de mapa (derecha), cortando TODO el tablero de controles
        "left": 0.62, "top": 0.56,
        "width": 0.37, "height": 0.43,
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
# ESTRATEGIA: detectar TODO lo brillante contra fondo negro, luego clasificar por hue.
# El GPS de ETS2 tiene fondo NEGRO con patron de malla. Los elementos brillantes
# destacan por alto V y S. Los grises de las carreteras tienen S baja.
HSV_RANGES = {
    # Azul cielo (flecha posicion del jugador)
    # H: 85-125 cubre azul cielo a azul cyan
    "blue":   ([85,  125,  50, 255,  60, 255], (255, 255,   0)),
    # Rojo/naranja/marron (ruta a seguir)
    # H: 0-25 y 155-180 cubre rojo, naranja, marron rojizo
    "red":    ([0,    30,  40, 255,  40, 255], (0,   255, 255)),
    "red2":   ([150, 180,  40, 255,  40, 255], (0,   255, 255)),
    # Verde lima/verde claro (flechas de direccion del giro)
    # H: 30-80 cubre verde lima a verde esmeralda
    "green":  ([30,   80,  40, 255,  50, 255], (0,   255,   0)),
    # Amarillo/naranja (centro de la flecha azul, lineas amarillas)
    "yellow": ([15,   35,  60, 255,  80, 255], (0,   255, 255)),
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
    Detecta elementos del GPS por contraste contra fondo negro.
    Estrategia: el fondo es oscuro (V<40). Todo lo con V>40 y S>30 es un elemento.
    Luego clasificar por hue en rangos amplios.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr, {}

    result = frame_bgr.copy()
    h, w = result.shape[:2]
    info = {"blue": 0, "red": 0, "green": 0, "yellow": 0, "text": 0}

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Paso 1: excluir fondo oscuro y grises de carretera
    # Fondo del GPS: V<30, patrón negro
    # Carreteras grises: S<30
    # Elementos brillantes: V>40 y S>25
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]
    mask_bright = (v_channel > 40) & (s_channel > 25)
    mask_bright = mask_bright.astype(np.uint8) * 255

    # Paso 2: clasificar brillantes por hue
    h_channel = hsv[:, :, 0].astype(np.int16)

    # Azul: H 80-130
    mask_blue = mask_bright.copy()
    mask_blue[(h_channel < 80) | (h_channel > 130)] = 0
    # Rojo/naranja: H 0-30 o 150-180
    mask_red = mask_bright.copy()
    mask_red[((h_channel > 30) & (h_channel < 150))] = 0
    # Verde: H 25-85
    mask_green = mask_bright.copy()
    mask_green[(h_channel < 25) | (h_channel > 85)] = 0
    # Amarillo: H 10-35
    mask_yellow = mask_bright.copy()
    mask_yellow[(h_channel < 10) | (h_channel > 35)] = 0

    kernel = np.ones((3, 3), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)

    # --- Azul (flecha posicion) ---
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 25 < area < 10000:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + bw, y + bh), (255, 255, 0), 2)
            cv2.putText(result, "POS", (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            info["blue"] += 1

    # --- Rojo/naranja (ruta) ---
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel_close)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 25:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            info["red"] += 1

    # --- Verde (flechas direccion) ---
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 8 < area < 5000:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(result, "DIR", (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            info["green"] += 1

    # --- Amarillo (centro flecha, lineas amarillas) ---
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 15 < area < 5000:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 255), 1)
            info["yellow"] += 1

    # --- Texto blanco (alto V, muy baja S) ---
    mask_text = (v_channel > 160) & (s_channel < 40)
    mask_text = mask_text.astype(np.uint8) * 255
    mask_text = cv2.morphologyEx(mask_text, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask_text, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    text_boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if 10 < bw < 250 and 6 < bh < 50 and y < h * 0.35:
            text_boxes.append((x, y, bw, bh))
    if text_boxes:
        xs = [b[0] for b in text_boxes]
        ys = [b[1] for b in text_boxes]
        x2s = [b[0] + b[2] for b in text_boxes]
        y2s = [b[1] + b[3] for b in text_boxes]
        cv2.rectangle(result, (min(xs) - 2, min(ys) - 2), (max(x2s) + 2, max(y2s) + 2), (255, 255, 255), 2)
        cv2.putText(result, "INFO", (min(xs), min(ys) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        info["text"] = len(text_boxes)

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
        # Exposed for autopilot (thread-safe read via _lock)
        self.last_gps_crop = None
        self.last_gps_info = {}

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
                self.last_gps_crop = gps_frame.copy() if gps_frame is not None else None
                self.last_gps_info = dict(gps_info)

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
        y = gps_info.get("yellow", 0)
        t = gps_info.get("text", 0)
        info_text = f"Nav FPS:{fps} | pos:{b} ruta:{r} dir:{g} y:{y} info:{t}"
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
