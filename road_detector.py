#!/usr/bin/env python3
"""
ETS2 Road Detector Overlay - OpenCV Edition (Optimizado)

Detecta carretera y carriles con OpenCV (Canny + Hough Lines + ROI),
captura la ventana de Euro Truck Simulator 2 via Quartz,
y muestra overlay flotante siempre encima.

Mas rapido que ONNX/YOLOP para este caso de uso especifico.
"""

import os
import subprocess
import threading
import time
import tkinter as tk
from tkinter import Label, Button

import cv2
import numpy as np
from PIL import Image, ImageTk

# -----------------------------------------------------------------------------
# CONFIGURACION
# -----------------------------------------------------------------------------
FPS_LIMIT = 30
SHOW_LANES = True
DISPLAY_MODE = "overlay"   # "overlay" | "mask" | "split"

# Colores overlay (BGR)
COLOR_ROAD = (0, 255, 0)      # verde
COLOR_LANE = (0, 0, 255)      # rojo

# Area de interes para deteccion (relativo al frame)
ROI_TOP = 0.50
ROI_BOTTOM = 1.0
ROI_LEFT = 0.10
ROI_RIGHT = 0.90

# Canny
CANNY_LOW = 50
CANNY_HIGH = 150

# Hough Lines
HOUGH_THRESHOLD = 50
HOUGH_MIN_LEN = 40
HOUGH_MAX_GAP = 100

# Nombres de ventana a buscar
WINDOW_NAMES = ["Euro Truck", "eurotrucks2", "ETS2", "Steam"]

# Reducir captura antes de procesar (menos pixeles = mas rapido)
CAPTURE_MAX_H = 540

# -----------------------------------------------------------------------------
# QUARTZ (macOS native)
# -----------------------------------------------------------------------------
def get_window_list():
    try:
        import Quartz
    except ImportError:
        return []
    window_list = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionAll,
        Quartz.kCGNullWindowID
    )
    result = []
    for win in window_list:
        win_id = win.get(Quartz.kCGWindowNumber, 0)
        owner = win.get(Quartz.kCGWindowOwnerName, "")
        name = win.get(Quartz.kCGWindowName, "")
        bounds = win.get(Quartz.kCGWindowBounds, {})
        result.append({
            "id": win_id,
            "owner": owner,
            "name": name,
            "bounds": bounds,
        })
    return result


def find_ets2_window():
    windows = get_window_list()
    candidates = []
    for w in windows:
        full_text = f"{w['owner']} {w['name']}".lower()
        for target in WINDOW_NAMES:
            if target.lower() in full_text:
                b = w["bounds"]
                width = int(b.get("Width", 0))
                height = int(b.get("Height", 0))
                if width < 200 or height < 150:
                    continue
                candidates.append({
                    "id": w["id"],
                    "left": int(b.get("X", 0)),
                    "top": int(b.get("Y", 0)),
                    "width": width,
                    "height": height,
                    "area": width * height,
                })
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["area"], reverse=True)
    best = candidates[0]
    return {
        "id": best["id"],
        "left": best["left"],
        "top": best["top"],
        "width": best["width"],
        "height": best["height"],
    }


def capture_window_quartz(win_id):
    try:
        import Quartz
    except ImportError:
        return None
    try:
        image = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionIncludingWindow,
            win_id,
            Quartz.kCGWindowImageBoundsIgnoreFraming
        )
        if image is None:
            return None
        width = Quartz.CGImageGetWidth(image)
        height = Quartz.CGImageGetHeight(image)
        bytesperrow = Quartz.CGImageGetBytesPerRow(image)
        cfdata = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
        buf = np.frombuffer(cfdata, dtype=np.uint8)
        if bytesperrow == width * 4:
            arr = buf.reshape((height, width, 4))
        else:
            arr = buf[:height * bytesperrow].reshape((height, bytesperrow))
            arr = arr[:, :width * 4].reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        print(f"[WARN] Quartz capture failed: {e}")
        return None


def select_region_manual():
    import mss
    print("[INFO] Capturando pantalla para seleccion manual...")
    with mss.MSS() as sct:
        mon = sct.monitors[1]
        screenshot = np.array(sct.grab(mon))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    print("[INFO] Arrastra para seleccionar la ventana de ETS2.")
    print("       ESPACIO/ENTER = confirmar | C = cancelar")
    roi = cv2.selectROI("Selecciona ventana ETS2", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Selecciona ventana ETS2")
    x, y, w, h = roi
    if w == 0 or h == 0:
        print("[ERROR] No se selecciono region.")
        exit(1)
    with mss.MSS() as sct:
        mon = sct.monitors[1]
    return {
        "id": None,
        "left": int(x + mon["left"]),
        "top": int(y + mon["top"]),
        "width": int(w),
        "height": int(h),
    }


def capture_fallback(region):
    import mss
    try:
        with mss.MSS() as sct:
            img = sct.grab(region)
            return cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# DETECCION DE CARRIL (OpenCV)
# -----------------------------------------------------------------------------
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1 * 0.6)
    if slope == 0:
        return np.array([0, y1, 0, y2])
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            continue
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if abs(slope) < 0.3:
            continue
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_line = None
    right_line = None
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
    return left_line, right_line


def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(image, mask)


def detect_road_cv(frame):
    """
    Detecta carriles/carretera con OpenCV.
    Retorna: (frame_con_overlay, left_line, right_line)
    """
    h, w = frame.shape[:2]

    # Vertices ROI (trapecio inferior centrado)
    vertices = np.array([
        [
            (int(w * ROI_LEFT), int(h * ROI_BOTTOM)),
            (int(w * 0.45), int(h * ROI_TOP)),
            (int(w * 0.55), int(h * ROI_TOP)),
            (int(w * ROI_RIGHT), int(h * ROI_BOTTOM)),
        ]
    ], dtype=np.int32)

    # 1. Escala de grises y blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Bordes Canny
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # 3. Mascara ROI sobre bordes
    masked_edges = region_of_interest(edges, vertices)

    # 4. Hough Lines
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        lines=np.array([]),
        minLineLength=HOUGH_MIN_LEN,
        maxLineGap=HOUGH_MAX_GAP,
    )

    # 5. Promediar lineas izquierda / derecha
    left_line, right_line = average_slope_intercept(frame, lines)

    # 6. Dibujar overlay
    overlay = np.zeros_like(frame)
    line_image = np.zeros_like(frame)

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(line_image, (x1, y1), (x2, y2), COLOR_LANE, 10)
    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(line_image, (x1, y1), (x2, y2), COLOR_LANE, 10)

    # Rellenar area de carretera entre los dos carriles
    if left_line is not None and right_line is not None:
        pts = np.array([
            [left_line[0], left_line[1]],
            [left_line[2], left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]],
        ], np.int32)
        cv2.fillPoly(overlay, [pts], COLOR_ROAD)

    # Combinar: original + area carretera + lineas
    result = cv2.addWeighted(frame, 1.0, overlay, 0.40, 0)
    result = cv2.addWeighted(result, 1.0, line_image, 1.0, 0)

    # Debug: dibujar ROI en rojo tenue
    cv2.polylines(result, vertices, True, (0, 0, 255), 2)

    return result, left_line, right_line


# -----------------------------------------------------------------------------
# OVERLAY TKINTER
# -----------------------------------------------------------------------------
class OverlayWindow:
    def __init__(self, win_info):
        self.win_info = win_info
        self._win_lock = threading.Lock()
        self._running = True
        self._display_mode = DISPLAY_MODE

        self.root = tk.Tk()
        self.root.title("ETS2 Road Detection")
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)
        self.root.configure(bg="black", highlightthickness=0)

        self.btn_close = Button(
            self.root, text="X", command=self.on_close,
            bg="#cc0000", fg="white", font=("Arial", 8, "bold"),
            bd=0, padx=5, pady=1, cursor="hand2"
        )
        self.btn_close.place(x=0, y=0)

        self.lbl_main = Label(self.root, bg="black", bd=0)
        self.lbl_main.place(x=0, y=0)

        self.lbl_info = Label(
            self.root,
            text="Iniciando...",
            fg="#00ff00", bg="black", font=("Courier", 9),
            bd=0, padx=4, pady=2,
        )

        self.root.bind("<Button-1>", self._start_move)
        self.root.bind("<B1-Motion>", self._on_move)
        self._drag_x = 0
        self._drag_y = 0

        self.frame_result = None
        self.fps = 0
        self.left_line = None
        self.right_line = None
        self._lock = threading.Lock()

        self.thread_capture = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread_capture.start()

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

            # ---- CAPTURA ----
            if info.get("id"):
                frame_bgr = capture_window_quartz(info["id"])
            else:
                frame_bgr = capture_fallback(info)

            if frame_bgr is None:
                time.sleep(0.2)
                continue

            h, w = frame_bgr.shape[:2]

            # Reducir resolucion ANTES de procesar
            if h > CAPTURE_MAX_H:
                scale = CAPTURE_MAX_H / h
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w

            # ---- DETECCION OpenCV ----
            result, left_line, right_line = detect_road_cv(frame_bgr)

            # ---- FPS ----
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps_display = frame_count
                frame_count = 0
                prev_time = now

            with self._lock:
                self.frame_result = result
                self.fps = fps_display
                self.left_line = left_line is not None
                self.right_line = right_line is not None

            elapsed = time.time() - loop_start
            sleep_time = max(0, (1.0 / FPS_LIMIT) - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _schedule_update(self):
        self.root.after(25, self._update_gui)

    def _update_gui(self):
        if not self._running:
            return
        with self._lock:
            frame = self.frame_result.copy() if self.frame_result is not None else None
            fps = self.fps
            has_left = self.left_line
            has_right = self.right_line

        if frame is not None:
            h, w = frame.shape[:2]

            max_w = 1440
            max_h = 900
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                new_w, new_h = w, h

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_main.imgtk = imgtk
            self.lbl_main.configure(image=imgtk)
            self.lbl_main.place(x=0, y=0, width=new_w, height=new_h)

            l = "L" if has_left else "-"
            r = "R" if has_right else "-"
            info_text = f"FPS:{fps} | carriles:{l}{r} | Arrastra para mover"
            self.lbl_info.configure(text=info_text)
            self.lbl_info.place(x=0, y=new_h, width=new_w, height=18)

            self.root.geometry(f"{new_w}x{new_h + 18}")
            self.btn_close.place(x=new_w - 22, y=0)

        self._schedule_update()

    def on_close(self):
        self._running = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  ETS2 Road Detector Overlay - OpenCV Optimizado")
    print("=" * 60)
    print()

    print("[INFO] Buscando ventana de Euro Truck Simulator 2...")
    win_info = find_ets2_window()
    if win_info:
        print(f"[INFO] Ventana detectada:")
        print(f"       ID: {win_info['id']}")
        print(f"       Pos: ({win_info['left']}, {win_info['top']})")
        print(f"       Size: {win_info['width']}x{win_info['height']}")
    else:
        print("[WARN] No se detecto automaticamente.")
        print("[INFO] Usando seleccion manual...")
        win_info = select_region_manual()

    print("[INFO] Iniciando overlay flotante...")
    overlay = OverlayWindow(win_info)
    overlay.run()
    print("[INFO] Cerrado.")


if __name__ == "__main__":
    main()
