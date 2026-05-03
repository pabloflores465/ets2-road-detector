#!/usr/bin/env python3
"""
ETS2 Object Detector - Google Coral Edge TPU Edition

Intenta usar un Coral USB Accelerator para deteccion de objetos en ETS2.
Si no hay Coral disponible, muestra instrucciones de instalacion.

IMPORTANTE - Limitaciones:
  - macOS Intel: funciona con libedgetpu installado
  - macOS Apple Silicon (M1/M2/M3): NO soportado oficialmente por Google
  - Linux: funciona perfecto
  - Windows: funciona con drivers
"""

import os
import threading
import time
import tkinter as tk
from tkinter import Label, Button

import cv2
import numpy as np
from PIL import Image, ImageTk

# Importar funciones de captura (sin dependencias de ONNX/TFLite)
from ets2_capture import (
    find_ets2_window, capture_window_quartz, capture_fallback,
    select_region_manual
)

# -----------------------------------------------------------------------------
# CONFIGURACION
# -----------------------------------------------------------------------------
FPS_LIMIT = 30
CAPTURE_MAX_H = 480

# Modelo Coral (descargado en coral_models/)
CORAL_MODEL = "coral_models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
CORAL_LABELS = "coral_models/coco_labels.txt"

# Umbral de confianza
CONF_THRESHOLD = 0.4

# -----------------------------------------------------------------------------
# CARGAR MODELO CORAL
# -----------------------------------------------------------------------------
def load_coral_model():
    """Carga el modelo TFLite compilado para Edge TPU."""
    try:
        import tflite_runtime.interpreter as tflite
        from pycoral.utils.edgetpu import make_interpreter
        from pycoral.adapters import common
        from pycoral.adapters import detect
        print("[INFO] Coral TPU detectado!")
    except ImportError as e:
        print(f"[ERROR] Falta dependencia Coral: {e}")
        print("[HINT] Ejecuta: ./coral_setup.sh")
        return None, None, None

    if not os.path.exists(CORAL_MODEL):
        print(f"[ERROR] No se encuentra modelo: {CORAL_MODEL}")
        return None, None, None

    print(f"[INFO] Cargando modelo Edge TPU: {CORAL_MODEL}")
    interpreter = make_interpreter(CORAL_MODEL)
    interpreter.allocate_tensors()

    # Cargar labels
    labels = {}
    if os.path.exists(CORAL_LABELS):
        with open(CORAL_LABELS, 'r') as f:
            for i, line in enumerate(f.readlines()):
                labels[i] = line.strip()

    return interpreter, labels, detect


def coral_inference(interpreter, detect_mod, frame_bgr, labels, conf_thresh=0.4):
    """
    Corre inferencia en el Coral TPU y retorna lista de detecciones.
    Cada deteccion: [x1, y1, x2, y2, conf, label_text]
    """
    import tflite_runtime.interpreter as tflite
    from pycoral.adapters import common

    h, w = frame_bgr.shape[:2]
    scale_w = w
    scale_h = h

    # Resize al input del modelo (tipicamente 300x300)
    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_width, input_height))

    # Inferencia
    common.set_input(interpreter, resized)
    interpreter.invoke()
    objs = detect_mod.get_objects(interpreter, score_threshold=conf_thresh)

    detections = []
    for obj in objs:
        x1 = int(obj.bbox.xmin * scale_w / input_width)
        y1 = int(obj.bbox.ymin * scale_h / input_height)
        x2 = int(obj.bbox.xmax * scale_w / input_width)
        y2 = int(obj.bbox.ymax * scale_h / input_height)
        label = labels.get(obj.id, f"id:{obj.id}")
        detections.append([x1, y1, x2, y2, obj.score, label])
    return detections


def draw_coral_detections(frame, detections):
    """Dibuja bounding boxes de Coral."""
    colors = {
        "car": (0, 255, 0), "truck": (0, 255, 255), "bus": (255, 255, 0),
        "person": (255, 0, 0), "bicycle": (255, 0, 255), "motorcycle": (128, 0, 255),
        "traffic light": (0, 128, 255), "stop sign": (0, 0, 255),
    }
    for x1, y1, x2, y2, conf, label in detections:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        color = colors.get(label, (255, 255, 255))
        text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


# -----------------------------------------------------------------------------
# OVERLAY TKINTER
# -----------------------------------------------------------------------------
class CoralOverlayWindow:
    def __init__(self, win_info, interpreter, labels, detect_mod):
        self.win_info = win_info
        self.interpreter = interpreter
        self.labels = labels
        self.detect_mod = detect_mod
        self._win_lock = threading.Lock()
        self._running = True

        self.root = tk.Tk()
        self.root.title("ETS2 Coral TPU Detection")
        self.root.attributes("-topmost", True)
        try:
            self.root.overrideredirect(True)
        except tk.TclError:
            self.root.wm_attributes("-type", "splash")
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
            text="Coral TPU iniciando...",
            fg="#00ff00", bg="black", font=("Courier", 9),
            bd=0, padx=4, pady=2,
        )

        self.root.bind("<Button-1>", self._start_move)
        self.root.bind("<B1-Motion>", self._on_move)
        self._drag_x = 0
        self._drag_y = 0

        self.frame_result = None
        self.fps = 0
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

            if info.get("id"):
                frame_bgr = capture_window_quartz(info["id"])
            else:
                frame_bgr = capture_fallback(info)

            if frame_bgr is None:
                time.sleep(0.2)
                continue

            h, w = frame_bgr.shape[:2]
            if h > CAPTURE_MAX_H:
                scale = CAPTURE_MAX_H / h
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w

            # ---- INFERENCIA CORAL ----
            detections = coral_inference(
                self.interpreter, self.detect_mod, frame_bgr,
                self.labels, CONF_THRESHOLD
            )

            result = frame_bgr.copy()
            if detections:
                result = draw_coral_detections(result, detections)
            else:
                cv2.putText(result, "SIN OBJETOS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps_display = frame_count
                frame_count = 0
                prev_time = now

            with self._lock:
                self.frame_result = result
                self.fps = fps_display
                self.obj_count = len(detections)

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
            obj_count = getattr(self, 'obj_count', 0)

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

            info_text = f"FPS:{fps} | objs:{obj_count} | Coral Edge TPU"
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
    print("  ETS2 Coral TPU Detector")
    print("=" * 60)
    print()

    # Intentar cargar Coral
    interpreter, labels, detect_mod = load_coral_model()

    if interpreter is None:
        print()
        print("=" * 60)
        print("  CORAL TPU NO DISPONIBLE")
        print("=" * 60)
        print()
        print("Posibles causas:")
        print("  1. El dispositivo USB no esta conectado")
        print("  2. Falta instalar pycoral + libedgetpu")
        print("  3. Mac Apple Silicon (M1/M2/M3) NO soportado oficialmente")
        print()
        print("Soluciones:")
        print("  - macOS Intel: ejecuta ./coral_setup.sh")
        print("  - macOS Apple Silicon: necesitas Linux (VM/Ubuntu)")
        print("  - Linux: sudo apt install libedgetpu1-std python3-pycoral")
        print("  - Asegurate de usar un hub USB con alimentacion externa")
        print()
        print("Usando detector ONNX en su lugar...")
        print("  python road_detector.py")
        return

    print("[INFO] Buscando ventana de Euro Truck Simulator 2...")
    win_info = find_ets2_window()
    if win_info:
        print(f"[INFO] Ventana detectada: {win_info['width']}x{win_info['height']}")
    else:
        print("[INFO] Usando seleccion manual...")
        win_info = select_region_manual()

    print("[INFO] Iniciando overlay Coral TPU...")
    overlay = CoralOverlayWindow(win_info, interpreter, labels, detect_mod)
    overlay.run()
    print("[INFO] Cerrado.")


if __name__ == "__main__":
    main()
