#!/usr/bin/env python3
"""
ETS2 Road Detector Overlay - YOLOP ONNX (macOS)

Usa modelo YOLOP 640x640 para detectar carretera y carriles en ETS2.
Captura la ventana especifica del juego via Quartz y muestra overlay flotante.
"""

import os
import threading
import time
import tkinter as tk
from tkinter import Label, Button
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageTk

# -----------------------------------------------------------------------------
# CONFIGURACION
# -----------------------------------------------------------------------------
FPS_LIMIT = 30
SHOW_LANES = True
DISPLAY_MODE = "overlay"   # "overlay" | "mask" | "split" | "debug"

ROAD_ALPHA = 1.0
LANE_ALPHA = 1.0

COLOR_ROAD = np.array([0, 255, 0], dtype=np.uint8)
COLOR_LANE = np.array([0, 0, 255], dtype=np.uint8)

# Clases BDD100K detectadas por YOLOP
YOLOP_CLASSES = [
    "bike", "bus", "car", "motor", "person",
    "rider", "traffic light", "traffic sign", "train", "truck"
]
YOLOP_COLORS = [
    (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (128, 255, 0), (0, 128, 255), (255, 128, 0), (0, 255, 128),
    (128, 0, 255), (255, 255, 255)
]

# Mostrar deteccion de objetos (obstaculos, coches, senales)
SHOW_OBJECTS = True
CONF_THRESHOLD = 0.4
NMS_IOU_THRESHOLD = 0.5

# Resolucion del modelo. 320 = rapido, 640 = preciso.
MODEL_RES = 640  # 640 detecta mucho mejor en ETS2

# Procesar 1 de cada N frames (1 = todos, 2 = mitad, 3 = un tercio)
FRAME_SKIP = 2

# Reducir captura a esta altura antes de mandar al modelo
# (menos pixeles = inferencia mas rapida)
CAPTURE_MAX_H = 480

# Ventanas a buscar, en orden de prioridad.
# Primero intenta nombres exactos de ETS2; Steam es ultimo recurso.
WINDOW_NAMES = ["Euro Truck Simulator 2", "Euro Truck", "eurotrucks2", "ETS2"]

MODEL_URL = (
    f"https://raw.githubusercontent.com/hustvl/YOLOP/main/weights/"
    f"yolop-{MODEL_RES}-{MODEL_RES}.onnx"
)
MODEL_PATH = f"weights/yolop-{MODEL_RES}-{MODEL_RES}.onnx"

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
    """Busca ETS2 respetando el orden de prioridad en WINDOW_NAMES."""
    windows = get_window_list()
    for target in WINDOW_NAMES:
        target_lower = target.lower()
        for w in windows:
            full_text = f"{w['owner']} {w['name']}".lower()
            if target_lower in full_text:
                b = w["bounds"]
                width = int(b.get("Width", 0))
                height = int(b.get("Height", 0))
                if width < 200 or height < 150:
                    continue
                print(f"[INFO] Ventana ETS2 encontrada: '{w['owner']}' / '{w['name']}' ({width}x{height})")
                return {
                    "id": w["id"],
                    "left": int(b.get("X", 0)),
                    "top": int(b.get("Y", 0)),
                    "width": width,
                    "height": height,
                }
    return None


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
# MODELO YOLOP
# -----------------------------------------------------------------------------
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"[INFO] Descargando modelo YOLOP {MODEL_RES}x{MODEL_RES} (~{ '8' if MODEL_RES==320 else '34' } MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Modelo guardado.")


def _resize_unscale(img, new_shape, color=114):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    canvas = np.zeros((new_shape[0], new_shape[1], 3), dtype=np.float32)
    canvas.fill(color)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    new_unpad_w, new_unpad_h = new_unpad
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h
    dw = pad_w // 2
    dh = pad_h // 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)
    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img
    return canvas, r, dw, dh, new_unpad_w, new_unpad_h


def preprocess(frame_bgr):
    height, width = frame_bgr.shape[:2]
    img_rgb = frame_bgr[:, :, ::-1].copy()
    canvas, r, dw, dh, new_unpad_w, new_unpad_h = _resize_unscale(img_rgb, (MODEL_RES, MODEL_RES))
    img = canvas.copy().astype(np.float32)
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    return img, (height, width, r, dw, dh, new_unpad_w, new_unpad_h)


def nms(boxes, scores, iou_threshold):
    """Non-maximum suppression simple en numpy."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def process_detections(det_out, meta, conf_thresh=0.4, iou_thresh=0.5):
    """
    Procesa det_out de YOLOP y retorna lista de detecciones filtradas.
    Cada deteccion: [x1, y1, x2, y2, conf, cls_id]
    """
    height, width, r, dw, dh, new_unpad_w, new_unpad_h = meta
    det = det_out[0]  # (25200, 6)

    # Filtrar por confianza
    mask = det[:, 4] > conf_thresh
    det = det[mask]
    if len(det) == 0:
        return []

    boxes = det[:, :4]
    confs = det[:, 4]
    clss = det[:, 5].astype(int)

    # Escalar coordenadas al tamano original
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r

    # Aplicar NMS por clase
    detections = []
    unique_classes = np.unique(clss)
    for cls_id in unique_classes:
        cls_mask = clss == cls_id
        cls_boxes = boxes[cls_mask]
        cls_confs = confs[cls_mask]
        keep = nms(cls_boxes, cls_confs, iou_thresh)
        for idx in keep:
            detections.append([
                int(cls_boxes[idx][0]), int(cls_boxes[idx][1]),
                int(cls_boxes[idx][2]), int(cls_boxes[idx][3]),
                float(cls_confs[idx]), int(cls_id)
            ])
    return detections


def draw_detections(frame, detections):
    """Dibuja bounding boxes y etiquetas en el frame."""
    for x1, y1, x2, y2, conf, cls_id in detections:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        color = YOLOP_COLORS[cls_id % len(YOLOP_COLORS)]
        label = f"{YOLOP_CLASSES[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def postprocess(da_seg_out, ll_seg_out, meta):
    height, width, r, dw, dh, new_unpad_w, new_unpad_h = meta
    da = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    # Probabilidad de clase "road" / "lane" (channel 1)
    da_prob = da[0, 1]  # (H, W)
    ll_prob = ll[0, 1]  # (H, W)

    da_mask = np.argmax(da, axis=1)[0]
    ll_mask = np.argmax(ll, axis=1)[0]

    da_mask = cv2.resize(da_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
    ll_mask = cv2.resize(ll_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
    da_prob = cv2.resize(da_prob.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    ll_prob = cv2.resize(ll_prob.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

    da_mask = (da_mask > 0.15).astype(np.uint8)
    ll_mask = (ll_mask > 0.15).astype(np.uint8)
    return da_mask, ll_mask, da_prob, ll_prob


# -----------------------------------------------------------------------------
# OVERLAY TKINTER
# -----------------------------------------------------------------------------
class OverlayWindow:
    def __init__(self, win_info, session):
        self.win_info = win_info
        self.session = session
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
        skip_counter = 0
        last_da_mask = None
        last_ll_mask = None
        last_da_prob = None
        last_ll_prob = None
        last_detections = []

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

            # Reducir resolucion de captura ANTES de inferencia
            if h > CAPTURE_MAX_H:
                scale = CAPTURE_MAX_H / h
                new_w = int(w * scale)
                new_h = int(h * scale)
                frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
                h, w = new_h, new_w

            # ---- INFERENCIA CON FRAME SKIPPING ----
            skip_counter += 1
            if skip_counter >= FRAME_SKIP:
                skip_counter = 0
                inp, meta = preprocess(frame_bgr)
                det_out, da_seg_out, ll_seg_out = self.session.run(
                    ["det_out", "drive_area_seg", "lane_line_seg"],
                    {"images": inp},
                )
                last_da_mask, last_ll_mask, last_da_prob, last_ll_prob = postprocess(da_seg_out, ll_seg_out, meta)
                if SHOW_OBJECTS:
                    last_detections = process_detections(det_out, meta, CONF_THRESHOLD, NMS_IOU_THRESHOLD)
                else:
                    last_detections = []

            da_mask = last_da_mask
            ll_mask = last_ll_mask
            da_prob = last_da_prob
            ll_prob = last_ll_prob
            detections = last_detections

            if da_mask is None:
                time.sleep(0.05)
                continue

            # Asegurar que las mascaras coincidan con el frame actual
            if da_mask.shape[:2] != (h, w):
                da_mask = cv2.resize(da_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
                da_mask = (da_mask > 0.15).astype(np.uint8)
                if da_prob is not None:
                    da_prob = cv2.resize(da_prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            if ll_mask is not None and ll_mask.shape[:2] != (h, w):
                ll_mask = cv2.resize(ll_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)
                ll_mask = (ll_mask > 0.15).astype(np.uint8)
                if ll_prob is not None:
                    ll_prob = cv2.resize(ll_prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

            # ---- CONSTRUIR VISUALIZACION ----
            if self._display_mode == "debug":
                result = frame_bgr.copy()
                if da_prob is not None:
                    heatmap = (np.clip(da_prob, 0, 1) * 255).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    result = cv2.addWeighted(result, 0.4, heatmap_color, 0.6, 0)
                if ll_prob is not None and SHOW_LANES:
                    lane_heatmap = (np.clip(ll_prob, 0, 1) * 255).astype(np.uint8)
                    lane_color = cv2.applyColorMap(lane_heatmap, cv2.COLORMAP_HOT)
                    result = cv2.addWeighted(result, 0.7, lane_color, 0.3, 0)

            elif self._display_mode == "mask":
                result = np.zeros((h, w, 3), dtype=np.uint8)
                result[da_mask == 1] = COLOR_ROAD
                if SHOW_LANES:
                    result[ll_mask == 1] = COLOR_LANE

            elif self._display_mode == "split":
                result = frame_bgr.copy()
                mid = w // 2
                overlay = np.zeros_like(result)
                overlay[da_mask == 1] = COLOR_ROAD
                if SHOW_LANES:
                    overlay[ll_mask == 1] = COLOR_LANE
                right = cv2.addWeighted(result[:, mid:, :], 1.0, overlay[:, mid:, :], ROAD_ALPHA, 0)
                result[:, mid:, :] = right

            else:  # overlay
                result = frame_bgr.copy()
                result[da_mask == 1] = COLOR_ROAD
                if SHOW_LANES:
                    result[ll_mask == 1] = COLOR_LANE

            # Dibujar detecciones de objetos
            if SHOW_OBJECTS and detections:
                result = draw_detections(result, detections)

            if np.sum(da_mask) == 0 and np.sum(ll_mask) == 0:
                cv2.putText(result, "NO DETECTA", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            da_max = float(np.max(da_prob)) if da_prob is not None else 0.0
            ll_max = float(np.max(ll_prob)) if ll_prob is not None else 0.0

            # ---- FPS ----
            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps_display = frame_count
                frame_count = 0
                prev_time = now

            obj_count = len(detections)
            obj_summary = ""
            if SHOW_OBJECTS and detections:
                cls_counts = {}
                for *_, cls_id in detections:
                    cls_counts[YOLOP_CLASSES[cls_id]] = cls_counts.get(YOLOP_CLASSES[cls_id], 0) + 1
                obj_summary = ", ".join([f"{n}:{c}" for n, c in cls_counts.items()])

            with self._lock:
                self.frame_result = result
                self.fps = fps_display
                self.road_pixels = int(np.sum(da_mask))
                self.lane_pixels = int(np.sum(ll_mask))
                self.da_max = da_max
                self.ll_max = ll_max
                self.obj_count = obj_count
                self.obj_summary = obj_summary

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

            road_pixels = getattr(self, 'road_pixels', 0)
            lane_pixels = getattr(self, 'lane_pixels', 0)
            da_max = getattr(self, 'da_max', 0.0)
            ll_max = getattr(self, 'll_max', 0.0)
            obj_count = getattr(self, 'obj_count', 0)
            obj_summary = getattr(self, 'obj_summary', "")
            info_text = f"FPS:{fps} | objs:{obj_count}"
            if obj_summary:
                info_text += f" | {obj_summary}"
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
    print("  ETS2 Road Detector Overlay - YOLOP ONNX (Optimizado)")
    print("=" * 60)
    print()

    ensure_model()

    print("[INFO] Cargando modelo ONNX...")
    ort.set_default_logger_severity(4)

    providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    print(f"[INFO] Modelo listo. Input: {session.get_inputs()[0].shape}")
    print(f"[INFO] Providers activos: {session.get_providers()}")
    print(f"[INFO] Config: res={MODEL_RES}, frame_skip={FRAME_SKIP}, capture_max_h={CAPTURE_MAX_H}")

    print("[INFO] Buscando ventana de Euro Truck Simulator 2...")
    win_info = find_ets2_window()
    if win_info:
        print(f"[INFO] Ventana detectada:")
        print(f"       ID: {win_info['id']}")
        print(f"       Pos: ({win_info['left']}, {win_info['top']})")
        print(f"       Size: {win_info['width']}x{win_info['height']}")
    else:
        print("[WARN] No se detecto la ventana automaticamente.")
        print("[INFO] Usando seleccion manual...")
        win_info = select_region_manual()

    print("[INFO] Iniciando overlay flotante...")
    overlay = OverlayWindow(win_info, session)
    overlay.run()
    print("[INFO] Cerrado.")


if __name__ == "__main__":
    main()
