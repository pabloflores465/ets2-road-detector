#!/usr/bin/env python3
"""
ETS2 Unified Detector — YOLOP ONNX + Coral Edge TPU

Dos ventanas flotantes siempre encima:
  1. Deteccion de carretera, carriles y objetos
  2. Panel GPS + retrovisores
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

from ets2_capture import find_ets2_window, capture_window_quartz, capture_fallback, select_region_manual
from nav_overlay import NavOverlayManager
from autopilot import Autopilot
from pynput import keyboard as pynput_kb

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

MODEL_RES = 640
FRAME_SKIP = 2
CAPTURE_MAX_H = 480

SHOW_OBJECTS = True
YOLOP_CONF = 0.4
NMS_IOU = 0.5

USE_CORAL = True
CORAL_MODEL = "coral_models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
CORAL_LABELS = "coral_models/coco_labels.txt"
CORAL_CONF = 0.4

MODEL_URL = (
    f"https://raw.githubusercontent.com/hustvl/YOLOP/main/weights/"
    f"yolop-{MODEL_RES}-{MODEL_RES}.onnx"
)
MODEL_PATH = f"weights/yolop-{MODEL_RES}-{MODEL_RES}.onnx"

YOLOP_CLASSES = [
    "bike", "bus", "car", "motor", "person",
    "rider", "traffic light", "traffic sign", "train", "truck"
]
YOLOP_COLORS = [
    (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
    (128, 255, 0), (0, 128, 255), (255, 128, 0), (0, 255, 128),
    (128, 0, 255), (255, 255, 255)
]


# -----------------------------------------------------------------------------
# CORAL TPU
# -----------------------------------------------------------------------------
def load_coral():
    if not USE_CORAL:
        return None, None, None
    try:
        from pycoral.utils.edgetpu import make_interpreter
        from pycoral.adapters import detect as coral_detect_mod
    except ImportError as e:
        print(f"[INFO] Coral no disponible: {e}")
        return None, None, None

    if not os.path.exists(CORAL_MODEL):
        print(f"[INFO] Modelo Coral no encontrado: {CORAL_MODEL}")
        return None, None, None

    print("[INFO] Cargando Coral Edge TPU...")
    interpreter = make_interpreter(CORAL_MODEL)
    interpreter.allocate_tensors()

    labels = {}
    if os.path.exists(CORAL_LABELS):
        with open(CORAL_LABELS, 'r') as f:
            for i, line in enumerate(f.readlines()):
                labels[i] = line.strip()

    print("[INFO] Coral TPU listo.")
    return interpreter, labels, coral_detect_mod


def coral_infer(interpreter, detect_mod, frame_bgr, labels, conf_thresh=0.4):
    import tflite_runtime.interpreter as tflite
    from pycoral.adapters import common

    h, w = frame_bgr.shape[:2]
    _, input_h, input_w, _ = interpreter.get_input_details()[0]['shape']
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_w, input_h))

    common.set_input(interpreter, resized)
    interpreter.invoke()
    objs = detect_mod.get_objects(interpreter, score_threshold=conf_thresh)

    detections = []
    for obj in objs:
        x1 = int(obj.bbox.xmin * w / input_w)
        y1 = int(obj.bbox.ymin * h / input_h)
        x2 = int(obj.bbox.xmax * w / input_w)
        y2 = int(obj.bbox.ymax * h / input_h)
        detections.append([x1, y1, x2, y2, obj.score, labels.get(obj.id, f"id:{obj.id}")])
    return detections


def draw_coral_dets(frame, detections):
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
        cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


# -----------------------------------------------------------------------------
# YOLOP
# -----------------------------------------------------------------------------
def ensure_model():
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"[INFO] Descargando YOLOP {MODEL_RES}x{MODEL_RES} (~34 MB)...")
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


def postprocess(da_seg_out, ll_seg_out, meta):
    height, width, r, dw, dh, new_unpad_w, new_unpad_h = meta
    da = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    da_prob = da[0, 1]
    ll_prob = ll[0, 1]
    da_mask = np.argmax(da, axis=1)[0]
    ll_mask = np.argmax(ll, axis=1)[0]
    da_mask = cv2.resize(da_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
    ll_mask = cv2.resize(ll_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
    da_prob = cv2.resize(da_prob.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    ll_prob = cv2.resize(ll_prob.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    da_mask = (da_mask > 0.15).astype(np.uint8)
    ll_mask = (ll_mask > 0.15).astype(np.uint8)
    return da_mask, ll_mask, da_prob, ll_prob


def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
        order = order[1:][iou <= iou_threshold]
    return keep


def process_yolop_dets(det_out, meta, conf_thresh=0.4, iou_thresh=0.5):
    height, width, r, dw, dh, new_unpad_w, new_unpad_h = meta
    det = det_out[0]
    det = det[det[:, 4] > conf_thresh]
    if len(det) == 0:
        return []
    boxes = det[:, :4]
    confs = det[:, 4]
    clss = det[:, 5].astype(int)
    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r
    detections = []
    for cls_id in np.unique(clss):
        mask = clss == cls_id
        k = nms(boxes[mask], confs[mask], iou_thresh)
        for idx in k:
            detections.append([
                int(boxes[mask][idx][0]), int(boxes[mask][idx][1]),
                int(boxes[mask][idx][2]), int(boxes[mask][idx][3]),
                float(confs[mask][idx]), int(cls_id)
            ])
    return detections


def draw_yolop_dets(frame, detections):
    for x1, y1, x2, y2, conf, cls_id in detections:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        color = YOLOP_COLORS[cls_id % len(YOLOP_COLORS)]
        label = f"{YOLOP_CLASSES[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


# -----------------------------------------------------------------------------
# VENTANA PRINCIPAL (DETECCION)
# -----------------------------------------------------------------------------
class MainWindow:
    def __init__(self, root, win_info, session, coral_interpreter, coral_labels, coral_detect_mod):
        self.root = root
        self.win_info = win_info
        self.session = session
        self.coral_interpreter = coral_interpreter
        self.coral_labels = coral_labels
        self.coral_detect_mod = coral_detect_mod
        self.use_coral = coral_interpreter is not None
        self._win_lock = threading.Lock()
        self._running = True
        self._display_mode = DISPLAY_MODE
        self.nav_win = None

        # Autopilot
        self.autopilot = Autopilot()

        # Frame principal
        self.frame = tk.Frame(root, bg="black")
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Forzar siempre encima periodicamente
        self._force_topmost()

        self.btn_close = Button(
            self.frame, text="X", command=self.on_close,
            bg="#cc0000", fg="white", font=("Arial", 8, "bold"),
            bd=0, padx=5, pady=1, cursor="hand2"
        )
        self.btn_close.place(x=0, y=0)

        self.btn_ap = Button(
            self.frame, text="AP:OFF", command=self._toggle_ap,
            bg="#333333", fg="#ffaa00", font=("Arial", 8, "bold"),
            bd=0, padx=4, pady=1, cursor="hand2"
        )
        self.btn_ap.place(x=24, y=0)

        self.lbl_main = Label(self.frame, bg="black", bd=0)
        self.lbl_main.place(x=0, y=0)

        self.lbl_info = Label(
            self.frame, text="Iniciando...", fg="#00ff00", bg="black",
            font=("Courier", 9), bd=0, padx=4, pady=2,
        )

        self.frame.bind("<Button-1>", self._start_move)
        self.frame.bind("<B1-Motion>", self._on_move)
        self._drag_x = 0
        self._drag_y = 0

        self.frame_result = None
        self.fps = 0
        self._tk_img = None
        self._lock = threading.Lock()

        self.thread_capture = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread_capture.start()

        self._schedule_update()

    def set_nav_overlay(self, nav_win):
        self.nav_win = nav_win

    def _on_hotkey(self, key):
        try:
            if key == pynput_kb.Key.f9:
                print("[HOTKEY] F9 pressed — toggling autopilot")
                self._toggle_ap()
        except Exception as e:
            print(f"[HOTKEY] error: {e}")

    def _toggle_ap(self):
        enabled = self.autopilot.toggle()
        self.btn_ap.configure(text=f"AP:{'ON' if enabled else 'OFF'}",
                              bg="#00aa00" if enabled else "#333333")

    def _normalize_dets(self, dets):
        """Convert detections to (x1, y1, x2, y2, label, score) for autopilot."""
        out = []
        for d in dets:
            if len(d) != 6:
                continue
            if isinstance(d[5], str):
                # Coral: [x1, y1, x2, y2, score, label]
                x1, y1, x2, y2, score, label = d
                out.append((x1, y1, x2, y2, label, score))
            else:
                # YOLOP: [x1, y1, x2, y2, score, cls_id]
                x1, y1, x2, y2, score, cls_id = d
                label = YOLOP_CLASSES[int(cls_id)] if int(cls_id) < len(YOLOP_CLASSES) else f"id:{cls_id}"
                out.append((x1, y1, x2, y2, label, score))
        return out

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
        last_da_mask = last_ll_mask = last_da_prob = last_ll_prob = None
        last_da_seg = last_ll_seg = None
        last_objs = []

        while self._running:
            loop_start = time.time()

            with self._win_lock:
                info = dict(self.win_info)

            frame_bgr = capture_window_quartz(info["id"]) if info.get("id") else capture_fallback(info)
            if frame_bgr is None:
                time.sleep(0.2)
                continue

            h, w = frame_bgr.shape[:2]
            if h > CAPTURE_MAX_H:
                scale = CAPTURE_MAX_H / h
                frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                h, w = frame_bgr.shape[:2]

            skip_counter += 1
            if skip_counter >= FRAME_SKIP:
                skip_counter = 0
                inp, meta = preprocess(frame_bgr)
                det_out, da_seg_out, ll_seg_out = self.session.run(
                    ["det_out", "drive_area_seg", "lane_line_seg"],
                    {"images": inp},
                )
                last_da_seg = da_seg_out
                last_ll_seg = ll_seg_out
                last_da_mask, last_ll_mask, last_da_prob, last_ll_prob = postprocess(da_seg_out, ll_seg_out, meta)
                if self.use_coral:
                    last_objs = coral_infer(self.coral_interpreter, self.coral_detect_mod, frame_bgr, self.coral_labels, CORAL_CONF)
                elif SHOW_OBJECTS:
                    last_objs = process_yolop_dets(det_out, meta, YOLOP_CONF, NMS_IOU)
                else:
                    last_objs = []

            da_mask = last_da_mask
            ll_mask = last_ll_mask
            da_prob = last_da_prob
            ll_prob = last_ll_prob
            da_seg_out = last_da_seg
            ll_seg_out = last_ll_seg
            objs = last_objs

            if da_mask is None:
                time.sleep(0.05)
                continue

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

            # Visualizacion
            if self._display_mode == "debug":
                result = frame_bgr.copy()
                if da_prob is not None:
                    hm = (np.clip(da_prob, 0, 1) * 255).astype(np.uint8)
                    result = cv2.addWeighted(result, 0.4, cv2.applyColorMap(hm, cv2.COLORMAP_JET), 0.6, 0)
                if ll_prob is not None and SHOW_LANES:
                    hm = (np.clip(ll_prob, 0, 1) * 255).astype(np.uint8)
                    result = cv2.addWeighted(result, 0.7, cv2.applyColorMap(hm, cv2.COLORMAP_HOT), 0.3, 0)
            elif self._display_mode == "mask":
                result = np.zeros((h, w, 3), dtype=np.uint8)
                result[da_mask == 1] = COLOR_ROAD
                if SHOW_LANES:
                    result[ll_mask == 1] = COLOR_LANE
            else:  # overlay
                result = frame_bgr.copy()
                result[da_mask == 1] = COLOR_ROAD
                if SHOW_LANES:
                    result[ll_mask == 1] = COLOR_LANE

            if self.use_coral and objs:
                result = draw_coral_dets(result, objs)
            elif objs and not self.use_coral:
                result = draw_yolop_dets(result, objs)

            if np.sum(da_mask) == 0 and np.sum(ll_mask) == 0:
                cv2.putText(result, "NO DETECTA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # --- Autopilot update ---
            ap_status = None
            if self.autopilot.enabled:
                gps_crop = None
                gps_info = {}
                if self.nav_win is not None:
                    try:
                        gps_crop = self.nav_win.last_gps_crop
                        gps_info = self.nav_win.last_gps_info
                    except Exception:
                        pass
                ap_dets = self._normalize_dets(objs)
                ap_status = self.autopilot.update(
                    frame_bgr,
                    da_seg_out if da_seg_out is not None else np.zeros((1, 2, MODEL_RES, MODEL_RES), dtype=np.float32),
                    ll_seg_out if ll_seg_out is not None else np.zeros((1, 2, MODEL_RES, MODEL_RES), dtype=np.float32),
                    ap_dets, gps_crop, gps_info
                )

                # Draw AP HUD
                ap_state = ap_status.get("state", "?")
                ap_steer = ap_status.get("steering", 0)
                ap_thr = ap_status.get("throttle", 0)
                ap_tgt = ap_status.get("target_speed", 0)
                hud = f"AP {ap_state} S:{ap_steer:+.2f} T:{ap_thr:+.2f} Vtgt:{ap_tgt:.0f}"
                cv2.putText(result, hud, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

                # Visual lane center guide
                lc = ap_status.get("lane_center")
                if lc is not None:
                    cv2.circle(result, (int(lc), h - 20), 8, (0, 255, 0), 2)
                    cv2.line(result, (w // 2, h - 20), (int(lc), h - 20), (0, 255, 0), 2)

            da_max = float(np.max(da_prob)) if da_prob is not None else 0.0
            ll_max = float(np.max(ll_prob)) if ll_prob is not None else 0.0

            frame_count += 1
            now = time.time()
            if now - prev_time >= 1.0:
                fps_display = frame_count
                frame_count = 0
                prev_time = now

            with self._lock:
                self.frame_result = result
                self.fps = fps_display
                self.road_pixels = int(np.sum(da_mask))
                self.lane_pixels = int(np.sum(ll_mask))
                self.da_max = da_max
                self.ll_max = ll_max
                self.obj_count = len(objs)
                self.backend = "Coral+YOLOP" if self.use_coral else "YOLOP"

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
            backend = getattr(self, 'backend', '?')

        if frame is not None:
            h, w = frame.shape[:2]
            max_w, max_h = 1440, 900
            scale = min(max_w / w, max_h / h, 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                new_w, new_h = w, h

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self._tk_img = imgtk
            self.lbl_main.configure(image=imgtk)
            self.lbl_main.place(x=0, y=0, width=new_w, height=new_h)

            ap_text = ""
            if self.autopilot.enabled and self.autopilot.status:
                s = self.autopilot.status
                ap_text = f" | AP:{s.get('state','?')} S:{s.get('steering',0):+.2f}"
            # Mostrar teclas activas del autopilot
            ap_keys = ""
            if self.autopilot.enabled and self.autopilot.status:
                keys = self.autopilot.status.get("keys", [])
                if keys:
                    ap_keys = " | " + ",".join(keys)

            info_text = f"FPS:{fps} | {backend} | objs:{obj_count}{ap_text}{ap_keys}"
            self.lbl_info.configure(text=info_text)
            self.lbl_info.place(x=0, y=new_h, width=new_w, height=18)
            self.root.geometry(f"{new_w}x{new_h + 18}")
            self.btn_close.place(x=new_w - 22, y=0)

        self._schedule_update()

    def _force_topmost(self):
        """Mantiene la ventana siempre encima de todo (incluso juegos)."""
        if not self._running:
            return
        try:
            self.root.lift()
            self.root.attributes("-topmost", True)
        except Exception:
            pass
        self.root.after(500, self._force_topmost)

    def on_close(self):
        self._running = False
        self.autopilot.shutdown()
        self.root.destroy()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  ETS2 Unified Detector — YOLOP + Coral TPU")
    print("=" * 60)
    print()

    ensure_model()

    print("[INFO] Cargando YOLOP ONNX...")
    ort.set_default_logger_severity(4)
    session = ort.InferenceSession(MODEL_PATH, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    print(f"[INFO] YOLOP listo. Input: {session.get_inputs()[0].shape}")
    print(f"[INFO] Providers: {session.get_providers()}")

    print("[INFO] Cargando Coral TPU (opcional)...")
    coral_interp, coral_labels, coral_detect = load_coral()
    if coral_interp:
        print("[INFO] Coral TPU activo para objetos.")
    else:
        print("[INFO] Coral TPU no disponible. Usando YOLOP para objetos.")

    print("[INFO] Buscando ventana ETS2...")
    win_info = find_ets2_window()
    if not win_info:
        print("[INFO] Seleccion manual...")
        win_info = select_region_manual()
    else:
        print(f"[INFO] Ventana: {win_info['width']}x{win_info['height']}")

    # Ventana raiz (tkinter no es thread-safe, todo en main thread)
    root = tk.Tk()
    root.title("ETS2 Unified Detector")
    root.attributes("-topmost", True)
    try:
        root.overrideredirect(True)
    except tk.TclError:
        root.wm_attributes("-type", "splash")
    root.configure(bg="black", highlightthickness=0)
    root.geometry("640x480+100+100")

    # Crear ventana principal
    main_win = MainWindow(root, win_info, session, coral_interp, coral_labels, coral_detect)

    # Crear ventana secundaria (Toplevel comparte el mismo intérprete Tcl)
    nav_root = tk.Toplevel(root)
    nav_root.title("ETS2 Nav + Mirrors")
    nav_root.attributes("-topmost", True)
    try:
        nav_root.overrideredirect(True)
    except tk.TclError:
        nav_root.wm_attributes("-type", "splash")
    nav_root.configure(bg="black", highlightthickness=0)
    nav_root.geometry("400x300+800+100")

    nav_win = NavOverlayManager(nav_root, win_info)
    main_win.set_nav_overlay(nav_win)

    # Hotkey listener (F9 = toggle autopilot)
    print("[INFO] Press F9 to toggle autopilot")
    hotkey_listener = pynput_kb.Listener(on_press=main_win._on_hotkey)
    hotkey_listener.start()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        hotkey_listener.stop()
        main_win.autopilot.shutdown()

    print("[INFO] Cerrado.")


if __name__ == "__main__":
    main()
