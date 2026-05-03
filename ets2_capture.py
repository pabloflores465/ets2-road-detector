#!/usr/bin/env python3
"""
Funciones compartidas de captura de ventana ETS2 via Quartz (macOS).
No depende de ONNX ni de TFLite — puede usarse desde cualquier detector.
"""

import subprocess

import cv2
import numpy as np

# Nombres de ventana a buscar
WINDOW_NAMES = ["Euro Truck Simulator 2", "Euro Truck", "eurotrucks2", "ETS2"]


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
                    "bounds": b,
                }
    return None


def capture_window_quartz(win_id, bounds=None):
    """Capture window and auto-crop macOS window chrome if detected."""
    try:
        import Quartz
    except ImportError:
        return None
    try:
        image = Quartz.CGWindowListCreateImage(
            Quartz.CGRectNull,
            Quartz.kCGWindowListOptionIncludingWindow,
            win_id,
            Quartz.kCGWindowImageDefault
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
        frame = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        # Auto-crop macOS window chrome (title bar ~22px, borders ~1px)
        h, w = frame.shape[:2]
        if bounds:
            # Use Quartz bounds to compute content region
            bh = int(bounds.get("Height", h))
            bw = int(bounds.get("Width", w))
            # If captured image is taller than bounds, we have title bar
            if h > bh + 5:
                offset_y = (h - bh) // 2
                frame = frame[offset_y:offset_y+bh, :, :]
            # If captured image is wider than bounds, we have side borders
            if w > bw + 5:
                offset_x = (w - bw) // 2
                frame = frame[:, offset_x:offset_x+bw, :]
        else:
            # Heuristic: detect title bar by checking if top rows are uniform
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            top_std = np.std(gray[:30, :], axis=1)
            if np.mean(top_std[:10]) < 8:  # likely title bar
                # Find where content starts (std increases)
                for y in range(10, min(40, h)):
                    if top_std[y] > 15:
                        frame = frame[y:, :, :]
                        break

        return frame
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
