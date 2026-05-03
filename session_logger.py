"""
session_logger.py
Guarda frames y log CSV de cada sesión de conducción para debug off-line.

Estructura:
  captures/2026-05-03_14-30-00/
    raw_0000.png      -> frame original del juego
    det_0000.png      -> frame con overlays (carretera, objetos, HUD)
    gps_0000.png      -> crop del GPS procesado
    log.csv           -> timestamp,steer,throttle,lane,obs,state,keys,...

Mantiene buffer circular de últimos 120 frames (~40 segundos a 3 fps)
para no llenar el disco.
"""
import os
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2


class SessionLogger:
    """Log frames + autopilot decisions for post-hoc analysis."""

    def __init__(self, max_frames: int = 120, save_every: int = 10):
        """
        Args:
            max_frames:  mantener solo los N más recientes (borra viejos)
            save_every:  guardar 1 de cada N frames del loop principal (~30Hz)
        """
        self.enabled = True
        self.max_frames = max_frames
        self.save_every = save_every
        self._frame_counter = 0
        self._saved_counter = 0
        self._saved_ids = []

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.base_dir = Path("captures") / stamp
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.base_dir / "log.csv"
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "id", "timestamp", "state", "steer_raw", "steer_smooth",
            "throttle_raw", "throttle_smooth", "target_speed", "est_speed",
            "lane_center", "lane_heading", "lane_width", "obs_brake",
            "obs_closest", "gps_bias", "keys", "notes"
        ])
        self._csv_file.flush()

        print(f"[LOGGER] Session folder: {self.base_dir}")

    def save(self, raw_bgr, det_bgr, gps_bgr, ap_status: dict):
        """
        Call once per processed frame. Only saves every Nth frame.
        """
        if not self.enabled:
            return

        self._frame_counter += 1
        if self._frame_counter % self.save_every != 0:
            return

        idx = self._saved_counter
        self._saved_counter += 1
        self._saved_ids.append(idx)

        ts = time.time()

        # Save images
        if raw_bgr is not None and raw_bgr.size > 0:
            cv2.imwrite(str(self.base_dir / f"raw_{idx:04d}.png"), raw_bgr)
        if det_bgr is not None and det_bgr.size > 0:
            cv2.imwrite(str(self.base_dir / f"det_{idx:04d}.png"), det_bgr)
        if gps_bgr is not None and gps_bgr.size > 0:
            cv2.imwrite(str(self.base_dir / f"gps_{idx:04d}.png"), gps_bgr)

        # Write CSV row
        self._csv_writer.writerow([
            idx,
            f"{ts:.3f}",
            ap_status.get("state", ""),
            ap_status.get("steer_raw", ""),
            ap_status.get("steering", ""),
            ap_status.get("throttle_raw", ""),
            ap_status.get("throttle", ""),
            ap_status.get("target_speed", ""),
            ap_status.get("current_speed", ""),
            ap_status.get("lane_center", ""),
            ap_status.get("lane_heading", ""),
            ap_status.get("lane_width", ""),
            ap_status.get("obstacle_brake", ""),
            ap_status.get("closest_obstacle", ""),
            ap_status.get("gps_bias", ""),
            ";".join(ap_status.get("keys", [])),
            ap_status.get("notes", ""),
        ])
        self._csv_file.flush()

        # Circular cleanup: delete oldest if over limit
        while len(self._saved_ids) > self.max_frames:
            old = self._saved_ids.pop(0)
            for prefix in ("raw", "det", "gps"):
                p = self.base_dir / f"{prefix}_{old:04d}.png"
                if p.exists():
                    p.unlink()

    def close(self):
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
        print(f"[LOGGER] Session saved to: {self.base_dir}")
