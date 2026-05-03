"""
autopilot.py
Autonomous driving for ETS2 using ChosunTruck lane detection approach.

Pipeline:
  1. Capture frame
  2. ChosunLaneDetector: IPM + Sobel + horizontal scan -> steering
  3. Constant throttle, reduce for curves/obstacles
  4. Send to VehicleController
"""
import time
import numpy as np
from vehicle_control import VehicleController
from chosun_lane import ChosunLaneDetector


class ObstacleAssessor:
    DANGEROUS = {"car", "truck", "bus", "person", "motorcycle", "bicycle"}

    def assess(self, detections, img_h: int, img_w: int):
        if not detections:
            return 0.0

        max_risk = 0.0
        img_area = img_h * img_w

        for det in detections:
            x1, y1, x2, y2, label, score = det
            if label not in self.DANGEROUS:
                continue
            if label in ("car", "truck", "bus") and score < 0.45:
                continue
            if label == "person" and score < 0.55:
                continue
            if score < 0.35:
                continue

            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / (img_area + 1e-6)
            if area_ratio > 0.20:
                continue
            if y2 > img_h * 0.88:
                continue

            dist_ratio = y2 / img_h
            in_lane = abs((x1 + x2) / 2.0 - img_w / 2.0) < (img_w * 0.34)
            if not in_lane and dist_ratio < 0.70:
                continue

            risk = 0.0
            if dist_ratio > 0.55:
                risk = ((dist_ratio - 0.55) / 0.45) ** 2.0
            max_risk = max(max_risk, risk)

        return max_risk


class GPSNavigator:
    def __init__(self):
        self.prev_bias = 0.0

    def analyze(self, gps_crop):
        if gps_crop is None or gps_crop.size == 0:
            return self.prev_bias

        import cv2
        h, w = gps_crop.shape[:2]
        hsv = cv2.cvtColor(gps_crop, cv2.COLOR_BGR2HSV)
        mask = (cv2.inRange(hsv, np.array([0, 30, 30], dtype=np.uint8),
                            np.array([35, 255, 255], dtype=np.uint8)) |
                cv2.inRange(hsv, np.array([150, 30, 30], dtype=np.uint8),
                            np.array([180, 255, 255], dtype=np.uint8)))

        ys, xs = np.where(mask > 0)
        if len(xs) < 15:
            return self.prev_bias

        route_cx = float(np.median(xs))
        bias = (route_cx - w * 0.5) / (w * 0.45)
        bias = float(np.clip(bias, -1.0, 1.0)) * 0.25
        self.prev_bias = bias
        return bias


class Autopilot:
    def __init__(self):
        self.vc = VehicleController(hz=30)
        self.enabled = False
        self.state = "IDLE"

        self.lane_detector = None  # initialized on first frame
        self.obstacle_assessor = ObstacleAssessor()
        self.gps_nav = GPSNavigator()

        self._last_update = time.time()
        self._frame_counter = 0
        self._lost_lane_frames = 0
        self._status_info = {}
        self._log_every = 15

        self._recover_timer = 0

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            print("[AP] DISABLED")
            self.vc.set_controls(0.0, 0.0)
            self.vc._release_all()
            self.state = "IDLE"
            self._recover_timer = 0
        else:
            print("[AP] ENABLED")
            self.state = "ACTIVE"
        return self.enabled

    def update(self, raw_bgr, da_seg, ll_seg, coral_dets, gps_crop, gps_info):
        if not self.enabled:
            return {"state": "DISABLED"}

        now = time.time()
        dt = now - self._last_update
        self._last_update = now
        self._frame_counter += 1

        h, w = raw_bgr.shape[:2]

        # Initialize lane detector on first frame
        if self.lane_detector is None or self.lane_detector.w != w or self.lane_detector.h != h:
            self.lane_detector = ChosunLaneDetector(img_width=w, img_height=h)

        # ── Lane Detection (Chosun approach) ──
        steer, debug_img, lane_info = self.lane_detector.detect(raw_bgr)

        # ── Obstacles ──
        obstacle_risk = self.obstacle_assessor.assess(coral_dets, h, w)

        # ── GPS fallback ──
        gps_bias = self.gps_nav.analyze(gps_crop)

        # ── State machine ──
        if lane_info.get("status") != "ok":
            self._lost_lane_frames += 1
        else:
            self._lost_lane_frames = 0

        if self.state == "RECOVER":
            self._recover_timer -= 1
            if self._recover_timer <= 0:
                print("[AP] RECOVER done")
                self.state = "ACTIVE"
        elif self._lost_lane_frames > 60:
            self.state = "RECOVER"
            self._recover_timer = 90
            print("[AP] LOST LANES -> RECOVER")
        elif obstacle_risk > 0.02:
            self.state = "BRAKING"
        else:
            if self.state == "BRAKING":
                self.state = "ACTIVE"

        # ── Steering ──
        if self._lost_lane_frames > 15 and gps_bias is not None:
            # Use GPS when lanes lost for >0.5s
            steer = gps_bias

        # Clamp and boost small corrections
        steer = float(np.clip(steer, -1.0, 1.0))

        # ── Throttle ──
        if self.state == "RECOVER":
            throttle = -1.0
            steer = 0.0
        elif obstacle_risk > 0.85:
            throttle = 0.0
        elif obstacle_risk > 0.40:
            throttle = 0.40
        else:
            # Slow down in curves
            if abs(steer) > 0.35:
                throttle = 0.65
            elif abs(steer) > 0.15:
                throttle = 0.85
            else:
                throttle = 1.0

        throttle = float(np.clip(throttle, 0.0, 1.0))

        # ── Apply ──
        self.vc.set_controls(steer, throttle)

        # Logging
        if self._frame_counter % self._log_every == 0:
            status = lane_info.get("status", "?")
            offset = lane_info.get("offset", 0)
            rows = lane_info.get("rows_found", 0)
            print(f"[AP] {self.state:8s} S={steer:+.2f} T={throttle:.2f} "
                  f"lane={status} off={offset} rows={rows} obs={obstacle_risk:.2f} "
                  f"keys={self.vc.active_keys}")

        self._status_info = {
            "state": self.state,
            "steering": round(steer, 2),
            "throttle": round(throttle, 2),
            "lane_status": lane_info.get("status"),
            "lane_offset": lane_info.get("offset"),
            "lane_rows": lane_info.get("rows_found"),
            "obstacle_risk": round(obstacle_risk, 2),
            "gps_bias": round(gps_bias, 2) if gps_bias is not None else None,
            "keys": self.vc.active_keys,
            "debug_img": debug_img,
        }
        return self._status_info

    def emergency_stop(self):
        self.state = "EMERGENCY"
        self.vc.emergency_stop()

    def shutdown(self):
        self.enabled = False
        self.vc.stop()

    @property
    def status(self):
        return self._status_info.copy()
