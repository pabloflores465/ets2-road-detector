"""
autopilot.py
Autonomous driving using ETSAuto bevlanedet model.

Pipeline:
  1. ETSAuto bevlanedet.onnx -> BEV lane detection
  2. Pure Pursuit controller -> steering angle
  3. Simple throttle logic
  4. pynput keyboard output (macOS)
"""
import time
import numpy as np
from vehicle_control import VehicleController
from etsauto_adapter import ETSAutoAdapter


class ObstacleAssessor:
    DANGEROUS = {"car", "truck", "bus", "person", "motorcycle", "bicycle"}

    def assess(self, detections, img_h, img_w):
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


class Autopilot:
    def __init__(self):
        self.vc = VehicleController(hz=30)
        self.enabled = False
        self.state = "IDLE"

        # ETSAuto lane detection + control
        self.etsauto = None  # lazy init on first frame
        self.obstacle_assessor = ObstacleAssessor()

        self._last_update = time.time()
        self._frame_counter = 0
        self._status_info = {}
        self._log_every = 15
        self._recover_timer = 0
        self._lost_frames = 0

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            print("[AP] DISABLED")
            self.vc.set_controls(0.0, 0.0)
            self.state = "IDLE"
            self._recover_timer = 0
        else:
            print("[AP] ENABLED")
            self.state = "ACTIVE"
            if self.etsauto is None:
                try:
                    self.etsauto = ETSAutoAdapter()
                    print("[AP] ETSAuto model loaded")
                except Exception as e:
                    print(f"[AP] ERROR loading ETSAuto: {e}")
                    self.enabled = False
        return self.enabled

    def update(self, raw_bgr, da_seg, ll_seg, coral_dets, gps_crop, gps_info):
        if not self.enabled:
            return {"state": "DISABLED"}

        now = time.time()
        self._last_update = now
        self._frame_counter += 1

        h, w = raw_bgr.shape[:2]

        # ── Lane Detection + Steering (ETSAuto) ──
        steer = 0.0
        throttle = 0.0
        lane_info = {"status": "no_model"}

        if self.etsauto is not None:
            try:
                steer, throttle, lane_info = self.etsauto.process(raw_bgr, speed_kmh=50.0)
            except Exception as e:
                print(f"[AP] ETSAuto error: {e}")
                lane_info = {"status": "error", "msg": str(e)}

        # ── Obstacles ──
        obstacle_risk = self.obstacle_assessor.assess(coral_dets, h, w)

        # ── State machine ──
        if self.state == "RECOVER":
            self._recover_timer -= 1
            if self._recover_timer <= 0:
                print("[AP] RECOVER done")
                self.state = "ACTIVE"
        elif lane_info.get("status") != "ok":
            self._lost_frames += 1
            if self._lost_frames > 60:
                self.state = "RECOVER"
                self._recover_timer = 90
                print("[AP] LOST LANES -> RECOVER")
        else:
            self._lost_frames = 0
            if obstacle_risk > 0.02:
                self.state = "BRAKING"
            else:
                if self.state == "BRAKING":
                    self.state = "ACTIVE"

        # ── Override for obstacles ──
        if self.state == "RECOVER":
            throttle = 0.0
            steer = 0.0
        elif obstacle_risk > 0.85:
            throttle = 0.0
        elif obstacle_risk > 0.40:
            throttle = min(throttle, 0.40)

        throttle = float(np.clip(throttle, 0.0, 1.0))
        steer = float(np.clip(steer, -1.0, 1.0))

        # ── Apply ──
        self.vc.set_controls(steer, throttle)

        # Logging
        if self._frame_counter % self._log_every == 0:
            status = lane_info.get("status", "?")
            print(f"[AP] {self.state:8s} S={steer:+.2f} T={throttle:.2f} "
                  f"lane={status} obs={obstacle_risk:.2f} keys={self.vc.active_keys}")

        self._status_info = {
            "state": self.state,
            "steering": round(steer, 2),
            "throttle": round(throttle, 2),
            "lane_status": lane_info.get("status"),
            "obstacle_risk": round(obstacle_risk, 2),
            "keys": self.vc.active_keys,
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
