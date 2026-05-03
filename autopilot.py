"""
autopilot.py
Autonomous driving stack for Euro Truck Simulator 2.
"""
import time
import collections
import cv2
import numpy as np
from vehicle_control import VehicleController


# ───────────────────────────────
# Lane Analyzer
# ───────────────────────────────
class LaneAnalyzer:
    """Extract lane geometry from YOLOP ll_seg output."""

    def __init__(self, lane_width_ratio: float = 0.32):
        self.lane_width_ratio = lane_width_ratio
        self.prev_center = None
        self.prev_heading = 0.0
        self._width_history = collections.deque(maxlen=20)
        self._ema_center = None
        self._ema_alpha = 0.60  # fast response, some smoothing

    def _estimated_lane_width(self, img_width: int):
        if self._width_history:
            return float(np.median(self._width_history))
        return img_width * 0.26

    def analyze(self, ll_seg, img_width: int, img_height: int):
        if ll_seg is None or ll_seg.size == 0:
            return self.prev_center, self.prev_heading, None

        prob = ll_seg[0, 1, :, :]
        h, w = prob.shape
        scale_x = img_width / w
        scale_y = img_height / h

        y_ratios = [0.55, 0.65, 0.75, 0.85, 0.92]
        centers = []
        left_xs = []
        right_xs = []
        widths = []

        mid = w // 2
        gap = int(w * 0.05)

        for yr in y_ratios:
            y = int(h * yr)
            if y >= h:
                continue
            row = prob[y, :]
            xs = np.where(row > 0.40)[0]
            if len(xs) < 3:
                continue

            left = xs[xs < mid - gap]
            right = xs[xs > mid + gap]

            if len(left) > 0 and len(right) > 0:
                lx = float(np.median(left))
                rx = float(np.median(right))
                cw = (rx - lx) * scale_x
                centers.append(((lx + rx) / 2.0) * scale_x)
                left_xs.append(lx * scale_x)
                right_xs.append(rx * scale_x)
                widths.append(cw)
            elif len(right) > 0:
                rx = float(np.median(right))
                ew = self._estimated_lane_width(img_width)
                centers.append((rx * scale_x) - ew / 2)
                right_xs.append(rx * scale_x)
            elif len(left) > 0:
                lx = float(np.median(left))
                ew = self._estimated_lane_width(img_width)
                centers.append((lx * scale_x) + ew / 2)
                left_xs.append(lx * scale_x)

        if not centers:
            return self.prev_center, self.prev_heading, None

        raw_center = float(np.median(centers))

        if widths:
            self._width_history.append(float(np.median(widths)))

        lane_width = self._estimated_lane_width(img_width)

        # EMA smoothing: fast reaction, no lag from history window
        if self._ema_center is None:
            self._ema_center = raw_center
        else:
            self._ema_center = self._ema_alpha * raw_center + (1 - self._ema_alpha) * self._ema_center
        lane_center = float(self._ema_center)

        heading = 0.0
        if len(centers) >= 2:
            dy = (y_ratios[-1] - y_ratios[0]) * h * scale_y
            dx = centers[-1] - centers[0]
            if abs(dy) > 1:
                heading = np.clip(dx / (dy + 1e-6), -1.0, 1.0)

        self.prev_center = lane_center
        self.prev_heading = heading
        return lane_center, heading, lane_width


# ───────────────────────────────
# Obstacle Assessor
# ───────────────────────────────
class ObstacleAssessor:
    DANGEROUS = {"car", "truck", "bus", "person", "motorcycle", "bicycle"}

    def assess(self, detections, img_h: int, img_w: int):
        if not detections:
            return 0.0, 0.0

        max_brake = 0.0
        closest = 0.0
        img_area = img_h * img_w

        for det in detections:
            x1, y1, x2, y2, label, score = det
            if label not in self.DANGEROUS:
                continue
            if score < 0.35:
                continue

            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / (img_area + 1e-6)

            # Filter 1: own vehicle / trailer (huge box)
            if area_ratio > 0.20:
                continue

            # Filter 2: object starts very low = hood/dashboard reflection
            if y1 > img_h * 0.72 and area_ratio > 0.05:
                continue

            # Filter 3: object touches bottom of frame = almost certainly own vehicle
            # In ETS2 camera view, nothing real should be this close except the hood
            if y2 > img_h * 0.88:
                continue

            dist_ratio = y2 / img_h
            dist_ratio = max(0.0, min(1.0, dist_ratio))
            cx = (x1 + x2) / 2.0
            in_lane = abs(cx - img_w / 2.0) < (img_w * 0.34)

            if not in_lane and dist_ratio < 0.70:
                continue

            # Risk curve: only brake for objects in lower 55% of frame
            risk = 0.0
            if dist_ratio > 0.55:
                risk = ((dist_ratio - 0.55) / 0.45) ** 2.0
            if label == "person":
                risk = min(1.0, risk * 1.5)

            max_brake = max(max_brake, risk)
            closest = max(closest, dist_ratio)

        return max_brake, closest


# ───────────────────────────────
# GPS Navigator
# ───────────────────────────────
class GPSNavigator:
    def __init__(self):
        self.prev_bias = 0.0

    def analyze(self, gps_crop):
        if gps_crop is None or gps_crop.size == 0:
            return self.prev_bias, None

        h, w = gps_crop.shape[:2]
        hsv = cv2.cvtColor(gps_crop, cv2.COLOR_BGR2HSV)

        lower1 = np.array([0, 30, 30], dtype=np.uint8)
        upper1 = np.array([35, 255, 255], dtype=np.uint8)
        lower2 = np.array([150, 30, 30], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

        ys, xs = np.where(mask > 0)
        if len(xs) < 15:
            return self.prev_bias, None

        route_cx = float(np.median(xs))
        player_cx = w * 0.5
        bias = (route_cx - player_cx) / (w * 0.45)
        bias = float(np.clip(bias, -1.0, 1.0)) * 0.25
        self.prev_bias = bias
        return bias, None


# ───────────────────────────────
# Smooth Rate Limiter
# ───────────────────────────────
class RateLimiter:
    def __init__(self, max_delta_per_cycle: float):
        self.max_delta = max_delta_per_cycle
        self.value = 0.0

    def update(self, target: float) -> float:
        delta = target - self.value
        if delta > self.max_delta:
            delta = self.max_delta
        elif delta < -self.max_delta:
            delta = -self.max_delta
        self.value += delta
        return self.value

    def reset(self, val: float = 0.0):
        self.value = val


# ───────────────────────────────
# Main Autopilot
# ───────────────────────────────
class Autopilot:
    """
    States: IDLE, ACTIVE, BRAKING, EMERGENCY, RECOVER
    """

    def __init__(self):
        self.vc = VehicleController(hz=30)
        self.enabled = False
        self.state = "IDLE"

        self.lane_analyzer = LaneAnalyzer()
        self.obstacle_assessor = ObstacleAssessor()
        self.gps_nav = GPSNavigator()

        # Rate limiters
        self.steer_limiter = RateLimiter(max_delta_per_cycle=0.12)
        self.throttle_limiter = RateLimiter(max_delta_per_cycle=0.06)

        # PID lateral
        self.kp_steer = 1.20
        self.ki_steer = 0.015
        self.kd_steer = 0.40
        self._steer_integral = 0.0
        self.prev_steer_error = 0.0

        # Speed control
        self.kp_speed = 0.008
        self.target_speed = 45.0   # slower = more time to correct
        self.min_speed = 0.0
        self.max_speed = 65.0

        self._last_update = time.time()
        self._frame_counter = 0
        self._lost_lane_frames = 0
        self._status_info = {}

        # Recovery
        self._collision_frames = 0
        self._recover_timer = 0
        self._last_lane_center = None
        self._stuck_timer = 0.0
        self._log_every = 15

        # Simple speed estimator
        self._estimated_speed_val = 0.0

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            print("[AP] DISABLED")
            self.vc.set_controls(0.0, 0.0)
            self.vc._release_all()
            self.state = "IDLE"
            self.steer_limiter.reset(0.0)
            self.throttle_limiter.reset(0.0)
            self._steer_integral = 0.0
            self.prev_steer_error = 0.0
            self._collision_frames = 0
            self._recover_timer = 0
            self._estimated_speed_val = 0.0
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

        # ── Perception ──
        lane_center, lane_heading, lane_width = self.lane_analyzer.analyze(ll_seg, w, h)
        obstacle_brake, closest_obstacle = self.obstacle_assessor.assess(coral_dets, h, w)
        gps_steer_bias, _ = self.gps_nav.analyze(gps_crop)

        # ── Crash / Stuck detection ──
        if obstacle_brake > 0.95 and closest_obstacle > 0.95:
            self._collision_frames += 1
        else:
            self._collision_frames = max(0, self._collision_frames - 2)

        if lane_center is not None:
            if self._last_lane_center is not None:
                if abs(lane_center - self._last_lane_center) < 15 and closest_obstacle > 0.85:
                    self._stuck_timer += dt
                else:
                    self._stuck_timer = max(0.0, self._stuck_timer - dt * 2)
            self._last_lane_center = lane_center
        else:
            self._stuck_timer += dt

        # State machine
        if self.state == "RECOVER":
            self._recover_timer -= 1
            if self._recover_timer <= 0:
                print("[AP] RECOVER done, resuming ACTIVE")
                self.state = "ACTIVE"
                self.throttle_limiter.reset(0.0)
                self._stuck_timer = 0.0
                self._collision_frames = 0
        elif self._collision_frames > 20 or self._stuck_timer > 2.5:
            print(f"[AP] CRASH/STUCK -> RECOVER")
            self.state = "RECOVER"
            self._recover_timer = 90
            self.throttle_limiter.reset(0.0)
            self.steer_limiter.reset(0.0)
            self._steer_integral = 0.0
        elif self.state == "EMERGENCY":
            pass
        elif obstacle_brake > 0.02:
            self.state = "BRAKING"
        else:
            if self.state == "BRAKING":
                self.state = "ACTIVE"

        # ── Speed Planning ──
        target = self.target_speed

        curve_factor = 1.0
        if abs(lane_heading) > 0.35:
            curve_factor = 0.45
        elif abs(lane_heading) > 0.18:
            curve_factor = 0.70
        elif abs(gps_steer_bias or 0) > 0.30:
            curve_factor = 0.65
        target *= curve_factor

        if obstacle_brake > 0.02 and self.state not in ("RECOVER", "EMERGENCY"):
            target *= (1.0 - min(1.0, obstacle_brake * 1.2))

        target = max(self.min_speed, min(self.max_speed, target))
        current_speed = self._estimate_speed(gps_info)

        # ── Lateral Control (PID) ──
        steer = 0.0
        if lane_center is not None:
            error = (lane_center - w / 2.0) / (w / 2.0 + 1e-6)

            # Tighter dead zone: respond to smaller offsets
            if abs(error) < 0.015:
                error = 0.0

            # Integral with anti-windup
            self._steer_integral += error * dt
            self._steer_integral = np.clip(self._steer_integral, -0.30, 0.30)

            derivative = (error - self.prev_steer_error) / max(dt, 0.001)
            steer = (self.kp_steer * error +
                     self.ki_steer * self._steer_integral +
                     self.kd_steer * derivative)
            self.prev_steer_error = error
            self._lost_lane_frames = 0
        else:
            self._lost_lane_frames += 1
            steer = self.prev_steer_error * 0.3
            if self._lost_lane_frames > 45:
                self.state = "EMERGENCY"
                steer = 0.0

        # Blend GPS
        if gps_steer_bias is not None:
            lane_conf = 1.0 if lane_center is not None else 0.0
            alpha = 0.20 if lane_conf else 0.75
            steer = (1.0 - alpha) * steer + alpha * gps_steer_bias

        steer = float(np.clip(steer, -1.0, 1.0))

        # ── Longitudinal Control ──
        # CRITICAL: in ETS2, DOWN key when stopped puts the truck in REVERSE.
        # We NEVER want reverse in normal driving. Throttle floor is 0 (coast).
        speed_error = target - current_speed
        throttle = self.kp_speed * speed_error

        if self.state == "RECOVER":
            # Only reverse in recovery mode after crash
            throttle = -1.0
            steer = 0.0
        elif self.state == "EMERGENCY":
            # Emergency = full brake but don't reverse
            throttle = 0.0
        elif obstacle_brake > 0.90 or closest_obstacle > 0.96:
            # Hard stop = release throttle (engine braking stops the truck)
            throttle = 0.0
        elif closest_obstacle > 0.65 and speed_error > 0:
            # Don't accelerate into obstacle
            throttle = min(throttle, 0.0)

        # Clamp: NEVER reverse in normal driving. 0 = coast/engine brake.
        throttle = float(np.clip(throttle, 0.0, 1.0))

        # ── Smooth & Apply ──
        smooth_steer = self.steer_limiter.update(steer)

        if self.state in ("RECOVER", "EMERGENCY") or obstacle_brake > 0.85:
            smooth_throttle = throttle
            self.throttle_limiter.reset(throttle)
        else:
            smooth_throttle = self.throttle_limiter.update(throttle)

        self.vc.set_controls(smooth_steer, smooth_throttle)

        # Logging
        if self._frame_counter % self._log_every == 0:
            lc = f"{lane_center:.0f}" if lane_center is not None else "--"
            print(f"[AP] {self.state:8s} S={smooth_steer:+.2f} T={smooth_throttle:+.2f} "
                  f"Vtgt={target:.0f} lane={lc} obs={obstacle_brake:.2f} "
                  f"keys={self.vc.active_keys}")

        self._status_info = {
            "state": self.state,
            "steering": round(smooth_steer, 2),
            "throttle": round(smooth_throttle, 2),
            "steer_raw": round(steer, 2),
            "throttle_raw": round(throttle, 2),
            "target_speed": round(target, 1),
            "current_speed": round(current_speed, 1),
            "lane_center": round(lane_center, 1) if lane_center is not None else None,
            "lane_heading": round(lane_heading, 2),
            "lane_width": round(lane_width, 1) if lane_width is not None else None,
            "obstacle_brake": round(obstacle_brake, 2),
            "closest_obstacle": round(closest_obstacle, 2),
            "gps_bias": round(gps_steer_bias, 2) if gps_steer_bias is not None else None,
            "keys": self.vc.active_keys,
            "notes": "",
        }
        return self._status_info

    def emergency_stop(self):
        print("[AP] EMERGENCY STOP")
        self.state = "EMERGENCY"
        self.vc.emergency_stop()

    def shutdown(self):
        print("[AP] Shutdown")
        self.enabled = False
        self.vc.stop()

    def _estimate_speed(self, gps_info):
        """
        Simple physics-based speed estimator.
        Tracks speed from throttle history since we don't have telemetry.
        """
        accel_rate = 1.2   # km/h gained per frame at full throttle
        drag = 0.25        # km/h lost per frame when coasting
        brake_drag = 0.8   # km/h lost per frame when braking

        thr = self.vc.throttle
        if thr > 0.05:
            self._estimated_speed_val += thr * accel_rate
        elif thr < -0.05:
            self._estimated_speed_val -= brake_drag
        else:
            self._estimated_speed_val -= drag

        self._estimated_speed_val = max(0.0, min(self.max_speed, self._estimated_speed_val))
        return self._estimated_speed_val

    @property
    def status(self):
        return self._status_info.copy()
