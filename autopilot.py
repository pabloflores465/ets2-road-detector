"""
autopilot.py
Autonomous driving stack for Euro Truck Simulator 2.

Strategy based on stefanos50/YOLOP-Lane-Keeping-Assist:
  - Two vertical raycasts from hood-level upward to find lane line hits
  - Compare hit heights: if left line is closer (lower hit), steer right
  - Constant throttle (always accelerate), only coast/brake for obstacles
  - No speed PID — ETS2 keyboard steering barely works at very low speeds
"""
import time
import cv2
import numpy as np
from vehicle_control import VehicleController


# ───────────────────────────────
# Lane Analyzer (raycast based)
# ───────────────────────────────
class LaneAnalyzer:
    """
    Instead of computing a global lane center, we cast two vertical rays
    from near the bottom of the image upward into the ll_seg probability map.
    The ray that hits its lane line lower (closer to the car) means the car
    is closer to that side -> steer away.
    """

    def __init__(self, ray_x_offset: float = 0.06):
        # Horizontal offset from image center for each ray (as ratio of width)
        # IMPORTANT: must be small enough that rays cross the RED lane lines,
        # not the green drivable area. At 640x480, lane lines are ~60-80px
        # from center. Offset 0.06 = ~38px at 640w.
        self.ray_x_offset = ray_x_offset
        self.prev_left_y = None
        self.prev_right_y = None
        self._steer_alpha = 0.65  # fast EMA for steering response

    def analyze(self, ll_seg, img_width: int, img_height: int):
        """
        Returns:
            steer_cmd: -1..1 (negative = steer left, positive = steer right)
            left_y:  pixel Y where left ray hit lane line (or None)
            right_y: pixel Y where right ray hit lane line (or None)
        """
        if ll_seg is None or ll_seg.size == 0:
            return 0.0, None, None

        prob = ll_seg[0, 1, :, :]  # (H, W) float32 probabilities
        h, w = prob.shape

        # Ray origin: near bottom of image
        start_y = int(h * 0.92)
        end_y = int(h * 0.30)  # look up to 30% from top

        # Ray X positions in model space
        mid_x = w // 2
        left_x = int(mid_x - w * self.ray_x_offset)
        right_x = int(mid_x + w * self.ray_x_offset)
        left_x = max(2, min(w - 3, left_x))
        right_x = max(2, min(w - 3, right_x))

        # Cast upward from start_y to end_y
        # Use 5-pixel wide rays for robustness + low threshold
        left_hit = self._raycast_up_wide(prob, left_x, start_y, end_y)
        right_hit = self._raycast_up_wide(prob, right_x, start_y, end_y)

        # Convert to original image coords
        scale_y = img_height / h
        left_y = left_hit * scale_y if left_hit is not None else None
        right_y = right_hit * scale_y if right_hit is not None else None

        if left_y is not None:
            self.prev_left_y = left_y
        if right_y is not None:
            self.prev_right_y = right_y

        # ── Steering decision ──
        steer = 0.0

        if left_y is not None and right_y is not None:
            # Both lines visible.
            # If left_y > right_y, left line is closer to car (lower in image)
            # -> car is closer to left line -> steer RIGHT
            diff = left_y - right_y
            # Very strong gain: 40 px diff -> full correction
            raw_steer = np.clip(diff / 40.0, -1.0, 1.0)
            # Boost small corrections so the truck doesn't drift
            if abs(raw_steer) < 0.15:
                raw_steer = np.sign(raw_steer) * 0.20
            steer = raw_steer

        elif left_y is not None and right_y is None:
            # Only left line visible.
            if left_y > img_height * 0.75:
                steer = +0.80  # very close to left edge, steer right hard
            elif left_y > img_height * 0.60:
                steer = +0.50
            else:
                steer = -0.40  # only left visible far away, drift left

        elif right_y is not None and left_y is None:
            if right_y > img_height * 0.75:
                steer = -0.80
            elif right_y > img_height * 0.60:
                steer = -0.50
            else:
                steer = +0.40

        else:
            # No lines at all.
            steer = 0.0

        return steer, left_y, right_y

    def _raycast_up_wide(self, prob, x, start_y, end_y):
        """Return first Y (from bottom going up) where prob > threshold.
        Samples a 5-pixel wide column with very low threshold."""
        for y in range(start_y, end_y - 1, -1):
            # Sample 5 pixels horizontally for robustness
            vals = [
                prob[y, x - 2], prob[y, x - 1], prob[y, x],
                prob[y, x + 1], prob[y, x + 2]
            ]
            if max(vals) > 0.12:  # very low threshold to catch thin lane lines
                return y
        return None


# ───────────────────────────────
# Obstacle Assessor
# ───────────────────────────────
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
            # Stricter thresholds to avoid false positives
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
            if y1 > img_h * 0.72 and area_ratio > 0.05:
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
            if label == "person":
                risk = min(1.0, risk * 1.5)

            max_risk = max(max_risk, risk)

        return max_risk


# ───────────────────────────────
# GPS Navigator
# ───────────────────────────────
class GPSNavigator:
    def __init__(self):
        self.prev_bias = 0.0

    def analyze(self, gps_crop):
        if gps_crop is None or gps_crop.size == 0:
            return self.prev_bias

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


# ───────────────────────────────
# Main Autopilot
# ───────────────────────────────
class Autopilot:
    def __init__(self):
        self.vc = VehicleController(hz=30)
        self.enabled = False
        self.state = "IDLE"

        self.lane_analyzer = LaneAnalyzer()
        self.obstacle_assessor = ObstacleAssessor()
        self.gps_nav = GPSNavigator()

        self._last_update = time.time()
        self._frame_counter = 0
        self._lost_lane_frames = 0
        self._status_info = {}

        # Recovery
        self._collision_frames = 0
        self._recover_timer = 0
        self._stuck_timer = 0.0
        self._log_every = 15

        # Smoothing
        self._steer_ema = 0.0
        self._steer_alpha = 0.60

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            print("[AP] DISABLED")
            self.vc.set_controls(0.0, 0.0)
            self.vc._release_all()
            self.state = "IDLE"
            self._steer_ema = 0.0
            self._collision_frames = 0
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

        # ── Perception ──
        steer_from_lanes, left_y, right_y = self.lane_analyzer.analyze(ll_seg, w, h)
        obstacle_risk = self.obstacle_assessor.assess(coral_dets, h, w)
        gps_bias = self.gps_nav.analyze(gps_crop)

        # ── Crash / Stuck detection ──
        if obstacle_risk > 0.95:
            self._collision_frames += 1
        else:
            self._collision_frames = max(0, self._collision_frames - 2)

        # Stuck: no lane info + not moving for a while
        if left_y is None and right_y is None:
            self._lost_lane_frames += 1
        else:
            self._lost_lane_frames = 0

        if self._lost_lane_frames > 60:
            self._stuck_timer += dt
        else:
            self._stuck_timer = max(0.0, self._stuck_timer - dt)

        # State machine
        if self.state == "RECOVER":
            self._recover_timer -= 1
            if self._recover_timer <= 0:
                print("[AP] RECOVER done")
                self.state = "ACTIVE"
                self._stuck_timer = 0.0
                self._collision_frames = 0
        elif self._collision_frames > 25 or self._stuck_timer > 3.0:
            print(f"[AP] CRASH/STUCK -> RECOVER")
            self.state = "RECOVER"
            self._recover_timer = 90  # ~3s
            self._steer_ema = 0.0
        elif self.state == "EMERGENCY":
            pass
        elif obstacle_risk > 0.02:
            self.state = "BRAKING"
        else:
            if self.state == "BRAKING":
                self.state = "ACTIVE"

        # ── Steering ──
        steer = steer_from_lanes

        # Blend with GPS when lanes are weak
        if gps_bias is not None:
            if left_y is None or right_y is None:
                alpha = 0.60  # GPS dominates when lanes lost
            else:
                alpha = 0.20  # GPS is gentle nudge
            steer = (1.0 - alpha) * steer + alpha * gps_bias

        steer = float(np.clip(steer, -1.0, 1.0))

        # EMA smooth steering to avoid jerky inputs
        self._steer_ema = self._steer_alpha * steer + (1.0 - self._steer_alpha) * self._steer_ema
        smooth_steer = float(np.clip(self._steer_ema, -1.0, 1.0))

        # ── Throttle ──
        # ETS2 keyboard steering barely works at very low speeds.
        # We use constant throttle and only reduce for obstacles or curves.
        if self.state == "RECOVER":
            throttle = -1.0
            smooth_steer = 0.0
        elif self.state == "EMERGENCY":
            throttle = 0.0
        elif obstacle_risk > 0.85:
            throttle = 0.0  # coast / engine brake
        elif obstacle_risk > 0.40:
            throttle = 0.35  # slow down
        else:
            # Full throttle on straights, reduce slightly in curves
            if abs(smooth_steer) > 0.30:
                throttle = 0.65
            elif abs(smooth_steer) > 0.15:
                throttle = 0.80
            else:
                throttle = 1.0

        throttle = float(np.clip(throttle, 0.0, 1.0))

        self.vc.set_controls(smooth_steer, throttle)

        # Logging
        if self._frame_counter % self._log_every == 0:
            ly = f"{left_y:.0f}" if left_y else "--"
            ry = f"{right_y:.0f}" if right_y else "--"
            print(f"[AP] {self.state:8s} S={smooth_steer:+.2f} T={throttle:.2f} "
                  f"L={ly} R={ry} obs={obstacle_risk:.2f} gps={gps_bias:+.2f} "
                  f"keys={self.vc.active_keys}")

        self._status_info = {
            "state": self.state,
            "steering": round(smooth_steer, 2),
            "throttle": round(throttle, 2),
            "left_y": round(left_y, 1) if left_y else None,
            "right_y": round(right_y, 1) if right_y else None,
            "obstacle_risk": round(obstacle_risk, 2),
            "gps_bias": round(gps_bias, 2) if gps_bias is not None else None,
            "keys": self.vc.active_keys,
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

    @property
    def status(self):
        return self._status_info.copy()
