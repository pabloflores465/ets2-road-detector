"""
autopilot.py
Autonomous driving stack for Euro Truck Simulator 2.

KEY INSIGHT from user screenshots:
  - YOLOP lane lines are barely visible in ETS2 cabin view (dashboard blocks bottom)
  - Raycasting from bottom hits the dashboard, not the road
  - The reference implementation (stefanos50) used EXTERIOR camera view

NEW STRATEGY (contour-based, no raycasting):
  1. Take the ll_mask (binary lane line mask from YOLOP)
  2. Find contours in the ROAD REGION only (ignore dashboard at bottom 25%)
  3. Split into left/right groups by image center
  4. Compute median X of each group -> lane boundaries
  5. Steer to keep the CENTER between boundaries aligned with image center

Fallback when no lanes: use GPS minimap heading.
"""
import time
import cv2
import numpy as np
from vehicle_control import VehicleController


# ───────────────────────────────
# Lane Analyzer (contour-based)
# ───────────────────────────────
class LaneAnalyzer:
    """
    Uses YOLOP ll_mask binary mask directly.
    Finds lane line blobs, groups into left/right, computes center.
    Ignores bottom 25% of image (dashboard/hood area).
    """

    def __init__(self):
        self._ema_center = None
        self._ema_alpha = 0.50

    def analyze(self, ll_mask, img_width: int, img_height: int):
        """
        Args:
            ll_mask: binary mask (H, W) uint8 from YOLOP postprocess
                     1 = lane line pixel, 0 = background
        Returns:
            steer_cmd: -1..1
            left_x: median X of left lane line pixels (or None)
            right_x: median X of right lane line pixels (or None)
        """
        if ll_mask is None or ll_mask.size == 0:
            return 0.0, None, None

        h, w = ll_mask.shape[:2]

        # Ignore bottom 25% (dashboard, hood, GPS panel)
        # Keep only middle section: y from 30% to 75% of height
        y_start = int(h * 0.30)
        y_end = int(h * 0.75)
        road_region = ll_mask[y_start:y_end, :]

        if np.sum(road_region) < 20:
            # Almost no lane pixels in road region
            return 0.0, None, None

        # Find connected components (blobs)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            road_region.astype(np.uint8), connectivity=8
        )

        left_xs = []
        right_xs = []
        mid_x = w // 2

        for i in range(1, num_labels):  # skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 15:  # ignore tiny noise
                continue

            cx = int(centroids[i][0])
            cy = int(centroids[i][1]) + y_start  # offset back to full image coords

            # Classify as left or right line
            if cx < mid_x - w * 0.08:
                left_xs.append(cx)
            elif cx > mid_x + w * 0.08:
                right_xs.append(cx)
            # Ignore blobs near center (could be hood reflections)

        left_x = int(np.median(left_xs)) if left_xs else None
        right_x = int(np.median(right_xs)) if right_xs else None

        # ── Steering ──
        steer = 0.0

        if left_x is not None and right_x is not None:
            lane_center = (left_x + right_x) / 2.0
            image_center = w / 2.0
            error = (lane_center - image_center) / (w / 2.0 + 1e-6)

            # EMA smooth the error to reduce jitter
            if self._ema_center is None:
                self._ema_center = error
            else:
                self._ema_center = self._ema_alpha * error + (1 - self._ema_alpha) * self._ema_center

            steer = float(np.clip(self._ema_center * 2.5, -1.0, 1.0))

        elif left_x is not None and right_x is None:
            # Only left line: steer right if too close
            dist_from_center = (mid_x - left_x) / (w / 2.0)
            if dist_from_center < 0.25:
                steer = +0.60
            else:
                steer = +0.20

        elif right_x is not None and left_x is None:
            dist_from_center = (right_x - mid_x) / (w / 2.0)
            if dist_from_center < 0.25:
                steer = -0.60
            else:
                steer = -0.20

        else:
            steer = 0.0

        return steer, left_x, right_x


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
        bias = float(np.clip(bias, -1.0, 1.0)) * 0.30
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

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            print("[AP] DISABLED")
            self.vc.set_controls(0.0, 0.0)
            self.vc._release_all()
            self.state = "IDLE"
            self._collision_frames = 0
            self._recover_timer = 0
        else:
            print("[AP] ENABLED")
            self.state = "ACTIVE"
        return self.enabled

    def update(self, raw_bgr, da_seg, ll_mask, coral_dets, gps_crop, gps_info):
        if not self.enabled:
            return {"state": "DISABLED"}

        now = time.time()
        dt = now - self._last_update
        self._last_update = now
        self._frame_counter += 1

        h, w = raw_bgr.shape[:2]

        # ── Perception ──
        steer_from_lanes, left_x, right_x = self.lane_analyzer.analyze(ll_mask, w, h)
        obstacle_risk = self.obstacle_assessor.assess(coral_dets, h, w)
        gps_bias = self.gps_nav.analyze(gps_crop)

        # ── Crash / Stuck detection ──
        if obstacle_risk > 0.95:
            self._collision_frames += 1
        else:
            self._collision_frames = max(0, self._collision_frames - 2)

        if left_x is None and right_x is None:
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
            self._recover_timer = 90
        elif self.state == "EMERGENCY":
            pass
        elif obstacle_risk > 0.02:
            self.state = "BRAKING"
        else:
            if self.state == "BRAKING":
                self.state = "ACTIVE"

        # ── Steering ──
        steer = steer_from_lanes

        # Fallback to GPS when no lanes detected
        if gps_bias is not None:
            if left_x is None or right_x is None:
                alpha = 0.70  # GPS dominates when lanes lost
            else:
                alpha = 0.20
            steer = (1.0 - alpha) * steer + alpha * gps_bias

        steer = float(np.clip(steer, -1.0, 1.0))

        # ── Throttle ──
        if self.state == "RECOVER":
            throttle = -1.0
            steer = 0.0
        elif self.state == "EMERGENCY":
            throttle = 0.0
        elif obstacle_risk > 0.85:
            throttle = 0.0
        elif obstacle_risk > 0.40:
            throttle = 0.35
        else:
            if abs(steer) > 0.30:
                throttle = 0.65
            elif abs(steer) > 0.15:
                throttle = 0.80
            else:
                throttle = 1.0

        throttle = float(np.clip(throttle, 0.0, 1.0))

        self.vc.set_controls(steer, throttle)

        # Logging
        if self._frame_counter % self._log_every == 0:
            lx = f"{left_x}" if left_x else "--"
            rx = f"{right_x}" if right_x else "--"
            print(f"[AP] {self.state:8s} S={steer:+.2f} T={throttle:.2f} "
                  f"L={lx} R={rx} obs={obstacle_risk:.2f} gps={gps_bias:+.2f} "
                  f"keys={self.vc.active_keys}")

        self._status_info = {
            "state": self.state,
            "steering": round(steer, 2),
            "throttle": round(throttle, 2),
            "left_x": left_x,
            "right_x": right_x,
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
