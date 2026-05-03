"""
autopilot.py
Autonomous driving stack for Euro Truck Simulator 2.

Architecture:
  Perception   ->  World Model  ->  Planning  ->  Control
  (YOLOP lanes, Coral objects, GPS minimap)  ->  (Lane keep, Path follow, Obstacle avoid)  ->  (PWM keys)

Control strategy:
  - Steering is a blend of short-term lane-keeping (70%) and long-term GPS heading (30%).
  - Throttle uses a target-speed follower with obstacle feed-forward braking.
  - All outputs are smoothed with rate-limiters to avoid jerky inputs.
"""
import time
import math
import cv2
import numpy as np
from vehicle_control import VehicleController


# ───────────────────────────────
# Lane Analyzer
# ───────────────────────────────
class LaneAnalyzer:
    """Extract lane geometry from YOLOP ll_seg output."""

    def __init__(self, lane_width_ratio: float = 0.45):
        # Expected lane width as ratio of image width (at bottom of crop)
        self.lane_width_ratio = lane_width_ratio
        self.prev_center = None
        self.prev_heading = 0.0

    def analyze(self, ll_seg, img_width: int, img_height: int):
        """
        Args:
            ll_seg: YOLOP lane-line output, shape (1, 2, H, W) float32
            img_width: original capture width (for scale)
            img_height: original capture height
        Returns:
            lane_center_x (pixels in original frame) or None,
            lane_heading  (normalized -1..1, positive = turning right),
            lane_width_px (estimated)
        """
        if ll_seg is None or ll_seg.size == 0:
            return self.prev_center, self.prev_heading, None

        # ll_seg[0,1] = lane-line probability map
        prob = ll_seg[0, 1, :, :]
        h, w = prob.shape
        scale_x = img_width / w
        scale_y = img_height / h

        # Sample rows in bottom half (closer to truck)
        y_ratios = [0.55, 0.65, 0.75, 0.85, 0.92]
        centers = []
        left_xs = []
        right_xs = []

        mid = w // 2
        gap = int(w * 0.06)  # min gap from center to avoid hood noise

        for yr in y_ratios:
            y = int(h * yr)
            if y >= h:
                continue
            row = prob[y, :]
            xs = np.where(row > 0.45)[0]
            if len(xs) < 3:
                continue

            left = xs[xs < mid - gap]
            right = xs[xs > mid + gap]

            if len(left) > 0 and len(right) > 0:
                lx = float(np.median(left))
                rx = float(np.median(right))
                centers.append(((lx + rx) / 2.0) * scale_x)
                left_xs.append(lx * scale_x)
                right_xs.append(rx * scale_x)
            elif len(right) > 0:
                rx = float(np.median(right))
                centers.append((rx - w * self.lane_width_ratio / 2) * scale_x)
                right_xs.append(rx * scale_x)
            elif len(left) > 0:
                lx = float(np.median(left))
                centers.append((lx + w * self.lane_width_ratio / 2) * scale_x)
                left_xs.append(lx * scale_x)

        if not centers:
            return self.prev_center, self.prev_heading, None

        lane_center = float(np.median(centers))
        lane_width = None
        if left_xs and right_xs:
            lane_width = float(np.median(right_xs) - np.median(left_xs))

        # Heading: slope between top and bottom center points
        heading = 0.0
        if len(centers) >= 2:
            # Use top-most and bottom-most samples
            dy = (y_ratios[-1] - y_ratios[0]) * h * scale_y
            dx = centers[-1] - centers[0]
            if abs(dy) > 1:
                # Normalize: dx per 100 px of forward distance
                heading = np.clip(dx / (dy + 1e-6), -1.0, 1.0)

        self.prev_center = lane_center
        self.prev_heading = heading
        return lane_center, heading, lane_width


# ───────────────────────────────
# Obstacle Assessor
# ───────────────────────────────
class ObstacleAssessor:
    """Turn Coral detections into brake/throttle recommendations."""

    DANGEROUS = {"car", "truck", "bus", "person", "motorcycle", "bicycle"}

    def assess(self, detections, img_h: int, img_w: int):
        """
        Returns:
            brake_intensity: 0.0 (free road) to 1.0 (emergency stop)
            closest_dist_ratio: 0 = far, 1 = imminent collision
        """
        if not detections:
            return 0.0, 0.0

        max_brake = 0.0
        closest = 0.0

        for det in detections:
            x1, y1, x2, y2, label, score = det
            if label not in self.DANGEROUS:
                continue
            if score < 0.4:
                continue

            # Distance proxy: bottom edge of bbox (y2). Lower in image = closer.
            dist_ratio = y2 / img_h  # 0 = top/far, 1 = bottom/close
            dist_ratio = max(0.0, min(1.0, dist_ratio))

            # In-lane: horizontal center near image center
            cx = (x1 + x2) / 2.0
            in_lane = abs(cx - img_w / 2.0) < (img_w * 0.32)

            if not in_lane and dist_ratio < 0.75:
                # Adjacent lane, not too close -> ignore
                continue

            # Risk curve: exponential near collision
            risk = dist_ratio ** 2.5
            if label == "person":
                risk = min(1.0, risk * 1.4)

            max_brake = max(max_brake, risk)
            closest = max(closest, dist_ratio)

        return max_brake, closest


# ───────────────────────────────
# GPS Navigator
# ───────────────────────────────
class GPSNavigator:
    """Extract steering bias and speed hint from GPS minimap crop."""

    def __init__(self):
        self.prev_bias = 0.0

    def analyze(self, gps_crop):
        """
        Returns:
            steer_bias: -1..1 (negative = route to left, turn left)
            speed_hint: estimated speed limit from text, or None
        """
        if gps_crop is None or gps_crop.size == 0:
            return self.prev_bias, None

        h, w = gps_crop.shape[:2]
        hsv = cv2.cvtColor(gps_crop, cv2.COLOR_BGR2HSV)

        # Route = red/orange on GPS
        lower1 = np.array([0, 30, 30], dtype=np.uint8)
        upper1 = np.array([35, 255, 255], dtype=np.uint8)
        lower2 = np.array([150, 30, 30], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

        ys, xs = np.where(mask > 0)
        if len(xs) < 15:
            return self.prev_bias, None

        # Route centroid vs player position (roughly center of minimap)
        route_cx = float(np.median(xs))
        player_cx = w * 0.5

        # Normalize: if route is to the right of player, bias positive (turn right)
        bias = (route_cx - player_cx) / (w * 0.45)
        bias = float(np.clip(bias, -1.0, 1.0))

        # Dampen bias so GPS is a gentle nudge, not a sharp command
        bias *= 0.35
        self.prev_bias = bias

        # Speed hint from white text in top of minimap (very crude)
        gray = cv2.cvtColor(gps_crop, cv2.COLOR_BGR2GRAY)
        top_region = gray[: int(h * 0.25), :]
        bright = np.sum(top_region > 180)
        speed_hint = None
        if bright > 200:
            # Heuristic: lots of white text = probably showing speed limit or distance
            # We don't OCR yet, so we leave None and let default target speed rule.
            pass

        return bias, speed_hint


# ───────────────────────────────
# Smooth Rate Limiter
# ───────────────────────────────
class RateLimiter:
    """Limit rate of change to avoid jerky control inputs."""

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


# ───────────────────────────────
# Main Autopilot
# ───────────────────────────────
class Autopilot:
    """
    High-level autonomous driving orchestrator.

    States:
        IDLE      = disabled
        ACTIVE    = normal driving
        BRAKING   = obstacle detected, slowing down
        EMERGENCY = watchdog / lost perception -> full stop
    """

    def __init__(self):
        self.vc = VehicleController(hz=30)

        self.enabled = False
        self.state = "IDLE"

        # Sub-modules
        self.lane_analyzer = LaneAnalyzer()
        self.obstacle_assessor = ObstacleAssessor()
        self.gps_nav = GPSNavigator()

        # Rate limiters (units per control cycle @ 30 Hz)
        self.steer_limiter = RateLimiter(max_delta_per_cycle=0.18)
        self.throttle_limiter = RateLimiter(max_delta_per_cycle=0.08)

        # Parameters
        self.target_speed = 75.0          # km/h default
        self.min_speed = 0.0
        self.max_speed = 90.0
        self.kp_steer = 1.1
        self.kd_steer = 0.35
        self.prev_steer_error = 0.0
        self.kp_speed = 0.015
        self.speed_integral = 0.0

        self._last_update = time.time()
        self._frame_counter = 0
        self._lost_lane_frames = 0
        self._status_info = {}

    # ── Public API ─────────────────

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self.vc.set_controls(0.0, 0.0)
            self.vc._release_all()
            self.state = "IDLE"
            self.steer_limiter.value = 0.0
            self.throttle_limiter.value = 0.0
        else:
            self.state = "ACTIVE"
        return self.enabled

    def update(self, raw_bgr, da_seg, ll_seg, coral_dets, gps_crop, gps_info):
        """
        Main call, invoked once per captured frame (~30 Hz).

        Args:
            raw_bgr: full BGR frame from screen capture
            da_seg:  YOLOP drivable area (1,2,H,W)
            ll_seg:  YOLOP lane lines (1,2,H,W)
            coral_dets: list of (x1,y1,x2,y2,label,score)
            gps_crop: BGR crop of GPS minimap region
            gps_info: dict from nav_overlay (counts etc.)
        """
        if not self.enabled:
            return {"state": "DISABLED"}

        now = time.time()
        dt = now - self._last_update
        self._last_update = now
        self._frame_counter += 1

        h, w = raw_bgr.shape[:2]

        # ── 1. Perception Fusion ──
        lane_center, lane_heading, lane_width = self.lane_analyzer.analyze(
            ll_seg, w, h
        )

        obstacle_brake, closest_obstacle = self.obstacle_assessor.assess(
            coral_dets, h, w
        )

        gps_steer_bias, gps_speed_hint = self.gps_nav.analyze(gps_crop)

        # ── 2. Speed Planning ──
        target = self.target_speed
        if gps_speed_hint:
            target = min(target, gps_speed_hint)

        # Slow for curves (detected by lane heading deviation or GPS bias)
        curve_factor = 1.0
        if abs(lane_heading) > 0.25:
            curve_factor = 0.65
        elif abs(lane_heading) > 0.12:
            curve_factor = 0.82
        elif abs(gps_steer_bias or 0) > 0.25:
            curve_factor = 0.75
        target *= curve_factor

        # Obstacle braking feed-forward
        if obstacle_brake > 0.02:
            target *= (1.0 - min(1.0, obstacle_brake * 1.3))
            self.state = "BRAKING"
        else:
            if self.state == "BRAKING":
                self.state = "ACTIVE"

        target = max(self.min_speed, min(self.max_speed, target))

        # Fake current speed (we don't have telemetry; infer from throttle state)
        # In future, read speedometer OCR from dashboard.
        current_speed = self._estimate_speed(gps_info)

        # ── 3. Lateral Control (Steering) ──
        steer = 0.0
        if lane_center is not None:
            # Normalized error: 0 = centered, +1 = far right, -1 = far left
            error = (lane_center - w / 2.0) / (w / 2.0 + 1e-6)

            # Dead zone: don't fight tiny offsets
            if abs(error) < 0.025:
                error = 0.0

            derivative = (error - self.prev_steer_error) / max(dt, 0.001)
            steer = self.kp_steer * error + self.kd_steer * derivative
            self.prev_steer_error = error
            self._lost_lane_frames = 0
        else:
            self._lost_lane_frames += 1
            # If lanes lost briefly, damp previous steering toward 0
            steer = self.prev_steer_error * 0.5
            if self._lost_lane_frames > 45:  # ~1.5 s
                self.state = "EMERGENCY"
                steer = 0.0

        # Blend with GPS heading (gentle nudge for long-term route)
        if gps_steer_bias is not None:
            # When lanes are strong, GPS is a small correction.
            # When lanes are missing, GPS dominates.
            lane_conf = 1.0 if lane_center is not None else 0.0
            alpha = 0.25 if lane_conf else 0.85
            steer = (1.0 - alpha) * steer + alpha * gps_steer_bias

        # Clamp
        steer = float(np.clip(steer, -1.0, 1.0))

        # ── 4. Longitudinal Control (Throttle) ──
        speed_error = target - current_speed
        throttle = self.kp_speed * speed_error

        # If emergency or obstacle very close, override to brake
        if self.state == "EMERGENCY" or obstacle_brake > 0.85:
            throttle = -1.0
        elif closest_obstacle > 0.7 and speed_error > 0:
            # Don't accelerate into a close obstacle even if under target speed
            throttle = min(throttle, 0.0)

        throttle = float(np.clip(throttle, -1.0, 1.0))

        # ── 5. Smooth & Apply ──
        smooth_steer = self.steer_limiter.update(steer)
        smooth_throttle = self.throttle_limiter.update(throttle)

        self.vc.set_controls(smooth_steer, smooth_throttle)

        self._status_info = {
            "state": self.state,
            "steering": round(smooth_steer, 2),
            "throttle": round(smooth_throttle, 2),
            "target_speed": round(target, 1),
            "current_speed": round(current_speed, 1),
            "lane_center": round(lane_center, 1) if lane_center is not None else None,
            "lane_heading": round(lane_heading, 2),
            "obstacle_brake": round(obstacle_brake, 2),
            "gps_bias": round(gps_steer_bias, 2) if gps_steer_bias is not None else None,
        }
        return self._status_info

    def emergency_stop(self):
        """External call for immediate full stop."""
        self.state = "EMERGENCY"
        self.vc.emergency_stop()

    def shutdown(self):
        self.enabled = False
        self.vc.stop()

    def _estimate_speed(self, gps_info):
        """
        Placeholder speed estimator.
        In the future, OCR the speedometer digits from the dashboard crop.
        For now we assume the truck roughly tracks target speed with lag.
        """
        # Very crude: if we see GPS text, maybe we can guess
        # Otherwise return a conservative estimate
        if gps_info and gps_info.get("text", 0) > 0:
            # Some text visible; can't read it yet without OCR
            pass
        # Return default target as proxy (control loop will handle error)
        return self.target_speed * 0.6

    @property
    def status(self):
        return self._status_info.copy()
