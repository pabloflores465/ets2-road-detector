"""
gps_navigator.py
Extracts steering commands from ETS2 GPS minimap.

Strategy:
  1. Crop GPS region from frame
  2. Detect red route line (HSV red/orange ranges)
  3. Fit a polyline to the route pixels
  4. Compute heading angle at truck position (bottom-center of GPS)
  5. Return steer bias proportional to heading deviation

This is the PRIMARY navigation source. Lane detection is only used
to fine-tune centering when both lanes are clearly visible.
"""
import cv2
import numpy as np


class GPSNavigator:
    """
    Extracts steering direction from GPS minimap route line.
    """

    def __init__(self):
        self.prev_heading = 0.0
        self._ema_heading = 0.0
        self._ema_alpha = 0.40

    def analyze(self, gps_crop):
        """
        Args:
            gps_crop: BGR image of GPS minimap region
        Returns:
            steer_bias: -1..1 (negative = turn left, positive = turn right)
            heading_deg: estimated route heading in degrees (0 = straight up)
            debug_info: dict
        """
        if gps_crop is None or gps_crop.size == 0:
            return self.prev_heading, 0.0, {"status": "no_crop"}

        h, w = gps_crop.shape[:2]
        hsv = cv2.cvtColor(gps_crop, cv2.COLOR_BGR2HSV)

        # Detect red/orange route line
        # ETS2 GPS route: red/orange/brownish
        lower1 = np.array([0, 40, 40], dtype=np.uint8)
        upper1 = np.array([25, 255, 255], dtype=np.uint8)
        lower2 = np.array([150, 40, 40], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        mask_red = cv2.inRange(hsv, lower1, upper1)
        mask_red2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask_red, mask_red2)

        # Also detect yellow/amber (some ETS2 versions use yellow route)
        lower_yellow = np.array([15, 60, 80], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bitwise_or(mask, mask_yellow)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Find route pixels
        ys, xs = np.where(mask > 0)
        if len(xs) < 20:
            return self.prev_heading, 0.0, {"status": "no_route", "pixels": len(xs)}

        # Player position in GPS: roughly bottom center
        player_y = h * 0.85
        player_x = w * 0.5

        # Filter route pixels: must be below player (ahead on road)
        # and not too far to the sides
        ahead_mask = ys > player_y - h * 0.3
        xs_ahead = xs[ahead_mask]
        ys_ahead = ys[ahead_mask]

        if len(xs_ahead) < 10:
            # Try all pixels if not enough ahead
            xs_ahead = xs
            ys_ahead = ys

        # Fit line to route pixels using polyfit
        # Use points near player for immediate heading
        near_player = ys_ahead > player_y - h * 0.25
        if np.sum(near_player) >= 5:
            xs_near = xs_ahead[near_player]
            ys_near = ys_ahead[near_player]
        else:
            xs_near = xs_ahead
            ys_near = ys_ahead

        # Compute heading: angle of best-fit line through route points
        # heading = 0 means straight up (negative Y direction in image)
        if len(xs_near) >= 3:
            # Fit y = mx + b (but we want x as function of y for heading)
            # Use linear regression: x = a*y + b
            # heading angle: atan2(dx, dy) where dy is negative (up in image)
            ys_norm = (ys_near - np.mean(ys_near)) / (np.std(ys_near) + 1e-6)
            xs_norm = (xs_near - np.mean(xs_near)) / (np.std(xs_near) + 1e-6)
            
            # Simple slope: dx/dy
            dy = ys_near[-1] - ys_near[0]  # should be negative (going up)
            dx = xs_near[-1] - xs_near[0]
            
            if abs(dy) > 1:
                heading_rad = np.arctan2(dx, -dy)  # negative dy because up is forward
                heading_deg = np.degrees(heading_rad)
            else:
                heading_deg = 0.0
        else:
            heading_deg = 0.0

        # EMA smooth heading
        self._ema_heading = self._ema_alpha * heading_deg + (1 - self._ema_alpha) * self._ema_heading
        smoothed_heading = self._ema_heading

        # Convert heading to steer bias
        # heading > 0 means route goes to the right -> steer right
        # heading < 0 means route goes to the left -> steer left
        max_heading = 45.0  # 45 degrees = full steering
        steer_bias = float(np.clip(smoothed_heading / max_heading, -1.0, 1.0))

        self.prev_heading = steer_bias

        debug = {
            "status": "ok",
            "heading_deg": round(heading_deg, 1),
            "smoothed_heading": round(smoothed_heading, 1),
            "steer_bias": round(steer_bias, 2),
            "pixels": len(xs),
            "pixels_ahead": len(xs_ahead),
        }

        return steer_bias, heading_deg, debug

    def draw_debug(self, gps_crop, heading_deg=0):
        """Draw route detection debug overlay."""
        if gps_crop is None:
            return None

        viz = gps_crop.copy()
        h, w = viz.shape[:2]

        # Draw player position
        player_x = int(w * 0.5)
        player_y = int(h * 0.85)
        cv2.circle(viz, (player_x, player_y), 5, (255, 0, 0), -1)

        # Draw heading arrow
        arrow_len = 40
        angle_rad = np.radians(-heading_deg)  # negative because image Y is down
        dx = int(arrow_len * np.sin(angle_rad))
        dy = -int(arrow_len * np.cos(angle_rad))
        cv2.arrowedLine(viz, (player_x, player_y), (player_x + dx, player_y + dy), (0, 255, 0), 2)

        return viz
