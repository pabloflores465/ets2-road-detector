"""
chosun_lane.py
Lane detection using ChosunTruck approach:
  1. Perspective Transform (bird's eye view) of road region
  2. Sobel edge detection + threshold
  3. Horizontal scan from center to find left/right lane lines
  4. Average centerline across rows -> steering

This ignores the dashboard completely by only transforming the road region.
"""
import cv2
import numpy as np


class ChosunLaneDetector:
    """
    ChosunTruck-style lane detector.
    Uses IPM + horizontal scanning for robust lane finding.
    """

    def __init__(self, img_width: int = 738, img_height: int = 480):
        self.w = img_width
        self.h = img_height

        # IPM source points (trapezoid on road in original image)
        # These need tuning for ETS2 cabin view
        # Bottom points: wide, near the car
        # Top points: narrow, far away
        self.ipm_src = np.float32([
            [self.w * 0.15, self.h * 0.75],   # bottom left
            [self.w * 0.85, self.h * 0.75],   # bottom right
            [self.w * 0.55, self.h * 0.42],   # top right
            [self.w * 0.45, self.h * 0.42],   # top left
        ])

        # IPM destination points (rectangle in bird's eye view)
        dst_w = 320
        dst_h = 240
        self.ipm_dst = np.float32([
            [0, dst_h],           # bottom left
            [dst_w, dst_h],       # bottom right
            [dst_w, 0],           # top right
            [0, 0],               # top left
        ])

        self.dst_size = (dst_w, dst_h)
        self.M = cv2.getPerspectiveTransform(self.ipm_src, self.ipm_dst)
        self.M_inv = cv2.getPerspectiveTransform(self.ipm_dst, self.ipm_src)

        # Scanning parameters
        self.center_x = dst_w // 2
        self.scan_top = 30     # start scanning from this row (from top)
        self.scan_bottom = dst_h - 10
        self.max_scan_dist = 150  # max pixels to scan left/right

    def detect(self, frame_bgr):
        """
        Args:
            frame_bgr: full captured frame
        Returns:
            steer: -1.0 to 1.0 (negative = left, positive = right)
            debug_img: visualization image
            info: dict with detection details
        """
        h, w = frame_bgr.shape[:2]

        # 1. Perspective Transform (bird's eye view)
        ipm = cv2.warpPerspective(frame_bgr, self.M, self.dst_size)

        # 2. Convert to grayscale + blur + Sobel X
        gray = cv2.cvtColor(ipm, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Sobel on X direction (vertical edges become horizontal in IPM)
        sobel = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel = np.absolute(sobel)
        sobel = np.uint8(np.clip(sobel, 0, 255))

        # 3. Threshold to get binary edges
        _, edges = cv2.threshold(sobel, 30, 255, cv2.THRESH_BINARY)

        # 4. Horizontal scan: for each row, scan from center to find lane lines
        centerlines = []
        left_dists = []
        right_dists = []

        for y in range(self.scan_bottom - 1, self.scan_top, -1):
            left_dist = -1
            right_dist = -1

            # Scan right from center
            for dx in range(0, self.max_scan_dist):
                x = self.center_x + dx
                if x >= self.dst_size[0]:
                    break
                if edges[y, x] > 0:
                    right_dist = dx
                    break

            # Scan left from center
            for dx in range(0, self.max_scan_dist):
                x = self.center_x - dx
                if x < 0:
                    break
                if edges[y, x] > 0:
                    left_dist = dx
                    break

            # If both lines found, compute centerline
            if left_dist >= 0 and right_dist >= 0:
                centerline = self.center_x + (right_dist - left_dist) // 2
                centerlines.append(centerline)
                left_dists.append(left_dist)
                right_dists.append(right_dist)

        # 5. Compute steering from centerlines
        if len(centerlines) < 5:
            # Not enough data
            debug = self._make_debug(frame_bgr, ipm, edges, [], None)
            return 0.0, debug, {"status": "no_lanes", "count": len(centerlines)}

        avg_centerline = int(np.mean(centerlines))
        offset = avg_centerline - self.center_x

        # Normalize to -1..1 (max offset ~100 px is full steering)
        steer = float(np.clip(offset / 60.0, -1.0, 1.0))

        # Boost small corrections
        if 0.02 < abs(steer) < 0.12:
            steer = np.sign(steer) * 0.15

        # 6. Compute road curvature from first vs last centerline
        first_center = centerlines[0]
        last_center = centerlines[-1]
        curve = last_center - first_center  # positive = turning right

        info = {
            "status": "ok",
            "steer": round(steer, 3),
            "offset": offset,
            "avg_center": avg_centerline,
            "center_x": self.center_x,
            "curve": curve,
            "rows_found": len(centerlines),
        }

        debug = self._make_debug(frame_bgr, ipm, edges, centerlines, avg_centerline)
        return steer, debug, info

    def _make_debug(self, orig, ipm, edges, centerlines, avg_center):
        """Create a debug visualization image."""
        h, w = orig.shape[:2]

        # Resize IPM and edges for display
        ipm_disp = cv2.resize(ipm, (w // 2, h // 2))
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_disp = cv2.resize(edges_color, (w // 2, h // 2))

        # Draw centerlines on edges image
        edges_viz = edges_color.copy()
        for i, cx in enumerate(centerlines):
            y = self.scan_bottom - 1 - i
            if 0 <= y < edges_viz.shape[0] and 0 <= cx < edges_viz.shape[1]:
                cv2.circle(edges_viz, (cx, y), 2, (0, 255, 0), -1)

        if avg_center is not None:
            cv2.line(edges_viz, (avg_center, 0), (avg_center, edges_viz.shape[0]), (0, 0, 255), 2)
            cv2.line(edges_viz, (self.center_x, 0), (self.center_x, edges_viz.shape[0]), (255, 0, 0), 1)

        edges_disp = cv2.resize(edges_viz, (w // 2, h // 2))

        # Draw IPM trapezoid on original
        orig_viz = orig.copy()
        pts = self.ipm_src.astype(np.int32)
        cv2.polylines(orig_viz, [pts], True, (0, 255, 255), 2)
        for pt in pts:
            cv2.circle(orig_viz, tuple(pt), 5, (0, 0, 255), -1)

        # Stack: original (top), ipm + edges (bottom)
        top = orig_viz
        bottom = np.hstack([ipm_disp, edges_disp])
        # Pad bottom to match top width
        if bottom.shape[1] < top.shape[1]:
            pad = np.zeros((bottom.shape[0], top.shape[1] - bottom.shape[1], 3), dtype=np.uint8)
            bottom = np.hstack([bottom, pad])

        debug = np.vstack([top, bottom])
        return debug

    def draw_ipm_zone(self, frame_bgr):
        """Draw the IPM zone on the original frame for calibration."""
        viz = frame_bgr.copy()
        pts = self.ipm_src.astype(np.int32)
        cv2.polylines(viz, [pts], True, (0, 255, 255), 2)
        for i, pt in enumerate(pts):
            cv2.circle(viz, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(viz, str(i), tuple(pt + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        return viz
