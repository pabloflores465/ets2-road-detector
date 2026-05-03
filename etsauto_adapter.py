"""
etsauto_adapter.py
Adapter for ETSAuto bevlanedet.onnx model on macOS.

Replaces YOLOP lane detection with ETSAuto's BEV lane detector.
Uses pynput (keyboard) instead of vJoy for macOS compatibility.
"""
import os
import sys
import math
import cv2
import numpy as np
from scipy.spatial import distance


# ── Post-processing functions from ETSAuto ──

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def collect_nd_embedding_with_position(seg, emb, conf):
    ret = []
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            if seg[i, j] >= conf:
                ret.append((i, j, emb[:, i, j]))
    return ret


def naive_cluster_nd(emb_list, gap):
    centers = []
    cids = []
    for x, y, emb in emb_list:
        min_gap = gap + 1
        min_cid = -1
        for cid, (center, num) in enumerate(centers):
            diff = distance.euclidean(emb, center)
            if diff < min_gap:
                min_gap = diff
                min_cid = cid
        if min_gap < gap:
            cids.append((x, y, min_cid))
            center, num = centers[min_cid]
            centers[min_cid] = ((center * num + emb) / (num + 1), num + 1)
        else:
            centers.append((emb, 1))
            cids.append((x, y, len(centers) - 1))
    return cids, centers


def embedding_post(pred, conf, emb_margin=6.0, min_cluster_size=100):
    seg, emb = pred
    seg, emb = seg[0][0], emb[0]
    ret = collect_nd_embedding_with_position(seg, emb, conf)
    c = naive_cluster_nd(ret, emb_margin)
    lanes = np.zeros(seg.shape, dtype=np.uint8)
    for x, y, cid in c[0]:
        if c[1][cid][1] < min_cluster_size:
            continue
        lanes[x][y] = cid + 1
    return lanes, c[0]


def mean_col_by_row(seg, offset_y):
    center_ids = np.unique(seg[seg > 0])
    lines = []
    for cid in center_ids:
        cols, rows = [], []
        for y_op in range(seg.shape[0]):
            condition = seg[y_op, :] == cid
            x_op = np.where(condition)[0]
            offset_op = offset_y[y_op, :]
            if x_op.size > 0:
                offset_op = offset_op[x_op]
                x_op_with_offset = x_op + offset_op
                x_op = np.mean(x_op_with_offset)
                cols.append(x_op)
                rows.append(y_op + 0.5)
        lines.append((cols, rows))
    return lines


def bev_instance2points(ids, max_x=80, meter_per_pixel=(0.5, 0.5), offset_y=None):
    center = ids.shape[1] / 2
    lines = mean_col_by_row(ids, offset_y)
    points = []
    for y, x in lines:
        x = np.array(x)[::-1]
        y = np.array(y)[::-1]
        x = max_x / meter_per_pixel[0] - x
        y = y * meter_per_pixel[1]
        y -= center * meter_per_pixel[1]
        x = x * meter_per_pixel[0]
        if len(x) < 2:
            continue
        points.append(np.concatenate((x.reshape(1, -1), y.reshape(1, -1)), axis=0).T)
    return points


def horizontal_rounding(line):
    ptx_start = math.ceil(min(line[:, 0]))
    ptx_end = math.floor(max(line[:, 0]))
    if ptx_end > 80:
        ptx_end = 80
    pts_x = np.linspace(ptx_start, ptx_end, (ptx_end - ptx_start) * 2 + 1)
    fit = np.polyfit(line[:, 0], line[:, 1], 3)
    pts_y = fit[0] * pts_x ** 3 + fit[1] * pts_x ** 2 + fit[2] * pts_x + fit[3]
    return np.concatenate((pts_x.reshape((-1, 1)), pts_y.reshape((-1, 1))), axis=1)


def get_skeleton(line_l, line_r):
    if line_l is None or line_r is None:
        return None
    line_m = (line_l + line_r) / 2
    line_m = np.concatenate((line_m, np.array([[-1, 0]])), axis=0)
    pts_x = np.linspace(0, 60, 60 * 2 + 1)
    fit = np.polyfit(line_m[:, 0], line_m[:, 1], 3)
    pts_y = fit[0] * pts_x ** 3 + fit[1] * pts_x ** 2 + fit[2] * pts_x + fit[3]
    return np.concatenate((pts_x.reshape((-1, 1)), pts_y.reshape((-1, 1))), axis=1)


# ── ETSAuto Lane Detector ──

class ETSAutoLaneDetector:
    def __init__(self, onnx_path="etsauto_models/bevlanedet.onnx"):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        # Get output names by index (safer than assuming order)
        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        print(f"[ETSAuto] Model outputs: {self.output_names}")

        self.input_shape = (320, 480)
        self.x_range = (0, 80)
        self.y_range = (-5, 5)
        self.meter_per_pixel = 0.5

        self.post_conf = -0.7
        self.post_emb_margin = 6.0
        self.post_min_cluster_size = 15
        self.lane_width = 3.6

    def preprocess(self, img):
        """img: BGR frame from ETS2"""
        h, w = img.shape[:2]
        # ETSAuto does img[50:640, :, :] on ~640px tall images.
        # They remove top 50px (sky) and KEEP the rest including dashboard.
        # Proportional: remove top 8%, keep everything else.
        crop_top = int(h * 50 / 640)  # proportional to ETSAuto's 50px crop
        img = img[crop_top:, :, :]  # keep bottom (dashboard included!)
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        # Model trained with Albumentations on images loaded by OpenCV (BGR)
        # Albumentations Normalize applies per-channel WITHOUT swapping.
        # So we keep BGR and normalize directly.
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return np.expand_dims(img.transpose(2, 0, 1), axis=0)

    def infer(self, img_bgr):
        input_tensor = self.preprocess(img_bgr)
        # Run inference and map outputs by name
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        output_dict = dict(zip(self.output_names, outputs))

        # Map to expected names (ETSAuto model may have different order)
        seg = output_dict.get('seg', outputs[0])
        embedding = output_dict.get('embedding', outputs[1])
        offset_y = output_dict.get('offset_y', outputs[2])
        z_pred = output_dict.get('z_pred', outputs[3])

        offset_y = sigmoid(offset_y)

        # Cluster embeddings to get lane instances
        canvas, _ = embedding_post(
            (seg, embedding),
            self.post_conf,
            emb_margin=self.post_emb_margin,
            min_cluster_size=self.post_min_cluster_size
        )

        # Convert to BEV points (meters)
        lines = bev_instance2points(
            canvas,
            max_x=self.x_range[1],
            meter_per_pixel=(self.meter_per_pixel, self.meter_per_pixel),
            offset_y=offset_y[0][0]
        )

        # Complete and classify lines
        return self._postprocess_lines(lines)

    def _interpolate_to_grid(self, line, grid_x):
        """Interpolate a line to a common x grid."""
        if line is None or len(line) < 2:
            return None
        # line: Nx2 where [:,0]=x, [:,1]=y
        # Sort by x
        line = line[np.argsort(line[:, 0])]
        x, y = line[:, 0], line[:, 1]
        # Remove duplicates
        mask = np.concatenate(([True], np.diff(x) > 0.001))
        x, y = x[mask], y[mask]
        if len(x) < 2:
            return None
        y_interp = np.interp(grid_x, x, y, left=y[0], right=y[-1])
        return np.column_stack((grid_x, y_interp))

    def _postprocess_lines(self, lines):
        # Round lines and get skeleton
        lines_temp = []
        for line in lines:
            if len(line) <= 30:
                continue
            try:
                lines_temp.append(horizontal_rounding(line))
            except Exception:
                continue

        # Classify by position (y coordinate at x=0)
        line_l = line_r = line_ll = line_rr = None
        for line in lines_temp:
            y_at_origin = line[0, 1]
            if -self.lane_width * 2 <= y_at_origin < -self.lane_width:
                line_ll = line
            elif -self.lane_width <= y_at_origin < 0:
                line_l = line
            elif 0 <= y_at_origin < self.lane_width:
                line_r = line
            elif self.lane_width <= y_at_origin <= self.lane_width * 2:
                line_rr = line

        # Complete missing lines
        grid_x = np.arange(0, 60.5, 0.5)
        if line_l is None and line_r is not None:
            line_r_interp = self._interpolate_to_grid(line_r, grid_x)
            if line_r_interp is not None:
                line_l = line_r_interp - np.array([0, self.lane_width])
        elif line_l is not None and line_r is None:
            line_l_interp = self._interpolate_to_grid(line_l, grid_x)
            if line_l_interp is not None:
                line_r = line_l_interp + np.array([0, self.lane_width])

        # Interpolate both to common grid before averaging
        line_l_interp = self._interpolate_to_grid(line_l, grid_x)
        line_r_interp = self._interpolate_to_grid(line_r, grid_x)

        if line_l_interp is not None and line_r_interp is not None:
            line_m = (line_l_interp + line_r_interp) / 2.0
            # Smooth with polynomial fit (no forced origin point)
            try:
                pts_x = np.linspace(0, 60, 121)
                fit = np.polyfit(line_m[:, 0], line_m[:, 1], 3)
                pts_y = fit[0] * pts_x**3 + fit[1] * pts_x**2 + fit[2] * pts_x + fit[3]
                line_m = np.column_stack((pts_x, pts_y))
            except Exception:
                line_m = None
        else:
            line_m = None

        return {
            "line_l": line_l_interp,
            "line_r": line_r_interp,
            "line_m": line_m,
            "line_ll": line_ll,
            "line_rr": line_rr,
        }


# ── Keyboard Lane Controller ──

class KeyboardLaneController:
    """
    Controller optimized for digital keyboard input.
    Instead of computing a steering angle (for analog joystick),
    computes a lateral offset error and returns a direct [-1, 1] command.
    """

    def __init__(self):
        self.lookahead_near = 10.0   # meters
        self.lookahead_far = 25.0    # meters
        self.wheelbase = 8.0

    def run(self, line_m, speed_ms):
        """
        Args:
            line_m: Nx2 array of (x, y) lane center points in meters
            speed_ms: current speed in m/s
        Returns:
            steer command in [-1, 1]
        """
        if line_m is None or len(line_m) < 20:
            return 0.0

        # Find points at lookahead distances
        near_idx = np.argmin(np.abs(line_m[:, 0] - self.lookahead_near))
        far_idx = np.argmin(np.abs(line_m[:, 0] - self.lookahead_far))

        # Lateral offset (positive = lane center is to the RIGHT of vehicle)
        near_offset = line_m[near_idx, 1]
        far_offset = line_m[far_idx, 1]

        # Heading error: angle of lane center relative to vehicle heading
        # Approximate as slope between near and far points
        dx = line_m[far_idx, 0] - line_m[near_idx, 0]
        dy = line_m[far_idx, 1] - line_m[near_idx, 1]
        heading_error = math.atan2(dy, dx) if dx > 0.5 else 0.0

        # Keyboard-optimized control:
        # For digital keys we need significant commands.
        # Use heading (road direction) as primary signal + small lateral correction.
        # A 0.05 rad heading (~3°) -> 0.4 steer -> 80ms key press
        heading_near = math.atan2(
            line_m[min(15, len(line_m)-1), 1] - line_m[0, 1],
            line_m[min(15, len(line_m)-1), 0] - line_m[0, 0] + 1e-6
        )
        heading_far = math.atan2(
            line_m[min(35, len(line_m)-1), 1] - line_m[min(15, len(line_m)-1), 1],
            line_m[min(35, len(line_m)-1), 0] - line_m[min(15, len(line_m)-1), 0] + 1e-6
        )

        # Combine: far heading for anticipation, near heading + offset for centering
        error = near_offset * 1.5 + heading_near * 4.0 + heading_far * 6.0

        # Clamp to [-1, 1] with tiny dead zone (1cm)
        if abs(error) < 0.02:
            return 0.0
        steer = np.clip(error, -1.0, 1.0)
        return float(steer)


# ── Main Interface ──

class ETSAutoAdapter:
    """Drop-in replacement for our old lane detection + control."""

    def __init__(self):
        model_path = "etsauto_models/bevlanedet.onnx"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Download from ETSAuto releases.")
        self.detector = ETSAutoLaneDetector(model_path)
        self.controller = KeyboardLaneController()

    def process(self, frame_bgr, speed_kmh=50.0):
        """
        Process a frame and return steering command.
        Returns: steer (-1 to 1), throttle (0 to 1), info dict
        """
        lanes = self.detector.infer(frame_bgr)
        line_m = lanes.get("line_m")

        if line_m is None:
            return 0.0, 0.0, {"status": "no_lanes"}

        # Controller expects m/s
        speed_ms = speed_kmh / 3.6
        steer = self.controller.run(line_m, speed_ms)

        # Debug: show lateral offset at 10m
        near_y = float(line_m[min(20, len(line_m)-1), 1]) if len(line_m) > 20 else 0.0
        print(f"[ETSAuto] off={near_y:+.2f}m steer={steer:+.2f} L={lanes['line_l'] is not None} R={lanes['line_r'] is not None}")

        # Throttle based on curve
        if abs(steer) > 0.3:
            throttle = 0.7
        elif abs(steer) > 0.15:
            throttle = 0.85
        else:
            throttle = 1.0

        info = {
            "status": "ok",
            "steer": round(steer, 3),
            "near_offset": round(near_y, 3),
            "throttle": throttle,
            "has_left": lanes["line_l"] is not None,
            "has_right": lanes["line_r"] is not None,
        }

        return steer, throttle, info
