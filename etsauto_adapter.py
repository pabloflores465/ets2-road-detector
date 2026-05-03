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
        # ETSAuto crops top 50 pixels (sky)
        img = img[50:640, :, :]
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        # Normalize (ImageNet)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        return np.expand_dims(img.transpose(2, 0, 1), axis=0)

    def infer(self, img_bgr):
        input_tensor = self.preprocess(img_bgr)
        seg, embedding, offset_y, z_pred = self.session.run(
            None, {self.input_name: input_tensor}
        )
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

    def _postprocess_lines(self, lines):
        # Round lines and get skeleton
        lines_temp = []
        for line in lines:
            if len(line) <= 30:
                continue
            lines_temp.append(horizontal_rounding(line))

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
        if line_l is None and line_r is not None:
            line_l = line_r - np.array([0, self.lane_width])
        elif line_l is not None and line_r is None:
            line_r = line_l + np.array([0, self.lane_width])

        # Get middle line (trajectory)
        line_m = get_skeleton(line_l, line_r)

        return {
            "line_l": line_l,
            "line_r": line_r,
            "line_m": line_m,
            "line_ll": line_ll,
            "line_rr": line_rr,
        }


# ── Pure Pursuit Controller ──

class PurePursuit:
    """Geometric controller for lane following."""

    def __init__(self):
        self.ld = 20.0  # lookahead distance
        self.lf = 3.63  # lookahead offset
        self.wheelbase = 8.0  # wheelbase

    def run(self, trajectory, speed):
        """
        Args:
            trajectory: Nx2 array of (x, y) points in meters
            speed: current speed (km/h)
        Returns:
            steering angle in radians
        """
        if trajectory is None or len(trajectory) < 40:
            return 0.0

        # Adjust lookahead based on speed
        self.ld = 1.0 * speed + self.lf

        # Vehicle state (rear axle)
        robot_state = np.array([-self.wheelbase, 0.0])

        # Target point on trajectory (ahead)
        target = np.average(trajectory[30:40], axis=0)

        dy = target[0] - robot_state[0]
        dx = target[1] - robot_state[1]

        alpha = np.arctan(dx / (dy + 1e-6))
        ang = math.atan2(2.0 * self.wheelbase * math.sin(alpha), self.ld)
        return ang


# ── Main Interface ──

class ETSAutoAdapter:
    """Drop-in replacement for our old lane detection + control."""

    def __init__(self):
        model_path = "etsauto_models/bevlanedet.onnx"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}. Download from ETSAuto releases.")
        self.detector = ETSAutoLaneDetector(model_path)
        self.controller = PurePursuit()

    def process(self, frame_bgr, speed_kmh=50.0):
        """
        Process a frame and return steering command.
        Returns: steer (-1 to 1), throttle (0 to 1), info dict
        """
        lanes = self.detector.infer(frame_bgr)
        line_m = lanes.get("line_m")

        if line_m is None:
            return 0.0, 0.0, {"status": "no_lanes"}

        # Pure pursuit steering
        ang_rad = self.controller.run(line_m, speed_kmh)

        # Normalize to -1..1 (typical range ±0.4 rad)
        steer = float(np.clip(ang_rad / 0.4, -1.0, 1.0))

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
            "angle_rad": round(ang_rad, 3),
            "throttle": throttle,
            "has_left": lanes["line_l"] is not None,
            "has_right": lanes["line_r"] is not None,
        }

        return steer, throttle, info
