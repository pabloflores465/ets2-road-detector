#!/usr/bin/env python3
"""
Debug script for ETSAuto lane detection.
Captures one frame from ETS2 and shows all intermediate steps.
"""
import sys
import os
import numpy as np
import cv2

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from etsauto_adapter import ETSAutoLaneDetector, sigmoid


def capture_ets2_frame():
    """Capture a single frame from ETS2 window."""
    try:
        import Quartz
        from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        from Quartz import CGWindowListCreateImage, kCGWindowImageDefault
        from Quartz import CGRectMake, CGRectNull
    except ImportError:
        print("[ERROR] Quartz not available")
        return None

    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    target = None
    for win in window_list:
        name = win.get('kCGWindowName', '') or ''
        owner = win.get('kCGWindowOwnerName', '') or ''
        if 'euro truck' in name.lower() or 'euro truck' in owner.lower() or 'ets2' in name.lower():
            target = win
            break

    if target is None:
        print("[ERROR] ETS2 window not found")
        return None

    window_id = target['kCGWindowNumber']
    img = CGWindowListCreateImage(CGRectNull, 1, window_id, kCGWindowImageDefault)
    if img is None:
        return None

    w = Quartz.CGImageGetWidth(img)
    h = Quartz.CGImageGetHeight(img)
    bpr = Quartz.CGImageGetBytesPerRow(img)
    raw = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
    try:
        arr = np.frombuffer(raw, dtype=np.uint8)
    except TypeError:
        arr = np.frombuffer(raw, dtype=np.uint8)

    if bpr == w * 4:
        arr = arr.reshape((h, w, 4))
    else:
        arr = arr.reshape((h, bpr))
        arr = arr[:, :w*4]
        arr = arr.reshape((h, w, 4))

    bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    return bgr


def main():
    print("[DEBUG] Loading ETSAuto model...")
    detector = ETSAutoLaneDetector("etsauto_models/bevlanedet.onnx")

    print("[DEBUG] Capturing frame from ETS2...")
    frame = capture_ets2_frame()
    if frame is None:
        print("[ERROR] Failed to capture frame")
        return

    print(f"[DEBUG] Frame shape: {frame.shape}")

    # Save original
    cv2.imwrite("debug_original.jpg", frame)
    print("[DEBUG] Saved debug_original.jpg")

    # Preprocess
    print("[DEBUG] Preprocessing...")
    preprocessed = detector.preprocess(frame)
    print(f"[DEBUG] Preprocessed shape: {preprocessed.shape}, dtype: {preprocessed.dtype}")
    print(f"[DEBUG] Preprocessed range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")

    # Denormalize for visualization
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    vis_img = preprocessed[0] * std + mean
    vis_img = np.clip(vis_img, 0, 1)
    vis_img = (vis_img.transpose(1, 2, 0) * 255).astype(np.uint8)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("debug_preprocessed.jpg", vis_img)
    print("[DEBUG] Saved debug_preprocessed.jpg")

    # Run inference
    print("[DEBUG] Running inference...")
    import onnxruntime as ort
    session = ort.InferenceSession("etsauto_models/bevlanedet.onnx", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: preprocessed})
    seg, embedding, offset_y, z_pred = outputs

    print(f"[DEBUG] seg shape: {seg.shape}, range: [{seg.min():.3f}, {seg.max():.3f}]")
    print(f"[DEBUG] embedding shape: {embedding.shape}")
    print(f"[DEBUG] offset_y shape: {offset_y.shape}")

    # Apply sigmoid to offset_y
    offset_y = sigmoid(offset_y)

    # Save segmentation
    seg_vis = ((sigmoid(seg[0, 0]) + 0.7) / 1.7 * 255).clip(0, 255).astype(np.uint8)
    seg_vis = cv2.resize(seg_vis, (480, 320), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("debug_seg.jpg", seg_vis)
    print("[DEBUG] Saved debug_seg.jpg")

    # Clustering
    print("[DEBUG] Running embedding_post...")
    from etsauto_adapter import embedding_post
    canvas, _ = embedding_post(
        (seg, embedding),
        detector.post_conf,
        emb_margin=detector.post_emb_margin,
        min_cluster_size=detector.post_min_cluster_size
    )
    print(f"[DEBUG] canvas unique ids: {np.unique(canvas)}")

    # Save clustered segmentation
    if canvas.max() > 0:
        canvas_vis = (canvas.astype(np.float32) / canvas.max() * 255).astype(np.uint8)
    else:
        canvas_vis = np.zeros_like(canvas, dtype=np.uint8)
    canvas_vis = cv2.resize(canvas_vis, (480, 320), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("debug_canvas.jpg", canvas_vis)
    print("[DEBUG] Saved debug_canvas.jpg")

    # Convert to BEV points
    print("[DEBUG] Converting to BEV points...")
    from etsauto_adapter import bev_instance2points
    lines = bev_instance2points(
        canvas,
        max_x=detector.x_range[1],
        meter_per_pixel=(detector.meter_per_pixel, detector.meter_per_pixel),
        offset_y=offset_y[0][0]
    )
    print(f"[DEBUG] Found {len(lines)} lines")
    for i, line in enumerate(lines):
        print(f"[DEBUG]  Line {i}: {len(line)} points, x range [{line[:, 0].min():.1f}, {line[:, 0].max():.1f}], y range [{line[:, 1].min():.1f}, {line[:, 1].max():.1f}]")

    # Postprocess lines
    print("[DEBUG] Postprocessing lines...")
    lanes = detector._postprocess_lines(lines)
    print(f"[DEBUG] line_l: {lanes['line_l'] is not None}")
    print(f"[DEBUG] line_r: {lanes['line_r'] is not None}")
    print(f"[DEBUG] line_m: {lanes['line_m'] is not None}")

    if lanes['line_m'] is not None:
        print(f"[DEBUG] line_m shape: {lanes['line_m'].shape}")
        print(f"[DEBUG] line_m first 5 points:\n{lanes['line_m'][:5]}")

    print("[DEBUG] Done. Check debug_*.jpg files.")


if __name__ == "__main__":
    main()
