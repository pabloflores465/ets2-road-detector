#!/usr/bin/env python3
"""
Test ETSAuto lane detection on a single captured frame.
Run this while ETS2 is open in cabin view on a road with visible lane markings.
"""
import sys
import os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def capture_ets2():
    try:
        import Quartz
        from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        from Quartz import CGWindowListCreateImage, kCGWindowImageDefault
        from Quartz import CGRectNull
    except ImportError:
        print("[ERROR] Quartz not available")
        return None

    window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    for win in window_list:
        name = win.get('kCGWindowName', '') or ''
        owner = win.get('kCGWindowOwnerName', '') or ''
        if 'euro truck' in name.lower() or 'euro truck' in owner.lower():
            img = CGWindowListCreateImage(CGRectNull, 1, win['kCGWindowNumber'], kCGWindowImageDefault)
            if img is None:
                return None
            w = Quartz.CGImageGetWidth(img)
            h = Quartz.CGImageGetHeight(img)
            bpr = Quartz.CGImageGetBytesPerRow(img)
            raw = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
            arr = np.frombuffer(raw, dtype=np.uint8)
            if bpr == w * 4:
                arr = arr.reshape((h, w, 4))
            else:
                arr = arr.reshape((h, bpr))[:, :w*4].reshape((h, w, 4))
            return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    return None


def main():
    from etsauto_adapter import ETSAutoLaneDetector, sigmoid, embedding_post, bev_instance2points
    import onnxruntime as ort

    print("[TEST] Capturing ETS2 frame...")
    frame = capture_ets2()
    if frame is None:
        print("[TEST] Could not capture ETS2. Is the game running?")
        return

    print(f"[TEST] Captured frame: {frame.shape}")
    cv2.imwrite("test_capture.jpg", frame)
    print("[TEST] Saved test_capture.jpg")

    detector = ETSAutoLaneDetector("etsauto_models/bevlanedet.onnx")

    # Preprocess
    pre = detector.preprocess(frame)
    print(f"[TEST] Preprocessed: {pre.shape}, dtype={pre.dtype}, range=[{pre.min():.3f}, {pre.max():.3f}]")

    # Inference
    sess = ort.InferenceSession("etsauto_models/bevlanedet.onnx", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: pre})
    seg, emb, offset_y, z_pred = out

    print(f"[TEST] seg: {seg.shape} range=[{seg.min():.3f}, {seg.max():.3f}]")

    # Segmentation after sigmoid
    seg_sig = sigmoid(seg[0, 0])
    print(f"[TEST] seg sigmoid: range=[{seg_sig.min():.3f}, {seg_sig.max():.3f}], mean={seg_sig.mean():.3f}")
    print(f"[TEST] seg > 0.5: {(seg_sig > 0.5).sum()} pixels")

    # Save seg visualization
    cv2.imwrite("test_seg.jpg", (seg_sig * 255).astype(np.uint8))

    # Clustering
    offset_y = sigmoid(offset_y)
    canvas, clusters = embedding_post((seg, emb), detector.post_conf,
                                       emb_margin=detector.post_emb_margin,
                                       min_cluster_size=detector.post_min_cluster_size)
    print(f"[TEST] Canvas unique values: {np.unique(canvas)}")
    print(f"[TEST] Number of clusters: {len(clusters)}")

    cv2.imwrite("test_canvas.jpg", (canvas.astype(np.float32) / max(canvas.max(), 1) * 255).astype(np.uint8))

    # BEV points
    lines = bev_instance2points(canvas, max_x=detector.x_range[1],
                                meter_per_pixel=(detector.meter_per_pixel, detector.meter_per_pixel),
                                offset_y=offset_y[0][0])
    print(f"[TEST] Lines found: {len(lines)}")
    for i, l in enumerate(lines):
        print(f"[TEST]  Line {i}: {len(l)} pts, x=[{l[:,0].min():.1f},{l[:,0].max():.1f}], y=[{l[:,1].min():.1f},{l[:,1].max():.1f}]")

    # Postprocess
    lanes = detector._postprocess_lines(lines)
    print(f"[TEST] line_l: {'YES' if lanes['line_l'] is not None else 'NO'}")
    print(f"[TEST] line_r: {'YES' if lanes['line_r'] is not None else 'NO'}")
    print(f"[TEST] line_m: {'YES' if lanes['line_m'] is not None else 'NO'}")

    if lanes['line_m'] is not None:
        print(f"[TEST] line_m shape: {lanes['line_m'].shape}")
        print(f"[TEST] line_m[:5]:\n{lanes['line_m'][:5]}")


if __name__ == "__main__":
    main()
