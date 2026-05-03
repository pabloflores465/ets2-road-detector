#!/usr/bin/env python3
"""
Test multiple preprocessing variants to find one that works with bevlanedet.onnx.
"""
import sys
import os
import numpy as np
import cv2
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from etsauto_adapter import sigmoid, embedding_post, bev_instance2points


def capture_ets2():
    try:
        import Quartz
        from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        from Quartz import CGWindowListCreateImage, kCGWindowImageDefault
        from Quartz import CGRectNull
    except ImportError:
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


def preprocess_variant(img, variant):
    h, w = img.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    if variant == "A":
        # Original ETSAuto: crop top 50px from 640px tall image
        crop_top = int(h * 50 / 640)
        img = img[crop_top:, :, :]
        img = cv2.resize(img, (480, 320))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
    elif variant == "B":
        # No crop, full image
        img = cv2.resize(img, (480, 320))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
    elif variant == "C":
        # Crop top 50px from actual resolution (not proportional)
        img = img[50:, :, :]
        img = cv2.resize(img, (480, 320))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
    elif variant == "D":
        # Crop bottom 15% (remove dashboard)
        crop_bottom = int(h * 0.85)
        img = img[:crop_bottom, :, :]
        img = cv2.resize(img, (480, 320))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
    elif variant == "E":
        # Center crop to 16:9 then resize
        target_ratio = 16 / 9
        current_ratio = w / h
        if current_ratio > target_ratio:
            new_w = int(h * target_ratio)
            start_x = (w - new_w) // 2
            img = img[:, start_x:start_x + new_w, :]
        else:
            new_h = int(w / target_ratio)
            start_y = (h - new_h) // 2
            img = img[start_y:start_y + new_h, :, :]
        img = cv2.resize(img, (480, 320))
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
    elif variant == "F":
        # Same as A but convert BGR->RGB
        crop_top = int(h * 50 / 640)
        img = img[crop_top:, :, :]
        img = cv2.resize(img, (480, 320))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
    elif variant == "G":
        # No normalization (just 0-1 range)
        crop_top = int(h * 50 / 640)
        img = img[crop_top:, :, :]
        img = cv2.resize(img, (480, 320))
        img = img.astype(np.float32) / 255.0
    elif variant == "H":
        # No division by 255 (model expects 0-255?)
        crop_top = int(h * 50 / 640)
        img = img[crop_top:, :, :]
        img = cv2.resize(img, (480, 320))
        img = img.astype(np.float32)
        img = (img - mean * 255) / (std * 255)

    return np.expand_dims(img.transpose(2, 0, 1), axis=0)


def main():
    print("[TEST] Capturing ETS2 frame...")
    frame = capture_ets2()
    if frame is None:
        print("[TEST] Could not capture ETS2.")
        return

    print(f"[TEST] Frame shape: {frame.shape}")
    cv2.imwrite("test_capture.jpg", frame)

    sess = ort.InferenceSession("etsauto_models/bevlanedet.onnx",
                                 providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    best_variant = None
    best_score = -1

    for variant in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        pre = preprocess_variant(frame, variant)
        out = sess.run(output_names, {input_name: pre})
        out_dict = dict(zip(output_names, out))
        seg = out_dict.get('seg', out[0])

        seg_sig = sigmoid(seg[0, 0])
        max_val = seg_sig.max()
        mean_val = seg_sig.mean()
        num_above_05 = (seg_sig > 0.5).sum()

        print(f"[TEST] Variant {variant}: seg max={max_val:.4f}, mean={mean_val:.4f}, >0.5={num_above_05}")

        # Save visualization
        vis = (seg_sig * 255).clip(0, 255).astype(np.uint8)
        vis = cv2.resize(vis, (480, 320), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"test_seg_{variant}.jpg", vis)

        if num_above_05 > best_score:
            best_score = num_above_05
            best_variant = variant

    print(f"[TEST] BEST VARIANT: {best_variant} with {best_score} pixels above 0.5")
    print("[TEST] Check test_seg_*.jpg files to see which variant detects lanes.")


if __name__ == "__main__":
    main()
