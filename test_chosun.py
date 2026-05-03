#!/usr/bin/env python3
"""
test_chosun.py
Test script to calibrate the IPM (perspective transform) parameters.
Run this while ETS2 is open to see if the trapezoid covers the road correctly.
"""
import cv2
import numpy as np
from ets2_capture import find_ets2_window, capture_window_quartz, capture_fallback, select_region_manual
from chosun_lane import ChosunLaneDetector


def main():
    print("[TEST] Finding ETS2 window...")
    win_info = find_ets2_window()
    if not win_info:
        win_info = select_region_manual()

    print("[TEST] Calibrating IPM zone...")
    print("Press 'q' to quit, 's' to save a snapshot")

    # Use full frame size for detector
    detector = None

    while True:
        frame = capture_window_quartz(win_info.get("id")) if win_info.get("id") else capture_fallback(win_info)
        if frame is None:
            time.sleep(0.1)
            continue

        h, w = frame.shape[:2]

        if detector is None or detector.w != w or detector.h != h:
            detector = ChosunLaneDetector(img_width=w, img_height=h)
            print(f"[TEST] Detector initialized for {w}x{h}")

        # Draw IPM zone only
        viz = detector.draw_ipm_zone(frame)

        # Also try full detection
        steer, debug, info = detector.detect(frame)
        status = info.get("status", "?")
        print(f"\r[TEST] status={status:10s} steer={steer:+.3f} rows={info.get('rows_found', 0)}   ", end="", flush=True)

        cv2.imshow("IPM Zone (press q to quit)", viz)
        if debug is not None:
            cv2.imshow("Debug", debug)

        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite("chosun_debug.png", debug)
            print("\n[TEST] Saved chosun_debug.png")

    cv2.destroyAllWindows()
    print("\n[TEST] Done.")


if __name__ == "__main__":
    import time
    main()
