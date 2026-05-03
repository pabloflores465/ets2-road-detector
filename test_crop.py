#!/usr/bin/env python3
"""
Show the captured ETS2 frame and allow interactive cropping to find the right region.
"""
import cv2
import numpy as np
from ets2_capture import find_ets2_window, capture_window_quartz

def main():
    win = find_ets2_window()
    if not win:
        print("[ERROR] ETS2 window not found")
        return

    frame = capture_window_quartz(win["id"])
    if frame is None:
        print("[ERROR] Capture failed")
        return

    print(f"[INFO] Captured: {frame.shape}")
    cv2.imwrite("test_full_capture.jpg", frame)

    # Show original
    display = cv2.resize(frame, (1280, 800))
    cv2.imshow("Original (resized)", display)

    # Try different crops
    h, w = frame.shape[:2]

    # Crop 1: remove top 22px (title bar) + bottom to make 16:9
    target_h = int(w / 16 * 9)
    if target_h < h:
        c1 = frame[22:22+target_h, :, :]
    else:
        c1 = frame[22:, :, :]
    cv2.imwrite("test_crop_16_9.jpg", c1)
    print(f"[INFO] 16:9 crop: {c1.shape}")

    # Crop 2: ETSAuto style [50:640] scaled
    crop_top = int(h * 50 / 640)
    crop_bottom = int(h * 590 / 640)
    c2 = frame[crop_top:crop_bottom, :, :]
    cv2.imwrite("test_crop_etsauto.jpg", c2)
    print(f"[INFO] ETSAuto crop: {c2.shape}")

    # Crop 3: remove all borders (detect grey edges)
    # Find first/last rows/cols that are not uniform grey
    def find_content_edges(img, threshold=10):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Top edge
        top = 0
        for y in range(min(50, h)):
            if np.std(gray[y, :]) > threshold:
                top = y
                break
        # Bottom edge
        bottom = h
        for y in range(h-1, max(h-50, 0), -1):
            if np.std(gray[y, :]) > threshold:
                bottom = y + 1
                break
        # Left edge
        left = 0
        for x in range(min(50, w)):
            if np.std(gray[:, x]) > threshold:
                left = x
                break
        # Right edge
        right = w
        for x in range(w-1, max(w-50, 0), -1):
            if np.std(gray[:, x]) > threshold:
                right = x + 1
                break
        return top, bottom, left, right

    t, b, l, r = find_content_edges(frame)
    c3 = frame[t:b, l:r, :]
    cv2.imwrite("test_crop_content.jpg", c3)
    print(f"[INFO] Content crop: {c3.shape} (edges: t={t}, b={b}, l={l}, r={r})")

    print("[INFO] Saved test_full_capture.jpg, test_crop_16_9.jpg, test_crop_etsauto.jpg, test_crop_content.jpg")
    print("[INFO] Check these images to see which crop shows the game correctly.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
