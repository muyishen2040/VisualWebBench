"""
action_label_detector.py

Prototype: given an image with red letter-labelled boxes, detect
(red) bounding boxes (no OCR, no labels).

Dependencies (pip only):
    pip install opencv-python numpy

Typical usage from code:
    from action_label_detector import detect_red_boxes

    img = cv2.imread("action_ground.png")
    boxes = detect_red_boxes(img)   # list of (x0, y0, x1, y1)

CLI usage:
    python action_label_detector.py action_ground.png --debug_dir=./bboxes

This will:
    - print a JSON list of {"bbox": [x0,y0,x1,y1]} to stdout
    - optionally dump red-mask debug image into --debug_dir
"""

from __future__ import annotations

from typing import List, Tuple, Optional

import argparse
import json
import os

import cv2
import numpy as np


# ---------------------- Red box detection ---------------------- #

def detect_red_boxes(
    img_bgr: np.ndarray,
    min_area: int = 300,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect red rectangles (candidate action regions) via HSV thresholding and contour analysis.

    Returns a list of bounding boxes in pixel coords: (x0, y0, x1, y1).
    """
    if img_bgr is None or img_bgr.size == 0:
        return []

    h, w = img_bgr.shape[:2]

    # Convert to HSV for robust red detection
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Red hue appears at BOTH low and high ranges in HSV
    lower_red1 = np.array([0, 80, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 80, 80], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Remove noise and connect fragmented borders
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        area = w_box * h_box
        if area < min_area:
            continue

        # Clamp for safety
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + w_box)
        y1 = min(h, y + h_box)
        boxes.append((x0, y0, x1, y1))

    return boxes


def detect_red_boxes_from_path(
    image_path: str,
    min_area: int = 300,
) -> List[Tuple[int, int, int, int]]:
    """
    Convenience wrapper: load image from disk and run detection.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    return detect_red_boxes(img, min_area=min_area)


# ---------------------- CLI ---------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Detect red boxes and output bbox list."
    )
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument(
        "--debug_dir",
        type=str,
        default=None,
        help="Optional directory to dump red-mask debug image.",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=300,
        help="Minimum bbox area to keep.",
    )
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        raise RuntimeError(f"Failed to load image: {args.image_path}")

    boxes = detect_red_boxes(img, min_area=args.min_area)

    out = [
        {"bbox": [int(x0), int(y0), int(x1), int(y1)]}
        for (x0, y0, x1, y1) in boxes
    ]
    print(json.dumps(out, indent=2))

    if args.debug_dir is not None:
        os.makedirs(args.debug_dir, exist_ok=True)
        # optional: visualize red mask
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 80, 80], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([160, 80, 80], dtype=np.uint8)
        upper_red2 = np.array([179, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        dbg_path = os.path.join(args.debug_dir, "red_mask.png")
        cv2.imwrite(dbg_path, mask)
        print(f"[LabelDet] Saved red mask to {dbg_path}")


if __name__ == "__main__":
    main()
