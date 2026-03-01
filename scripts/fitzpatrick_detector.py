# fitzpatrick_detector.py
# Estimates Fitzpatrick skin type from a video frame using the ITA (Individual Typology Angle)
# colorimetric method applied to palm pixels extracted via MediaPipe landmarks.
# Called once after the first hand is detected; the caller sets a flag to stop re-running.
#
# Method: ITA = arctan((L* - 50) / b*) * (180 / pi)
# Thresholds from Chardon et al. (1991), widely used in CV bias research.

import cv2
import numpy as np

# (min ITA angle, Fitzpatrick type int, display label) — ordered highest to lowest
_ITA_THRESHOLDS = [
    (55,  1, "I   – Very Light"),
    (41,  2, "II  – Light"),
    (28,  3, "III – Medium Light"),
    (10,  4, "IV  – Medium Dark"),
    (-30, 5, "V   – Dark"),
]

# MediaPipe palm landmark indices (wrist + MCP joints of all fingers)
_PALM_IDS = [0, 1, 5, 9, 13, 17]


def _ita_to_fitzpatrick(ita_angle: float):
    for threshold, ftype, label in _ITA_THRESHOLDS:
        if ita_angle > threshold:
            return ftype, label
    return 6, "VI  – Very Dark"


def detect_skin_type(frame_bgr: np.ndarray, hand_landmarks, image_w: int, image_h: int):
    """
    Estimate Fitzpatrick skin type from the palm region of a single BGR frame.

    Parameters
    ----------
    frame_bgr      : BGR frame that landmarks were detected on (process_frame size)
    hand_landmarks : mediapipe NormalizedLandmarkList for one hand
    image_w        : width  of frame_bgr
    image_h        : height of frame_bgr

    Returns
    -------
    (fitzpatrick_type: int 1–6, label: str)
    Returns (0, "Unknown") if the palm region yields too few pixels.
    """
    # Build palm polygon from landmark positions
    pts = []
    for idx in _PALM_IDS:
        lm = hand_landmarks.landmark[idx]
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts)

    # Create and erode mask to avoid edge bleed
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)

    # Sample Lab colour from within the palm mask
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
    lab_pixels = lab[mask == 255]

    if len(lab_pixels) < 30:
        return 0, "Unknown"

    # OpenCV Lab encoding: L in [0,255], b in [0,255] (128 = neutral)
    # Convert to CIE L* (0–100) and b* (−128 to +127)
    L_star = float(np.mean(lab_pixels[:, 0])) * (100.0 / 255.0)
    b_star = float(np.mean(lab_pixels[:, 2])) - 128.0

    if b_star == 0.0:
        return 0, "Unknown"

    ita = np.degrees(np.arctan((L_star - 50.0) / b_star))
    return _ita_to_fitzpatrick(ita)
