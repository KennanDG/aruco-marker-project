import cv2
import numpy as np


def detect_moving_objects(frame, bg_subtractor, min_area=500):
    """
    Detect moving objects using background subtraction.
    Returns the frame with bounding boxes drawn and a list of bounding boxes.
    """
    fg_mask = bg_subtractor.apply(frame)
    # morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    # find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame, bboxes
