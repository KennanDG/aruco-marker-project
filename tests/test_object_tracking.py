import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cv2
from object_tracking import detect_moving_objects


def test_detect_moving_objects():
    """Unit test for the moving object detection function."""
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=1, varThreshold=1, detectShadows=False
    )
    # learn a static background
    bg_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    bg_subtractor.apply(bg_frame)

    # frame with a moving white blob
    fg_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(fg_frame, (30, 30), (80, 80), (255, 255, 255), -1)

    result, bboxes = detect_moving_objects(fg_frame, bg_subtractor, min_area=500)
    # area is 50*50 = 2500 > 500, so a bounding box should be found
    assert len(bboxes) > 0, "Should detect at least one moving object"
    # a green bounding box (0,255,0) should be drawn
    assert np.any((result[:, :, 1] == 255) & (result[:, :, 0] == 0)), "Green box not found on frame"
