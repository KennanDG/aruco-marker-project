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


def bbox_area(bbox):
    """
    Return the area of an (x, y, w, h) bounding box.

    ``None`` yields 0 and negative width/height are treated as zero.
    """
    if bbox is None:
        return 0
    _, _, w, h = bbox
    return max(0, w) * max(0, h)


def bbox_center(bbox):
    """
    Return the (x, y) center of an (x, y, w, h) bounding box as floats.

    ``None`` yields ``None``.
    """
    if bbox is None:
        return None
    x, y, w, h = bbox
    return (x + w / 2.0, y + h / 2.0)


def bbox_iou(bbox_a, bbox_b):
    """
    Compute the intersection-over-union of two (x, y, w, h) boxes.

    Returns 0.0 when either box is ``None``, the boxes do not overlap, or the
    combined area is zero.
    """
    if bbox_a is None or bbox_b is None:
        return 0.0
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = bbox_area(bbox_a) + bbox_area(bbox_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / float(union)


def union_bbox(bbox_a, bbox_b):
    """
    Return the smallest (x, y, w, h) box covering both inputs.

    ``None`` inputs are ignored; if both are ``None`` the result is ``None``.
    """
    if bbox_a is None:
        return None if bbox_b is None else tuple(bbox_b)
    if bbox_b is None:
        return tuple(bbox_a)
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    x1 = min(ax, bx)
    y1 = min(ay, by)
    x2 = max(ax + aw, bx + bw)
    y2 = max(ay + ah, by + bh)
    return (x1, y1, x2 - x1, y2 - y1)


def filter_bboxes(bboxes, min_area=0, max_area=None):
    """
    Filter an iterable of bounding boxes by area.

    Returns a new list preserving input order. ``bboxes=None`` or empty input
    yields an empty list. ``max_area=None`` disables the upper bound.
    """
    if not bboxes:
        return []
    result = []
    for bbox in bboxes:
        area = bbox_area(bbox)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        result.append(bbox)
    return result


def merge_overlapping_bboxes(bboxes, iou_threshold=0.3):
    """
    Merge bounding boxes that overlap more than ``iou_threshold``.

    Boxes are processed in input order; each box either merges with the first
    matching accumulated box or starts a new one. ``None`` entries are skipped.
    Empty or ``None`` input yields an empty list. The result is deterministic.
    """
    if not bboxes:
        return []
    merged = []
    for bbox in bboxes:
        if bbox is None:
            continue
        for index, existing in enumerate(merged):
            if bbox_iou(existing, bbox) >= iou_threshold:
                merged[index] = union_bbox(existing, bbox)
                break
        else:
            merged.append(tuple(bbox))
    return merged


def match_tracks(previous_bboxes, current_bboxes, iou_threshold=0.3):
    """
    Greedily match previous tracks to current detections by IoU.

    Returns a list of ``(previous_index, current_index, iou)`` tuples, sorted by
    descending IoU with ties broken by index order so the result is
    deterministic. ``None`` entries are skipped; empty or ``None`` inputs yield
    an empty list.
    """
    if not previous_bboxes or not current_bboxes:
        return []
    candidates = []
    for prev_index, prev_bbox in enumerate(previous_bboxes):
        if prev_bbox is None:
            continue
        for cur_index, cur_bbox in enumerate(current_bboxes):
            if cur_bbox is None:
                continue
            iou = bbox_iou(prev_bbox, cur_bbox)
            if iou >= iou_threshold:
                candidates.append((prev_index, cur_index, iou))
    candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
    used_prev = set()
    used_cur = set()
    matches = []
    for prev_index, cur_index, iou in candidates:
        if prev_index in used_prev or cur_index in used_cur:
            continue
        used_prev.add(prev_index)
        used_cur.add(cur_index)
        matches.append((prev_index, cur_index, iou))
    return matches


def bbox_from_center(center, size):
    """
    Build an (x, y, w, h) bounding box from a center point and a (w, h) size.

    ``center`` or ``size`` being ``None`` yields ``None``. Negative width or
    height in ``size`` are treated as zero so the result is never degenerate by
    sign.
    """
    if center is None or size is None:
        return None
    cx, cy = center
    w, h = size
    w = max(0, w)
    h = max(0, h)
    return (cx - w / 2.0, cy - h / 2.0, w, h)


def bbox_contains_point(bbox, point, margin=0):
    """
    Return ``True`` when ``point`` lies inside ``bbox``.

    ``bbox`` or ``point`` being ``None`` yields ``False``. ``margin`` expands the
    box on every side, which is handy for tolerant hit tests.
    """
    if bbox is None or point is None:
        return False
    x, y, w, h = bbox
    px, py = point
    if px < x - margin or px > x + w + margin:
        return False
    if py < y - margin or py > y + h + margin:
        return False
    return True


def clamp_bbox(bbox, frame_width, frame_height):
    """
    Clip an (x, y, w, h) box to a ``frame_width`` x ``frame_height`` frame.

    ``bbox`` being ``None``, non-positive frame sizes, and boxes that fall
    completely outside the frame all yield ``None``.
    """
    if bbox is None:
        return None
    if frame_width <= 0 or frame_height <= 0:
        return None
    x, y, w, h = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_width, x + w)
    y2 = min(frame_height, y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def scale_bbox(bbox, scale_x, scale_y=None):
    """
    Scale the width and height of an (x, y, w, h) box, keeping its corner fixed.

    When ``scale_y`` is ``None`` the same factor is applied to both axes.
    ``bbox`` being ``None`` yields ``None``.
    """
    if bbox is None:
        return None
    if scale_y is None:
        scale_y = scale_x
    x, y, w, h = bbox
    return (x, y, w * scale_x, h * scale_y)


def bbox_distance(bbox_a, bbox_b):
    """
    Return the Euclidean distance between the centers of two boxes.

    ``None`` inputs yield ``None`` so callers can tell "unknown" apart from a
    genuine zero distance.
    """
    center_a = bbox_center(bbox_a)
    center_b = bbox_center(bbox_b)
    if center_a is None or center_b is None:
        return None
    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]
    return (dx * dx + dy * dy) ** 0.5


def bboxes_to_centers(bboxes):
    """
    Convert an iterable of boxes into a list of ``(x, y)`` centers.

    ``bboxes=None`` or empty input yields an empty list. ``None`` entries are
    skipped, so the result may be shorter than the input.
    """
    if bboxes is None:
        return []
    centers = []
    for bbox in bboxes:
        center = bbox_center(bbox)
        if center is not None:
            centers.append(center)
    return centers


def _empty_bbox_summary():
    """Return a fresh, zeroed result for :func:`summarize_bboxes`."""
    return {
        "count": 0,
        "total_area": 0,
        "mean_area": 0.0,
        "mean_center": None,
    }


def summarize_bboxes(bboxes):
    """
    Summarize an iterable of boxes into deterministic aggregate statistics.

    Returns a dict with ``count``, ``total_area``, ``mean_area`` and
    ``mean_center``. ``None`` entries are skipped. When no valid boxes are
    available the dict reports zero/``None`` values instead of raising.
    """
    if bboxes is None:
        return _empty_bbox_summary()
    valid = []
    for bbox in bboxes:
        if bbox is not None:
            valid.append(tuple(bbox))
    if not valid:
        return _empty_bbox_summary()
    count = len(valid)
    total_area = sum(bbox_area(bbox) for bbox in valid)
    centers = [bbox_center(bbox) for bbox in valid]
    mean_x = sum(center[0] for center in centers) / float(count)
    mean_y = sum(center[1] for center in centers) / float(count)
    return {
        "count": count,
        "total_area": total_area,
        "mean_area": total_area / float(count),
        "mean_center": (mean_x, mean_y),
    }
