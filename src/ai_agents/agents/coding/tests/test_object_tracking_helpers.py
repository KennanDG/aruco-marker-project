"""Focused unit tests for the object tracking helpers in ``object_tracking.py``.

The module under test imports ``cv2`` and ``numpy``; these tests skip cleanly
when either dependency is unavailable so they never fail for environmental
reasons.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("numpy")


def _load_object_tracking():
    """Load ``object_tracking`` even when the repo root is not on ``sys.path``."""
    try:
        return importlib.import_module("object_tracking")
    except ImportError:
        pass

    test_dir = Path(__file__).resolve().parent
    for parent in [test_dir] + list(test_dir.parents):
        candidate = parent / "object_tracking.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("object_tracking", candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules["object_tracking"] = module
            spec.loader.exec_module(module)
            return module

    pytest.skip("object_tracking.py could not be located in the repository tree")


object_tracking = _load_object_tracking()


def test_existing_public_helpers_are_preserved():
    for name in (
        "detect_moving_objects",
        "bbox_area",
        "bbox_center",
        "bbox_iou",
        "union_bbox",
        "filter_bboxes",
        "merge_overlapping_bboxes",
        "match_tracks",
    ):
        assert callable(getattr(object_tracking, name)), name


def test_bbox_area_and_center_handle_none():
    assert object_tracking.bbox_area(None) == 0
    assert object_tracking.bbox_center(None) is None
    assert object_tracking.bbox_area((10, 20, 5, 4)) == 20
    assert object_tracking.bbox_area((10, 20, -5, 4)) == 0
    assert object_tracking.bbox_center((10, 20, 5, 4)) == (12.5, 22.0)


def test_bbox_iou_and_union_handle_none_and_overlap():
    assert object_tracking.bbox_iou(None, (0, 0, 1, 1)) == 0.0
    assert object_tracking.bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert object_tracking.bbox_iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert object_tracking.union_bbox(None, None) is None
    assert object_tracking.union_bbox(None, (1, 2, 3, 4)) == (1, 2, 3, 4)
    assert object_tracking.union_bbox((0, 0, 10, 10), (5, 5, 10, 10)) == (0, 0, 15, 15)


def test_filter_and_merge_bboxes_preserve_order():
    bboxes = [(0, 0, 10, 10), (100, 100, 2, 2), None]
    assert object_tracking.filter_bboxes(None) == []
    assert object_tracking.filter_bboxes([]) == []
    assert object_tracking.filter_bboxes(bboxes, min_area=10) == [(0, 0, 10, 10)]
    assert object_tracking.merge_overlapping_bboxes(None) == []
    assert object_tracking.merge_overlapping_bboxes([(0, 0, 10, 10), (1, 1, 10, 10)]) == [
        (0, 0, 11, 11)
    ]


def test_match_tracks_is_deterministic_and_empty_safe():
    assert object_tracking.match_tracks(None, [(0, 0, 1, 1)]) == []
    assert object_tracking.match_tracks([], []) == []
    matches = object_tracking.match_tracks([(0, 0, 10, 10), (0, 0, 10, 10)], [(0, 0, 10, 10)])
    assert [match[:2] for match in matches] == [(0, 0)]
    assert matches[0][2] == pytest.approx(1.0)


def test_bbox_from_center_round_trips_and_handles_none():
    assert object_tracking.bbox_from_center(None, (4, 4)) is None
    assert object_tracking.bbox_from_center((5, 5), None) is None
    bbox = object_tracking.bbox_from_center((5, 5), (4, 2))
    assert bbox == (3.0, 4.0, 4, 2)
    assert object_tracking.bbox_center(bbox) == (5.0, 5.0)
    assert object_tracking.bbox_from_center((5, 5), (-4, 2)) == (5.0, 4.0, 0, 2)


def test_bbox_contains_point():
    bbox = (0, 0, 10, 10)
    assert object_tracking.bbox_contains_point(bbox, (5, 5)) is True
    assert object_tracking.bbox_contains_point(bbox, (9, 9)) is True
    assert object_tracking.bbox_contains_point(bbox, (20, 5)) is False
    assert object_tracking.bbox_contains_point(bbox, (11, 0), margin=2) is True
    assert object_tracking.bbox_contains_point(None, (0, 0)) is False
    assert object_tracking.bbox_contains_point(bbox, None) is False


def test_clamp_bbox():
    assert object_tracking.clamp_bbox(None, 100, 100) is None
    assert object_tracking.clamp_bbox((10, 10, 20, 20), 100, 100) == (10, 10, 20, 20)
    assert object_tracking.clamp_bbox((-5, -5, 10, 10), 100, 100) == (0, 0, 5, 5)
    assert object_tracking.clamp_bbox((90, 90, 40, 40), 100, 100) == (90, 90, 10, 10)
    assert object_tracking.clamp_bbox((200, 200, 10, 10), 100, 100) is None
    assert object_tracking.clamp_bbox((0, 0, 10, 10), 0, 100) is None


def test_scale_bbox():
    assert object_tracking.scale_bbox(None, 2) is None
    assert object_tracking.scale_bbox((10, 20, 30, 40), 2) == (10, 20, 60, 80)
    assert object_tracking.scale_bbox((10, 20, 30, 40), 2, 0.5) == (10, 20, 60, 20)


def test_bbox_distance():
    assert object_tracking.bbox_distance(None, (0, 0, 2, 2)) is None
    assert object_tracking.bbox_distance((0, 0, 2, 2), (0, 0, 2, 2)) == 0.0
    assert object_tracking.bbox_distance((0, 0, 2, 2), (3, 4, 2, 2)) == pytest.approx(5.0)


def test_bboxes_to_centers_skips_none():
    assert object_tracking.bboxes_to_centers(None) == []
    assert object_tracking.bboxes_to_centers([]) == []
    assert object_tracking.bboxes_to_centers([(0, 0, 2, 2), None, (4, 4, 2, 2)]) == [
        (1.0, 1.0),
        (5.0, 5.0),
    ]


def test_summarize_bboxes_handles_empty_and_valid_input():
    assert object_tracking.summarize_bboxes(None) == {
        "count": 0,
        "total_area": 0,
        "mean_area": 0.0,
        "mean_center": None,
    }
    assert object_tracking.summarize_bboxes([]) == object_tracking.summarize_bboxes(None)
    summary = object_tracking.summarize_bboxes([(0, 0, 2, 2), None, (4, 4, 4, 4)])
    assert summary["count"] == 2
    assert summary["total_area"] == 20
    assert summary["mean_area"] == pytest.approx(10.0)
    assert summary["mean_center"] == pytest.approx((3.5, 3.5))
