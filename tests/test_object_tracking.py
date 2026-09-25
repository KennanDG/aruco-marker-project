import importlib.util
import sys
from pathlib import Path

import pytest


def _load_object_tracking():
    """Load object_tracking.py by searching upward from this test file."""
    search_bases = [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    for base in search_bases:
        candidate = base / "object_tracking.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("object_tracking", candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules["object_tracking"] = module
            spec.loader.exec_module(module)
            return module
    pytest.skip("object_tracking.py not found in repository")


object_tracking = _load_object_tracking()


def test_bbox_area_handles_empty_input():
    assert object_tracking.bbox_area(None) == 0
    assert object_tracking.bbox_area((0, 0, 0, 0)) == 0


def test_bbox_area_basic():
    assert object_tracking.bbox_area((10, 20, 30, 40)) == 1200


def test_bbox_center_handles_none():
    assert object_tracking.bbox_center(None) is None


def test_bbox_center_basic():
    assert object_tracking.bbox_center((0, 0, 10, 20)) == (5.0, 10.0)


def test_bbox_iou_identical_boxes():
    assert object_tracking.bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_bbox_iou_disjoint_boxes():
    assert object_tracking.bbox_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_bbox_iou_handles_none():
    assert object_tracking.bbox_iou(None, (0, 0, 5, 5)) == 0.0


def test_filter_bboxes_handles_empty_input():
    assert object_tracking.filter_bboxes(None) == []
    assert object_tracking.filter_bboxes([]) == []


def test_filter_bboxes_by_area():
    bboxes = [(0, 0, 10, 10), (0, 0, 2, 2), (0, 0, 5, 5)]
    assert object_tracking.filter_bboxes(bboxes, min_area=20) == [(0, 0, 10, 10), (0, 0, 5, 5)]
    assert object_tracking.filter_bboxes(bboxes, max_area=25) == [(0, 0, 2, 2), (0, 0, 5, 5)]


def test_union_bbox_handles_none():
    assert object_tracking.union_bbox(None, None) is None
    assert object_tracking.union_bbox(None, (1, 2, 3, 4)) == (1, 2, 3, 4)
    assert object_tracking.union_bbox((1, 2, 3, 4), None) == (1, 2, 3, 4)


def test_union_bbox_basic():
    assert object_tracking.union_bbox((0, 0, 10, 10), (5, 5, 10, 10)) == (0, 0, 15, 15)


def test_merge_overlapping_bboxes_handles_empty_input():
    assert object_tracking.merge_overlapping_bboxes(None) == []
    assert object_tracking.merge_overlapping_bboxes([]) == []


def test_merge_overlapping_bboxes_merges_overlap():
    merged = object_tracking.merge_overlapping_bboxes([(0, 0, 10, 10), (5, 5, 10, 10)])
    assert merged == [(0, 0, 15, 15)]


def test_merge_overlapping_bboxes_keeps_disjoint():
    bboxes = [(0, 0, 10, 10), (100, 100, 10, 10)]
    assert object_tracking.merge_overlapping_bboxes(bboxes) == bboxes


def test_match_tracks_handles_empty_input():
    assert object_tracking.match_tracks(None, [(0, 0, 5, 5)]) == []
    assert object_tracking.match_tracks([(0, 0, 5, 5)], []) == []


def test_match_tracks_matches_by_position_deterministically():
    previous = [(0, 0, 10, 10), (100, 100, 10, 10)]
    current = [(1, 1, 10, 10), (101, 101, 10, 10)]
    matches = object_tracking.match_tracks(previous, current, iou_threshold=0.3)
    assert [(prev, cur) for prev, cur, _ in matches] == [(0, 0), (1, 1)]
    assert all(iou >= 0.3 for _, _, iou in matches)


def test_match_tracks_returns_empty_below_threshold():
    matches = object_tracking.match_tracks([(0, 0, 5, 5)], [(50, 50, 5, 5)], iou_threshold=0.3)
    assert matches == []
