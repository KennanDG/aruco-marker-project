import importlib.util
import sys
from pathlib import Path

import pytest


def _load_kinematics():
    """Load kinematics.py by searching upward from this test file."""
    search_bases = [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
    for base in search_bases:
        candidate = base / "kinematics.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("kinematics", candidate)
            module = importlib.util.module_from_spec(spec)
            sys.modules["kinematics"] = module
            spec.loader.exec_module(module)
            return module
    pytest.skip("kinematics.py not found in repository")


kinematics = _load_kinematics()


def test_magnitude():
    assert kinematics.magnitude(None) is None
    assert kinematics.magnitude((3, 4)) == pytest.approx(5.0)
    assert kinematics.magnitude((0, 0)) == 0.0


def test_add_and_subtract_vectors():
    assert kinematics.add_vectors(None, (1, 1)) is None
    assert kinematics.add_vectors((1, 2), (3, 4)) == (4, 6)
    assert kinematics.subtract_vectors((3, 4), (1, 2)) == (2, 2)


def test_scale_vector():
    assert kinematics.scale_vector(None, 2) is None
    assert kinematics.scale_vector((1, 2, 3), 2) == (2, 4, 6)


def test_distance_and_displacement():
    assert kinematics.distance(None, (0, 0)) is None
    assert kinematics.distance((0, 0), (3, 4)) == pytest.approx(5.0)
    assert kinematics.displacement((0, 0), (3, 4)) == (3, 4)


def test_velocity_and_speed():
    assert kinematics.velocity((0, 0), (10, 0), 0) is None
    assert kinematics.velocity((0, 0), (10, 0), 2) == (5.0, 0.0)
    assert kinematics.speed((0, 0), (3, 4), 1) == pytest.approx(5.0)


def test_acceleration():
    assert kinematics.acceleration((0, 0), (10, 0), 0) is None
    assert kinematics.acceleration((0, 0), (10, 0), 2) == (5.0, 0.0)


def test_position_after():
    assert kinematics.position_after(None, (1, 1), 1) is None
    assert kinematics.position_after((0, 0), (2, 0), 3) == (6.0, 0.0)
    assert kinematics.position_after((0, 0), (0, 0), 2, acceleration=(2, 0)) == (4.0, 0.0)


def test_average_speed():
    assert kinematics.average_speed(None, 1) is None
    assert kinematics.average_speed(10, 2) == 5.0
    assert kinematics.average_speed(10, 0) is None
    assert kinematics.average_speed(-1, 2) is None
