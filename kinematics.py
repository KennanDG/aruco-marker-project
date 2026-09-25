"""Helper functions for calculating kinematics.

Small, dependency-free utilities for reasoning about motion in 2D or 3D
space. They are ``None``-safe, mirroring the bounding-box helpers in
``object_tracking`` so callers can tell "unknown" apart from a genuine zero.
"""

import math


def _require_same_length(vector_a, vector_b):
    """Raise ``ValueError`` when two vectors have different dimensions."""
    if len(vector_a) != len(vector_b):
        raise ValueError("vectors must have the same number of dimensions")


def magnitude(vector):
    """Return the Euclidean magnitude (length) of ``vector``.

    ``vector`` being ``None`` yields ``None``; an empty vector has magnitude
    ``0.0``.
    """
    if vector is None:
        return None
    return math.sqrt(sum(component * component for component in vector))


def add_vectors(vector_a, vector_b):
    """Return the element-wise sum of two vectors.

    ``None`` inputs yield ``None``. Vectors must have matching dimensions.
    """
    if vector_a is None or vector_b is None:
        return None
    _require_same_length(vector_a, vector_b)
    return tuple(a + b for a, b in zip(vector_a, vector_b))


def subtract_vectors(vector_a, vector_b):
    """Return the element-wise difference ``vector_a - vector_b``.

    ``None`` inputs yield ``None``. Vectors must have matching dimensions.
    """
    if vector_a is None or vector_b is None:
        return None
    _require_same_length(vector_a, vector_b)
    return tuple(a - b for a, b in zip(vector_a, vector_b))


def scale_vector(vector, scalar):
    """Return ``vector`` scaled element-wise by ``scalar``.

    ``None`` inputs yield ``None``.
    """
    if vector is None or scalar is None:
        return None
    return tuple(component * scalar for component in vector)


def displacement(point_a, point_b):
    """Return the displacement vector from ``point_a`` to ``point_b``.

    ``None`` inputs yield ``None``. Points must have matching dimensions.
    """
    if point_a is None or point_b is None:
        return None
    _require_same_length(point_a, point_b)
    return tuple(b - a for a, b in zip(point_a, point_b))


def distance(point_a, point_b):
    """Return the Euclidean distance between two points.

    ``None`` inputs yield ``None``.
    """
    delta = displacement(point_a, point_b)
    if delta is None:
        return None
    return magnitude(delta)


def velocity(point_a, point_b, dt):
    """Return the average velocity vector from ``point_a`` to ``point_b``.

    ``dt`` is the elapsed time. ``None`` inputs or a non-positive ``dt`` yield
    ``None``.
    """
    if dt is None or dt <= 0:
        return None
    delta = displacement(point_a, point_b)
    if delta is None:
        return None
    return tuple(component / float(dt) for component in delta)


def speed(point_a, point_b, dt):
    """Return the scalar average speed from ``point_a`` to ``point_b``.

    ``None`` inputs or a non-positive ``dt`` yield ``None``.
    """
    velocity_vector = velocity(point_a, point_b, dt)
    if velocity_vector is None:
        return None
    return magnitude(velocity_vector)


def acceleration(velocity_a, velocity_b, dt):
    """Return the average acceleration vector between two velocity vectors.

    ``None`` inputs or a non-positive ``dt`` yield ``None``.
    """
    if dt is None or dt <= 0:
        return None
    delta = subtract_vectors(velocity_b, velocity_a)
    if delta is None:
        return None
    return tuple(component / float(dt) for component in delta)


def position_after(position, velocity, dt, acceleration=None):
    """Return the position after ``dt`` for constant-velocity/acceleration.

    Uses ``p + v * dt`` and, when ``acceleration`` is provided, the constant
    acceleration term ``0.5 * a * dt ** 2``. ``None`` position, velocity, or
    ``dt`` yields ``None``.
    """
    if position is None or velocity is None or dt is None:
        return None
    delta = scale_vector(velocity, dt)
    if acceleration is not None:
        delta = add_vectors(delta, scale_vector(acceleration, 0.5 * dt * dt))
    return add_vectors(position, delta)


def average_speed(distance_travelled, dt):
    """Return the average speed for ``distance_travelled`` over ``dt``.

    ``None`` inputs, a non-positive ``dt``, or a negative distance yield
    ``None``.
    """
    if distance_travelled is None or dt is None or dt <= 0:
        return None
    if distance_travelled < 0:
        return None
    return distance_travelled / float(dt)
