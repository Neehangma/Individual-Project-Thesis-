"""Geometry helpers for estimating a 2D elbow angle from pose landmarks."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def calculate_elbow_angle(
    shoulder: Sequence[float], elbow: Sequence[float], wrist: Sequence[float]
) -> float:
    """Return the shoulder-elbow-wrist angle in degrees (0..180).

    Coordinates may be normalized image coordinates or pixel coordinates. The
    measurement is an image-plane estimate from a single camera, not a clinical
    or three-dimensional joint measurement.
    """
    points = [np.asarray(point, dtype=float)[:2] for point in (shoulder, elbow, wrist)]
    if any(point.size != 2 or not np.isfinite(point).all() for point in points):
        raise ValueError("Landmarks must contain two finite coordinates")
    upper_arm = points[0] - points[1]
    forearm = points[2] - points[1]
    lengths = np.linalg.norm(upper_arm), np.linalg.norm(forearm)
    if min(lengths) <= 1e-12:
        raise ValueError("Cannot calculate an angle when a segment has zero length")
    cosine = float(np.dot(upper_arm, forearm) / (lengths[0] * lengths[1]))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
