"""Angle conversion and wrapping helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def deg2rad(deg: float) -> float:
    """Convert degrees to radians.

    Args:
        deg: Angle in degrees.

    Returns:
        Angle in radians.
    """
    return math.radians(deg)


def rad2deg(rad: float) -> float:
    """Convert radians to degrees.

    Args:
        rad: Angle in radians.

    Returns:
        Angle in degrees.
    """
    return math.degrees(rad)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the range ``[-pi, pi)``.

    Args:
        angle: Angle in radians.

    Returns:
        Wrapped angle.
    """
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def wrap_angles(angles: Sequence[float]) -> np.ndarray:
    """Wrap a sequence of angles to ``[-pi, pi)``.

    Args:
        angles: Angles in radians.

    Returns:
        Array of wrapped angles.
    """
    a = np.asarray(angles, dtype=np.float64)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def deg_array_to_rad(deg_values: Sequence[float]) -> np.ndarray:
    """Convert a sequence of degree values to a radians array.

    Args:
        deg_values: Angles in degrees.

    Returns:
        1-D array of angles in radians.
    """
    return np.radians(np.asarray(deg_values, dtype=np.float64))


def rad_array_to_deg(rad_values: Sequence[float]) -> np.ndarray:
    """Convert a sequence of radian values to a degrees array.

    Args:
        rad_values: Angles in radians.

    Returns:
        1-D array of angles in degrees.
    """
    return np.degrees(np.asarray(rad_values, dtype=np.float64))
