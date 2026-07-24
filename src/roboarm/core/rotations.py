"""SO(3) rotation utilities.

Conversions between rotation matrices, Euler angles (ZYX), axis-angle,
and quaternions.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def rotx(angle: float) -> np.ndarray:
    """Rotation matrix about the X axis.

    Args:
        angle: Rotation angle (radians).

    Returns:
        3x3 rotation matrix.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, c,   -s],
                     [0.0, s,    c]], dtype=np.float64)


def roty(angle: float) -> np.ndarray:
    """Rotation matrix about the Y axis.

    Args:
        angle: Rotation angle (radians).

    Returns:
        3x3 rotation matrix.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[ c,  0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s,  0.0, c]], dtype=np.float64)


def rotz(angle: float) -> np.ndarray:
    """Rotation matrix about the Z axis.

    Args:
        angle: Rotation angle (radians).

    Returns:
        3x3 rotation matrix.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c,  -s,  0.0],
                     [s,   c,  0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def euler_to_rotation(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert ZYX Euler angles to a 3x3 rotation matrix.

    Rotation order: ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    Args:
        roll: Rotation about X (radians).
        pitch: Rotation about Y (radians).
        yaw: Rotation about Z (radians).

    Returns:
        3x3 rotation matrix.
    """
    return rotz(yaw) @ roty(pitch) @ rotx(roll)


def rotation_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Extract ZYX Euler angles from a 3x3 rotation matrix.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Tuple ``(roll, pitch, yaw)`` in radians.
    """
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return roll, pitch, yaw


def axis_angle_to_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert axis-angle representation to a 3x3 rotation matrix (Rodrigues).

    Args:
        axis: 3-element unit vector.
        angle: Rotation magnitude (radians).

    Returns:
        3x3 rotation matrix.
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def rotation_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """Extract axis-angle from a 3x3 rotation matrix.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        Tuple ``(axis, angle)`` where axis is a unit vector and angle in radians.
    """
    angle = math.acos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if abs(angle) < 1e-10:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if abs(angle - math.pi) < 1e-10:
        col = np.argmax(np.diag(R))
        axis = R[:, col] + np.eye(3)[col]
        axis = axis / np.linalg.norm(axis)
        return axis, angle
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * math.sin(angle))
    return axis, angle


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert a unit quaternion ``[w, x, y, z]`` to a 3x3 rotation matrix.

    Args:
        q: 4-element quaternion array ``[w, x, y, z]``.

    Returns:
        3x3 rotation matrix.
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion ``[w, x, y, z]``.

    Uses Shepperd's method for numerical stability.

    Args:
        R: 3x3 rotation matrix.

    Returns:
        4-element quaternion array ``[w, x, y, z]``.
    """
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def is_valid_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    """Check whether a 3x3 matrix is a valid rotation (SO(3)).

    Args:
        R: Matrix to check.
        tol: Numerical tolerance.

    Returns:
        ``True`` if ``R @ R^T == I`` and ``det(R) == 1``.
    """
    if R.shape != (3, 3):
        return False
    if not np.allclose(R @ R.T, np.eye(3), atol=tol):
        return False
    return abs(np.linalg.det(R) - 1.0) <= tol
