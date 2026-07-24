"""Homogeneous transformation utilities for serial-link robots.

Provides functions to build 4x4 SE(3) transforms from DH and Modified DH
parameters, plus helpers for composition and inversion.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from roboarm.core.types import DHParams

logger = logging.getLogger(__name__)


def dh_transform(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    """Build a 4x4 homogeneous transform using *standard* DH convention.

    Order: ``Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)``

    Args:
        alpha: Link twist (radians).
        a: Link length.
        theta: Joint angle (radians).
        d: Link offset.

    Returns:
        4x4 homogeneous transformation matrix.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d],
        [0.0, 0.0,      0.0,     1.0],
    ], dtype=np.float64)


def mdh_transform(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    """Build a 4x4 homogeneous transform using *modified* DH (Craig) convention.

    Order: ``Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)``

    Args:
        alpha: Twist angle alpha_{i-1} (radians).
        a: Link length a_{i-1}.
        theta: Joint angle theta_i (radians).
        d: Link offset d_i.

    Returns:
        4x4 homogeneous transformation matrix.
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct,          -st,          0.0,  a],
        [st * ca,      ct * ca,    -sa,  -d * sa],
        [st * sa,      ct * sa,     ca,   d * ca],
        [0.0,          0.0,         0.0,  1.0],
    ], dtype=np.float64)


def transform_from_dh_params(params: DHParams, q: float = 0.0) -> np.ndarray:
    """Build a transform from a :class:`DHParams` object.

    For revolute joints the variable ``q`` is added to ``theta``.

    Args:
        params: DH parameter set for one link.
        q: Joint variable (added to theta for revolute joints).

    Returns:
        4x4 homogeneous transformation matrix.
    """
    theta = params.theta + q
    if params.convention == "modified":
        return mdh_transform(params.alpha, params.a, theta, params.d)
    return dh_transform(params.alpha, params.a, theta, params.d)


def chain_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Multiply a sequence of 4x4 transforms left-to-right.

    Args:
        transforms: Ordered list of 4x4 matrices ``[T_01, T_12, ...]``.

    Returns:
        Cumulative 4x4 transform ``T_0n``.
    """
    result = np.eye(4, dtype=np.float64)
    for t in transforms:
        result = result @ t
    return result


def inverse_transform(T: np.ndarray) -> np.ndarray:
    """Compute the inverse of a 4x4 homogeneous transform efficiently.

    Uses the property ``R^T`` and ``-R^T * p`` instead of a general inverse.

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        4x4 inverse transform.
    """
    R = T[:3, :3]
    p = T[:3, 3]
    Rt = R.T
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = Rt
    inv[:3, 3] = -Rt @ p
    return inv


def extract_position(T: np.ndarray) -> np.ndarray:
    """Extract the translation vector from a 4x4 transform.

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        3-element position array ``[x, y, z]``.
    """
    return T[:3, 3].copy()


def extract_rotation(T: np.ndarray) -> np.ndarray:
    """Extract the 3x3 rotation matrix from a 4x4 transform.

    Args:
        T: 4x4 homogeneous transformation matrix.

    Returns:
        3x3 rotation matrix.
    """
    return T[:3, :3].copy()


def is_valid_transform(T: np.ndarray, tol: float = 1e-6) -> bool:
    """Check whether a 4x4 matrix is a valid SE(3) transform.

    Validates orthonormality of the rotation block and the ``[0,0,0,1]``
    bottom row.

    Args:
        T: Matrix to check.
        tol: Numerical tolerance.

    Returns:
        ``True`` if valid.
    """
    if T.shape != (4, 4):
        return False
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=tol):
        return False
    if abs(np.linalg.det(R) - 1.0) > tol:
        return False
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=tol):
        return False
    return True
