"""Joint-space trajectory interpolation functions.

Provides linear, cubic, and quintic polynomial interpolation between
two joint configurations with smooth start and stop behaviour.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def linear_interpolation(
    q_start: Sequence[float],
    q_end: Sequence[float],
    n_steps: int = 50,
) -> np.ndarray:
    """Linear joint-space interpolation between two configurations.

    Args:
        q_start: Starting joint angles (radians).
        q_end: Goal joint angles (radians).
        n_steps: Number of interpolation steps (inclusive of endpoints).

    Returns:
        ``(n_steps, n_dof)`` array where each row is an interpolated
        configuration.

    Example::

        traj = linear_interpolation([0, 0], [1.0, -0.5], n_steps=20)
    """
    q0 = np.asarray(q_start, dtype=np.float64).ravel()
    q1 = np.asarray(q_end, dtype=np.float64).ravel()

    s = np.linspace(0.0, 1.0, n_steps).reshape(-1, 1)
    trajectory = q0 + s * (q1 - q0)

    logger.debug(
        "Linear interpolation: %d steps, %d DOF", n_steps, q0.size
    )
    return trajectory


def cubic_interpolation(
    q_start: Sequence[float],
    q_end: Sequence[float],
    n_steps: int = 50,
) -> np.ndarray:
    """Cubic polynomial interpolation with zero velocity at endpoints.

    Uses the polynomial ``q(s) = q0 + (3s^2 - 2s^3) * (q1 - q0)``
    where ``s = t / T`` ranges from 0 to 1.  This ensures both position
    continuity and zero velocity at the start and end.

    Args:
        q_start: Starting joint angles (radians).
        q_end: Goal joint angles (radians).
        n_steps: Number of interpolation steps.

    Returns:
        ``(n_steps, n_dof)`` array of interpolated configurations.

    Example::

        traj = cubic_interpolation([0, 0], [1.0, -0.5])
    """
    q0 = np.asarray(q_start, dtype=np.float64).ravel()
    q1 = np.asarray(q_end, dtype=np.float64).ravel()

    s = np.linspace(0.0, 1.0, n_steps).reshape(-1, 1)
    blend = 3.0 * s**2 - 2.0 * s**3
    trajectory = q0 + blend * (q1 - q0)

    logger.debug(
        "Cubic interpolation: %d steps, %d DOF", n_steps, q0.size
    )
    return trajectory


def quintic_interpolation(
    q_start: Sequence[float],
    q_end: Sequence[float],
    n_steps: int = 50,
) -> np.ndarray:
    """Quintic polynomial interpolation with zero velocity and acceleration.

    Uses the polynomial
    ``q(s) = q0 + (10s^3 - 15s^4 + 6s^5) * (q1 - q0)``
    where ``s = t / T`` ranges from 0 to 1.  This ensures position
    continuity and zero first *and* second derivatives at both endpoints.

    Args:
        q_start: Starting joint angles (radians).
        q_end: Goal joint angles (radians).
        n_steps: Number of interpolation steps.

    Returns:
        ``(n_steps, n_dof)`` array of interpolated configurations.

    Example::

        traj = quintic_interpolation([0, 0], [1.0, -0.5])
    """
    q0 = np.asarray(q_start, dtype=np.float64).ravel()
    q1 = np.asarray(q_end, dtype=np.float64).ravel()

    s = np.linspace(0.0, 1.0, n_steps).reshape(-1, 1)
    blend = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    trajectory = q0 + blend * (q1 - q0)

    logger.debug(
        "Quintic interpolation: %d steps, %d DOF", n_steps, q0.size
    )
    return trajectory
