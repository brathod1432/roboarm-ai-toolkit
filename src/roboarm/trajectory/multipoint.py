"""Multi-waypoint trajectory planning.

Extends the two-point interpolators in :mod:`~roboarm.trajectory.interpolation`
to handle an arbitrary sequence of joint-space waypoints, producing a single
smooth trajectory array with C1 continuity at interior junctions (velocity
matches across segment boundaries).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def via_point_trajectory(
    waypoints: Sequence[Sequence[float]],
    n_steps_per_segment: int = 50,
    method: str = "cubic",
    blend_velocities: bool = True,
) -> np.ndarray:
    """Generate a smooth multi-segment joint-space trajectory.

    Connects *N* joint-space waypoints through piecewise polynomial
    segments.  When *blend_velocities* is ``True``, interior junction
    velocities are set to the average of the incoming and outgoing
    segment slopes, guaranteeing C1 (velocity) continuity.

    Args:
        waypoints: Sequence of *N* joint configurations (each a list or
            array of joint angles in radians).  Must have at least 2
            elements.
        n_steps_per_segment: Number of samples per segment including the
            start of each segment (the final waypoint is included as the
            last row).
        method: Interpolation method — ``"linear"``, ``"cubic"``, or
            ``"quintic"``.
        blend_velocities: If ``True`` (default), compute junction
            velocities as the average of adjacent segment slopes and use
            them to anchor a cubic Hermite spline at each interior point.
            If ``False``, each segment is interpolated independently
            (may have velocity discontinuities at junctions).

    Returns:
        ``(total_steps, n_dof)`` trajectory array where
        ``total_steps = (N - 1) * n_steps_per_segment + 1``.

    Raises:
        ValueError: If fewer than 2 waypoints are provided, or if
            waypoints have inconsistent DOF.

    Example::

        from roboarm.trajectory.multipoint import via_point_trajectory

        waypoints = [
            [0.0, 0.0],
            [1.0, -0.5],
            [0.5,  0.8],
            [0.0,  0.0],
        ]
        traj = via_point_trajectory(waypoints, n_steps_per_segment=50)
        # traj.shape == (151, 2)  for 3 segments × 50 + 1
    """
    pts = [np.asarray(wp, dtype=np.float64).ravel() for wp in waypoints]
    if len(pts) < 2:
        raise ValueError(
            f"At least 2 waypoints required, got {len(pts)}"
        )
    n_dof = pts[0].size
    for i, p in enumerate(pts):
        if p.size != n_dof:
            raise ValueError(
                f"Waypoint {i} has {p.size} DOF, expected {n_dof}"
            )

    n_seg = len(pts) - 1

    if method == "linear" or not blend_velocities:
        return _independent_segments(pts, n_steps_per_segment, method)

    # Compute junction velocities (average slope of adjacent segments)
    velocities = _blend_velocities(pts)

    # Build each segment with the pre-computed endpoint velocities
    segments: list[np.ndarray] = []
    for i in range(n_seg):
        seg = _cubic_hermite_segment(
            q0=pts[i],
            q1=pts[i + 1],
            v0=velocities[i],
            v1=velocities[i + 1],
            n_steps=n_steps_per_segment + 1,  # +1 so dedup yields n_steps per segment
        )
        # Exclude the last point of each segment except the final one
        # to avoid duplication at junctions.
        if i < n_seg - 1:
            segments.append(seg[:-1])
        else:
            segments.append(seg)

    traj = np.vstack(segments)
    logger.info(
        "Via-point trajectory: %d waypoints, %d segments, "
        "%d total steps, method=%s",
        len(pts), n_seg, len(traj), method,
    )
    return traj


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _blend_velocities(pts: list[np.ndarray]) -> list[np.ndarray]:
    """Compute junction velocities using the average-slope formula.

    End velocities are zero (arm starts and stops at rest).
    Interior velocities are the mean of the slopes of the two
    adjacent segments, scaled so sign discontinuities (direction
    reversals) produce zero velocity to avoid overshoot.
    """
    n = len(pts)
    n_dof = pts[0].size
    velocities: list[np.ndarray] = [np.zeros(n_dof)]

    for i in range(1, n - 1):
        slope_in  = pts[i] - pts[i - 1]
        slope_out = pts[i + 1] - pts[i]
        # At direction reversals, clamp to zero to prevent overshoot
        vel = np.where(
            np.sign(slope_in) == np.sign(slope_out),
            0.5 * (slope_in + slope_out),
            np.zeros(n_dof),
        )
        velocities.append(vel)

    velocities.append(np.zeros(n_dof))
    return velocities


def _cubic_hermite_segment(
    q0: np.ndarray,
    q1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    n_steps: int,
) -> np.ndarray:
    """Cubic Hermite spline between two configurations.

    Parameterised by s ∈ [0, 1]:
        q(s) = (2s³-3s²+1)·q0 + (s³-2s²+s)·v0
             + (-2s³+3s²)·q1 + (s³-s²)·v1
    """
    s = np.linspace(0.0, 1.0, n_steps).reshape(-1, 1)
    h00 =  2*s**3 - 3*s**2 + 1
    h10 =    s**3 - 2*s**2 + s
    h01 = -2*s**3 + 3*s**2
    h11 =    s**3 -   s**2
    return h00 * q0 + h10 * v0 + h01 * q1 + h11 * v1


def _independent_segments(
    pts: list[np.ndarray],
    n_steps: int,
    method: str,
) -> np.ndarray:
    """Fallback: interpolate each segment independently without velocity blending."""
    from roboarm.trajectory.interpolation import (
        cubic_interpolation,
        linear_interpolation,
        quintic_interpolation,
    )

    interp_fn = {
        "linear": linear_interpolation,
        "cubic": cubic_interpolation,
        "quintic": quintic_interpolation,
    }.get(method)
    if interp_fn is None:
        raise ValueError(
            f"Unknown method {method!r}. Choose 'linear', 'cubic', or 'quintic'."
        )

    segments: list[np.ndarray] = []
    for i in range(len(pts) - 1):
        seg = interp_fn(pts[i], pts[i + 1], n_steps=n_steps + 1)  # +1 for dedup
        if i < len(pts) - 2:
            segments.append(seg[:-1])
        else:
            segments.append(seg)
    return np.vstack(segments)
