"""Linear Segment with Parabolic Blend (LSPB) trajectory generator.

Implements the classic trapezoidal-velocity profile with three phases:
acceleration (parabolic), constant velocity (linear), and deceleration
(parabolic).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def lspb(
    q0: float,
    qf: float,
    t_total: float,
    v_max: Optional[float] = None,
    n_steps: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LSPB trajectory for a single joint.

    Generates a trapezoidal velocity profile with parabolic blends at
    the start and end, and a constant-velocity linear segment in the
    middle.

    Args:
        q0: Starting joint angle (radians).
        qf: Final joint angle (radians).
        t_total: Total trajectory duration (seconds).
        v_max: Maximum cruise velocity.  If ``None``, defaults to
            ``1.5 * (qf - q0) / t_total``.
        n_steps: Number of time samples.

    Returns:
        Tuple of ``(positions, velocities, times)`` each as 1-D arrays
        of length *n_steps*.

    Raises:
        ValueError: If the requested *v_max* is infeasible.

    Example::

        pos, vel, t = lspb(0.0, 1.5, t_total=2.0)
    """
    delta_q = qf - q0

    # Handle zero-displacement edge case
    if abs(delta_q) < 1e-12:
        times = np.linspace(0.0, t_total, n_steps)
        positions = np.full(n_steps, q0)
        velocities = np.zeros(n_steps)
        return positions, velocities, times

    if v_max is None:
        v_max = 1.5 * delta_q / t_total

    # Ensure sign consistency: v_max should match direction of motion
    if delta_q < 0 and v_max > 0:
        v_max = -v_max
    elif delta_q > 0 and v_max < 0:
        v_max = -v_max

    # Blend time (parabolic phase duration)
    t_blend = t_total - delta_q / v_max

    # Feasibility check: blend time must be positive and less than half
    # of the total time (otherwise there is no linear segment).
    if t_blend <= 0 or t_blend >= t_total:
        raise ValueError(
            f"Infeasible LSPB parameters: t_blend={t_blend:.4f}, "
            f"t_total={t_total:.4f}, v_max={v_max:.4f}"
        )

    # Acceleration during parabolic phase
    accel = v_max / t_blend

    times = np.linspace(0.0, t_total, n_steps)
    positions = np.empty(n_steps, dtype=np.float64)
    velocities = np.empty(n_steps, dtype=np.float64)

    for i, t in enumerate(times):
        if t <= t_blend:
            # Phase 1 — parabolic acceleration
            positions[i] = q0 + 0.5 * accel * t**2
            velocities[i] = accel * t
        elif t <= t_total - t_blend:
            # Phase 2 — constant velocity (linear segment)
            positions[i] = q0 + 0.5 * accel * t_blend**2 + v_max * (t - t_blend)
            velocities[i] = v_max
        else:
            # Phase 3 — parabolic deceleration
            t_rem = t_total - t
            positions[i] = qf - 0.5 * accel * t_rem**2
            velocities[i] = accel * t_rem

    logger.debug(
        "LSPB: q0=%.3f qf=%.3f t_total=%.2f v_max=%.3f t_blend=%.3f",
        q0, qf, t_total, v_max, t_blend,
    )
    return positions, velocities, times


def multi_joint_lspb(
    q_start: Sequence[float],
    q_end: Sequence[float],
    t_total: float,
    v_max: Optional[Sequence[float]] = None,
    n_steps: int = 100,
) -> np.ndarray:
    """LSPB trajectory for multiple joints simultaneously.

    Each joint follows its own trapezoidal velocity profile over the
    same total duration.

    Args:
        q_start: Starting joint angles (radians).
        q_end: Goal joint angles (radians).
        t_total: Total trajectory duration (seconds).
        v_max: Per-joint maximum velocities.  ``None`` for automatic.
        n_steps: Number of time samples.

    Returns:
        ``(n_steps, n_dof)`` array of joint positions.

    Example::

        traj = multi_joint_lspb([0, 0], [1.5, -0.5], t_total=2.0)
    """
    q0_arr = np.asarray(q_start, dtype=np.float64).ravel()
    qf_arr = np.asarray(q_end, dtype=np.float64).ravel()
    n_dof = q0_arr.size

    trajectory = np.empty((n_steps, n_dof), dtype=np.float64)

    for j in range(n_dof):
        vm = None if v_max is None else float(v_max[j])
        pos, _vel, _t = lspb(
            q0=float(q0_arr[j]),
            qf=float(qf_arr[j]),
            t_total=t_total,
            v_max=vm,
            n_steps=n_steps,
        )
        trajectory[:, j] = pos

    logger.debug(
        "Multi-joint LSPB: %d DOF, %d steps, t_total=%.2f",
        n_dof, n_steps, t_total,
    )
    return trajectory
