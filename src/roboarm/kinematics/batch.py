"""Batch (vectorised) forward and inverse kinematics.

Exposes :func:`batch_fk` and :func:`batch_ik` for processing many
configurations or targets in a single call — useful for workspace
analysis, machine-learning data generation, and parameter sweeps.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from roboarm.core.robot import RobotArm
from roboarm.core.types import EndEffectorPose, IKSolution

logger = logging.getLogger(__name__)


def batch_fk(
    robot: RobotArm,
    q_array: np.ndarray,
    full: bool = False,
) -> np.ndarray | list[EndEffectorPose]:
    """Compute forward kinematics for many joint configurations at once.

    Args:
        robot: The robot arm model.
        q_array: ``(N, n_dof)`` array of joint configurations in radians.
        full: If ``False`` (default), return an ``(N, 3)`` array of
            end-effector positions only.  If ``True``, return a list of
            :class:`~roboarm.core.types.EndEffectorPose` objects.

    Returns:
        ``(N, 3)`` position array when *full* is ``False``, otherwise a
        list of :class:`~roboarm.core.types.EndEffectorPose` objects.

    Example::

        import numpy as np
        from roboarm.kinematics.batch import batch_fk

        Q = np.random.uniform(-np.pi, np.pi, (1000, robot.n_dof))
        positions = batch_fk(robot, Q)          # (1000, 3)
        poses = batch_fk(robot, Q, full=True)   # list of 1000 EndEffectorPose
    """
    Q = np.asarray(q_array, dtype=np.float64)
    if Q.ndim == 1:
        Q = Q.reshape(1, -1)
    if Q.ndim != 2 or Q.shape[1] != robot.n_dof:
        raise ValueError(
            f"q_array must be (N, {robot.n_dof}), got {Q.shape}"
        )

    n = Q.shape[0]
    logger.debug("batch_fk: %d configurations, %d DOF", n, robot.n_dof)

    if full:
        return [robot.forward_kinematics(Q[i]) for i in range(n)]

    positions = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        pose = robot.forward_kinematics(Q[i])
        positions[i] = pose.position
    return positions


def batch_ik(
    robot: RobotArm,
    targets: Sequence[Sequence[float]],
    solver_name: str = "damped_least_squares",
    q0_list: Sequence[Sequence[float]] | None = None,
    warm_start: bool = True,
) -> list[IKSolution]:
    """Solve inverse kinematics for many target positions at once.

    Args:
        robot: The robot arm model.
        targets: Sequence of target positions, each ``[x, y]`` or
            ``[x, y, z]``.
        solver_name: IK solver registry name.
        q0_list: Optional per-target initial guesses.  When provided, its
            length must match *targets*.
        warm_start: If ``True`` (default), use the previous solution as the
            initial guess for the next target (warm-start chaining).
            Ignored when *q0_list* is supplied.

    Returns:
        List of :class:`~roboarm.core.types.IKSolution` objects, one per
        target (in order).

    Example::

        from roboarm.kinematics.batch import batch_ik

        targets = [(1.0, 0.5), (0.8, 0.6), (1.2, 0.3)]
        results = batch_ik(robot, targets)
        successes = sum(r.success for r in results)
        print(f"{successes}/{len(results)} targets solved")
    """
    import roboarm.kinematics.solvers  # noqa: F401 — triggers registration
    from roboarm.kinematics.solvers.registry import IKSolverRegistry

    solver = IKSolverRegistry.create(solver_name, robot)

    prev_q: list[float] | None = None  # warm-start chain

    results: list[IKSolution] = []
    for idx, target in enumerate(targets):
        pos = np.asarray(target, dtype=np.float64).ravel()
        if pos.size == 2:
            pos = np.append(pos, 0.0)
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = pos
        pose = EndEffectorPose(
            position=pos,
            rotation=np.eye(3, dtype=np.float64),
            transform=T,
        )

        # Determine initial guess: explicit list > warm-start > zeros
        if q0_list is not None:
            if idx >= len(q0_list):
                raise IndexError(
                    f"q0_list has {len(q0_list)} entries but targets has "
                    f"{len(list(targets))} — lengths must match"
                )
            q0 = q0_list[idx]
        elif warm_start and prev_q is not None:
            q0 = prev_q
        else:
            q0 = None

        result = solver.solve(pose, q0=q0)
        results.append(result)

        # Update warm-start: use solution if successful, else best_attempt
        if warm_start:
            if result.success and result.primary is not None:
                prev_q = result.primary.values.tolist()
            elif result.best_attempt is not None:
                prev_q = result.best_attempt.values.tolist()

    n = len(results)
    successes = sum(r.success for r in results)
    logger.info(
        "batch_ik: %d/%d targets solved (solver=%s)",
        successes, n, solver_name,
    )
    return results
