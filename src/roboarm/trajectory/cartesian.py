"""Cartesian-space trajectory planning.

Generates a straight-line end-effector path by interpolating between two
poses in Cartesian space and solving IK at each waypoint.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from roboarm.core.robot import RobotArm
from roboarm.core.types import EndEffectorPose, IKSolution

logger = logging.getLogger(__name__)


def cartesian_trajectory(
    robot: RobotArm,
    start_pose: EndEffectorPose,
    end_pose: EndEffectorPose,
    n_steps: int = 50,
    solver_name: str = "damped_least_squares",
    q0: Sequence[float] | None = None,
) -> tuple[np.ndarray, list[IKSolution]]:
    """Generate a straight-line Cartesian trajectory.

    Linearly interpolates between *start_pose* and *end_pose* in position
    space, then solves inverse kinematics at each waypoint.  The joint
    solution from each step is used as the initial guess for the next
    (warm-start chaining), which significantly improves convergence speed
    and smoothness for short-step trajectories.

    Args:
        robot: The robot arm model.
        start_pose: Starting end-effector pose.
        end_pose: Goal end-effector pose.
        n_steps: Number of waypoints including the endpoints.
        solver_name: Registry name of the IK solver to use.  Defaults to
            ``"damped_least_squares"`` which is the most robust near
            singularities.
        q0: Optional initial joint-angle guess for the first step.
            Subsequent steps use the previous solution as a warm start.

    Returns:
        A tuple ``(trajectory, ik_results)`` where:

        * ``trajectory`` is an ``(n_steps, n_dof)`` float array of joint
          angles.  Rows where IK failed contain the best attempt (if any)
          or zeros.
        * ``ik_results`` is a list of :class:`IKSolution` objects, one per
          waypoint, so callers can inspect convergence details.

    Raises:
        ImportError: If the solver package cannot be imported.

    Example::

        from roboarm.robots.two_link_planar import create_two_link_planar
        from roboarm.trajectory.cartesian import cartesian_trajectory

        robot = create_two_link_planar()
        start_q = [0.0, 0.0]
        end_q = [1.0, -0.5]
        start_pose = robot.forward_kinematics(start_q)
        end_pose = robot.forward_kinematics(end_q)

        traj, results = cartesian_trajectory(robot, start_pose, end_pose, n_steps=20)
        success_count = sum(r.success for r in results)
        print(f"{success_count}/{len(results)} waypoints solved")
    """
    import roboarm.kinematics.solvers  # ensure all solvers are registered  # noqa: F401
    from roboarm.kinematics.solvers.registry import IKSolverRegistry

    solver = IKSolverRegistry.create(solver_name, robot)

    p_start = start_pose.position.copy()
    p_end = end_pose.position.copy()

    n_dof = robot.n_dof
    trajectory = np.zeros((n_steps, n_dof), dtype=np.float64)
    ik_results: list[IKSolution] = []

    # Use caller-supplied q0 for the first step; warm-start thereafter
    current_q0: Sequence[float] | None = q0

    failed_count = 0

    for i in range(n_steps):
        s = i / max(n_steps - 1, 1)
        position = p_start + s * (p_end - p_start)

        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = position
        waypoint_pose = EndEffectorPose(
            position=position,
            rotation=np.eye(3, dtype=np.float64),
            transform=transform,
        )

        result = solver.solve(waypoint_pose, q0=current_q0)
        ik_results.append(result)

        if result.success and result.primary is not None:
            q = result.primary.values
            trajectory[i] = q
            current_q0 = q.tolist()  # warm start for next step
        elif result.best_attempt is not None:
            # Use best attempt for continuity even on failure
            q = result.best_attempt.values
            trajectory[i] = q
            current_q0 = q.tolist()
            failed_count += 1
            logger.warning(
                "Cartesian trajectory step %d/%d failed (error=%.4e); "
                "using best attempt",
                i + 1, n_steps, result.residual_error,
            )
        else:
            # No attempt available — keep zeros and do not warm-start
            failed_count += 1
            logger.warning(
                "Cartesian trajectory step %d/%d failed with no attempt",
                i + 1, n_steps,
            )

    if failed_count == 0:
        logger.info(
            "Cartesian trajectory: all %d steps solved (solver=%s)",
            n_steps, solver_name,
        )
    else:
        logger.warning(
            "Cartesian trajectory: %d/%d steps failed (solver=%s)",
            failed_count, n_steps, solver_name,
        )

    return trajectory, ik_results
