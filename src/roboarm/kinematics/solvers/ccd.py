"""Cyclic Coordinate Descent (CCD) iterative IK solver.

Optimises one joint at a time from the tip back to the base, rotating
each joint to minimise the distance between the end-effector and the
target.  Registered as ``"ccd"``.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence

import numpy as np

from roboarm.core.robot import RobotArm
from roboarm.core.types import EndEffectorPose, IKSolution, JointSolution
from roboarm.kinematics.inverse import IKConfig, IKSolverBase
from roboarm.kinematics.solvers.registry import IKSolverRegistry
from roboarm.utils.log_event import log_event

logger = logging.getLogger(__name__)


@IKSolverRegistry.register("ccd")
class CCDSolver(IKSolverBase):
    """Cyclic Coordinate Descent inverse kinematics solver.

    CCD sweeps through the joints from the tip to the base.  For each
    joint it computes the angle that rotates the vector from the joint
    origin to the current end-effector toward the vector from the joint
    origin to the target, and applies that rotation.

    This is a simple, robust heuristic that works well for serial chains
    of any length, though it may converge slowly for long chains.

    Args:
        robot: The robot arm model.
        config: Optional :class:`IKConfig` with solver parameters.
        max_iterations: Override for ``config.max_iterations``.
        tolerance: Override for ``config.tolerance``.
    """

    def __init__(
        self,
        robot: RobotArm,
        *,
        config: IKConfig | None = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        **kwargs: object,
    ) -> None:
        super().__init__(robot, **kwargs)
        cfg = config or IKConfig()
        self._max_iter = max_iterations if config is None else cfg.max_iterations
        self._tol = tolerance if config is None else cfg.tolerance
        self._planar = robot.is_planar

        # Build map from variable-joint index to overall joint index
        self._var_indices: list[int] = [
            i for i, jc in enumerate(robot.joints) if jc.is_variable
        ]

    def solve(
        self,
        target: EndEffectorPose,
        q0: Sequence[float] | None = None,
    ) -> IKSolution:
        """Run CCD iterations to solve inverse kinematics.

        Args:
            target: Desired end-effector pose.
            q0: Initial joint-angle guess (zeros if ``None``).

        Returns:
            :class:`IKSolution` with convergence information.
        """
        t_start = time.perf_counter()
        n_dof = self._robot.n_dof
        messages: list[str] = []

        q = np.zeros(n_dof, dtype=np.float64)
        if q0 is not None:
            q = np.asarray(q0, dtype=np.float64).ravel().copy()

        target_pos = target.position[:2] if self._planar else target.position[:3]

        error_norm = float("inf")
        iteration = 0
        best_error = float("inf")
        best_q = q.copy()

        for iteration in range(1, self._max_iter + 1):
            # Sweep joints from tip to base
            for vi in reversed(range(n_dof)):
                positions = self._robot.joint_positions(q)
                ee_pos = positions[-1][:2] if self._planar else positions[-1][:3]

                # Joint origin for this variable joint
                overall_idx = self._var_indices[vi]
                joint_pos = (
                    positions[overall_idx][:2]
                    if self._planar
                    else positions[overall_idx][:3]
                )

                # Vectors from joint to end-effector and to target
                vec_ee = ee_pos - joint_pos
                vec_target = target_pos - joint_pos

                if self._planar:
                    # 2-D: compute angle between the two vectors
                    angle_ee = math.atan2(vec_ee[1], vec_ee[0])
                    angle_target = math.atan2(vec_target[1], vec_target[0])
                    delta = angle_target - angle_ee
                    # Wrap to [-pi, pi]
                    delta = math.atan2(math.sin(delta), math.cos(delta))
                else:
                    # 3-D: rotation axis is z-axis of the joint frame
                    # For revolute joints rotating around their local z,
                    # project onto the joint's local XY plane.
                    cumulative = self._robot.joint_transforms(q)
                    T_joint = cumulative[overall_idx]
                    R = T_joint[:3, :3]

                    # Transform vectors into joint-local frame
                    vec_ee_local = R.T @ vec_ee
                    vec_tgt_local = R.T @ vec_target

                    angle_ee = math.atan2(vec_ee_local[1], vec_ee_local[0])
                    angle_tgt = math.atan2(vec_tgt_local[1], vec_tgt_local[0])
                    delta = angle_tgt - angle_ee
                    delta = math.atan2(math.sin(delta), math.cos(delta))

                q[vi] += delta

            # Evaluate error after full sweep
            fk = self._robot.forward_kinematics(q)
            current_pos = fk.position[:2] if self._planar else fk.position[:3]
            error_norm = float(np.linalg.norm(target_pos - current_pos))

            if error_norm < best_error:
                best_error = error_norm
                best_q = q.copy()

            if error_norm < self._tol:
                logger.debug(
                    "CCD converged at iteration %d (error=%.2e)",
                    iteration, error_norm,
                )
                break

        elapsed = (time.perf_counter() - t_start) * 1000.0
        success = error_norm < self._tol

        if not success:
            messages.append(
                f"CCD did not converge after {iteration} iterations "
                f"(error={error_norm:.2e})"
            )
            log_event(logger, logging.WARNING, "ik_solve",
                      solver=self.name,
                      success=False,
                      iterations=iteration,
                      error=round(error_norm, 8),
                      duration_ms=round(elapsed, 3))
        else:
            messages.append(
                f"CCD converged in {iteration} iterations "
                f"(error={error_norm:.2e})"
            )
            log_event(logger, logging.DEBUG, "ik_solve",
                      solver=self.name,
                      success=True,
                      iterations=iteration,
                      error=round(error_norm, 8),
                      duration_ms=round(elapsed, 3))

        return IKSolution(
            success=success,
            primary=JointSolution(values=q) if success else None,
            best_attempt=JointSolution(values=best_q),
            iterations=iteration,
            residual_error=error_norm,
            computation_time_ms=elapsed,
            solver_name=self.name,
            messages=messages,
        )
