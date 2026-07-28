"""Forward And Backward Reaching Inverse Kinematics (FABRIK) solver.

FABRIK operates directly on joint positions rather than angles, using
two alternating phases per iteration:

1. **Forward reach** -- move the end-effector to the target and adjust
   each joint toward the base while preserving link lengths.
2. **Backward reach** -- pin the base and adjust each joint toward the
   tip while preserving link lengths.

Joint angles are recovered from the final positions via ``atan2``.
Registered as ``"fabrik"``.
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


@IKSolverRegistry.register("fabrik")
class FABRIKSolver(IKSolverBase):
    """FABRIK inverse kinematics solver.

    FABRIK is a fast, iterative, heuristic solver that manipulates joint
    positions directly.  It is especially effective for long serial
    chains and converges quickly in practice.

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

    def solve(
        self,
        target: EndEffectorPose,
        q0: Sequence[float] | None = None,
    ) -> IKSolution:
        """Run FABRIK iterations to solve inverse kinematics.

        Args:
            target: Desired end-effector pose.
            q0: Initial joint-angle guess (zeros if ``None``).

        Returns:
            :class:`IKSolution` with convergence information.
        """
        t_start = time.perf_counter()
        n_dof = self._robot.n_dof
        messages: list[str] = []
        dim = 2 if self._planar else 3

        # Initialise joint angles
        q = np.zeros(n_dof, dtype=np.float64)
        if q0 is not None:
            q = np.asarray(q0, dtype=np.float64).ravel().copy()

        target_pos = target.position[:dim].copy()

        # Get initial joint positions and derive link lengths
        all_positions = self._robot.joint_positions(q)  # (n_joints+1, 3)
        points = all_positions[:, :dim].copy()  # (n_points, dim)
        n_points = points.shape[0]

        link_lengths = np.array([
            float(np.linalg.norm(points[i + 1] - points[i]))
            for i in range(n_points - 1)
        ], dtype=np.float64)

        total_reach = float(np.sum(link_lengths))
        target_dist = float(np.linalg.norm(target_pos - points[0]))

        # Quick reachability check
        if target_dist > total_reach + 1e-10:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            messages.append(
                f"Target is unreachable (distance={target_dist:.4f}, "
                f"reach={total_reach:.4f})"
            )
            log_event(logger, logging.WARNING, "ik_unreachable",
                      solver=self.name,
                      target_dist=round(target_dist, 4),
                      reach=round(total_reach, 4))
            return IKSolution(
                success=False,
                iterations=0,
                residual_error=target_dist - total_reach,
                computation_time_ms=elapsed,
                solver_name=self.name,
                messages=messages,
            )

        base = points[0].copy()
        best_error = float("inf")
        best_points = points.copy()
        error_norm = float("inf")
        iteration = 0

        for iteration in range(1, self._max_iter + 1):
            # --- Forward reaching phase ---
            # Move end-effector to target, then adjust backward
            points[-1] = target_pos.copy()
            for i in range(n_points - 2, -1, -1):
                direction = points[i] - points[i + 1]
                dist = float(np.linalg.norm(direction))
                if dist < 1e-12:
                    # Degenerate case: nudge slightly
                    direction = np.zeros(dim, dtype=np.float64)
                    direction[0] = 1e-6
                    dist = 1e-6
                points[i] = points[i + 1] + (direction / dist) * link_lengths[i]

            # --- Backward reaching phase ---
            # Pin the base and adjust forward
            points[0] = base.copy()
            for i in range(n_points - 1):
                direction = points[i + 1] - points[i]
                dist = float(np.linalg.norm(direction))
                if dist < 1e-12:
                    direction = np.zeros(dim, dtype=np.float64)
                    direction[0] = 1e-6
                    dist = 1e-6
                points[i + 1] = points[i] + (direction / dist) * link_lengths[i]

            # Check convergence
            error_norm = float(np.linalg.norm(points[-1] - target_pos))
            if error_norm < best_error:
                best_error = error_norm
                best_points = points.copy()
            if error_norm < self._tol:
                logger.debug(
                    "FABRIK converged at iteration %d (error=%.2e)",
                    iteration, error_norm,
                )
                break

        # --- Recover joint angles from positions ---
        q = self._positions_to_angles(points, n_dof)
        best_q = self._positions_to_angles(best_points, n_dof)

        # Verify with FK
        fk = self._robot.forward_kinematics(q)
        current_pos = fk.position[:dim]
        error_norm = float(np.linalg.norm(target_pos - current_pos))

        elapsed = (time.perf_counter() - t_start) * 1000.0
        success = error_norm < self._tol

        if not success:
            messages.append(
                f"FABRIK did not converge after {iteration} iterations "
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
                f"FABRIK converged in {iteration} iterations "
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _positions_to_angles(
        self,
        points: np.ndarray,
        n_dof: int,
    ) -> np.ndarray:
        """Recover joint angles from FABRIK joint positions.

        For a planar chain, consecutive link directions yield
        cumulative angles; individual joint angles are the differences.

        Args:
            points: ``(n_points, dim)`` array of joint positions.
            n_dof: Number of variable degrees of freedom.

        Returns:
            1-D array of joint angles in radians.
        """
        joints = self._robot.joints
        q = np.zeros(n_dof, dtype=np.float64)

        cumulative_angle = 0.0
        var_idx = 0
        for i, jc in enumerate(joints):
            if not jc.is_variable:
                continue
            direction = points[i + 1] - points[i]
            link_angle = math.atan2(direction[1], direction[0])
            q[var_idx] = link_angle - cumulative_angle - jc.dh_params.theta
            cumulative_angle += q[var_idx] + jc.dh_params.theta
            var_idx += 1

        return q
