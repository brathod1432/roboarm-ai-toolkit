"""Damped Least Squares (Levenberg-Marquardt) iterative IK solver.

Computes joint updates via
``delta_q = J^T (J J^T + lambda^2 I)^{-1} @ error``
which avoids the numerical instability of the plain pseudo-inverse near
singularities.  Registered as ``"damped_least_squares"``.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence

import numpy as np

from roboarm.core.robot import RobotArm
from roboarm.core.types import EndEffectorPose, IKSolution, JointSolution
from roboarm.kinematics.inverse import IKConfig, IKSolverBase
from roboarm.kinematics.jacobian import JacobianComputer
from roboarm.kinematics.solvers.registry import IKSolverRegistry

logger = logging.getLogger(__name__)


@IKSolverRegistry.register("damped_least_squares")
class DampedLeastSquaresSolver(IKSolverBase):
    """Iterative IK solver using Damped Least Squares (DLS).

    DLS adds a damping term ``lambda^2 I`` to the Jacobian product to
    improve numerical conditioning near singularities at the cost of
    slightly slower convergence away from them.

    Args:
        robot: The robot arm model.
        config: Optional :class:`IKConfig` with solver parameters.
        max_iterations: Override for ``config.max_iterations``.
        tolerance: Override for ``config.tolerance``.
        damping: Override for ``config.damping``.
        step_size: Override for ``config.step_size``.
    """

    def __init__(
        self,
        robot: RobotArm,
        *,
        config: Optional[IKConfig] = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        damping: float = 0.01,
        step_size: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(robot, **kwargs)
        cfg = config or IKConfig()
        self._max_iter = max_iterations if config is None else cfg.max_iterations
        self._tol = tolerance if config is None else cfg.tolerance
        self._damping = damping if config is None else cfg.damping
        self._step = step_size if config is None else cfg.step_size
        self._jac = JacobianComputer(robot)

    def solve(
        self,
        target: EndEffectorPose,
        q0: Optional[Sequence[float]] = None,
    ) -> IKSolution:
        """Solve IK iteratively using Damped Least Squares.

        Args:
            target: Desired end-effector pose.
            q0: Initial joint-angle guess (zeros if ``None``).

        Returns:
            :class:`IKSolution` with convergence information.
        """
        t_start = time.perf_counter()
        n_dof = self._robot.n_dof
        is_planar = self._jac.is_planar
        messages: List[str] = []

        q = np.zeros(n_dof, dtype=np.float64)
        if q0 is not None:
            q = np.asarray(q0, dtype=np.float64).ravel().copy()

        target_pos = target.position[:2] if is_planar else target.position[:3]

        error_norm = float("inf")
        iteration = 0
        lam_sq = self._damping * self._damping

        for iteration in range(1, self._max_iter + 1):
            fk = self._robot.forward_kinematics(q)
            current_pos = fk.position[:2] if is_planar else fk.position[:3]
            error = target_pos - current_pos
            error_norm = float(np.linalg.norm(error))

            if error_norm < self._tol:
                logger.debug(
                    "DLS converged at iteration %d (error=%.2e)",
                    iteration, error_norm,
                )
                break

            J = self._jac.compute(q)
            task_dim = J.shape[0]
            # delta_q = J^T (J J^T + lambda^2 I)^{-1} error
            JJT = J @ J.T + lam_sq * np.eye(task_dim, dtype=np.float64)
            delta_q = J.T @ np.linalg.solve(JJT, error)
            q = q + self._step * delta_q

        elapsed = (time.perf_counter() - t_start) * 1000.0
        success = error_norm < self._tol

        if not success:
            messages.append(
                f"DLS did not converge after {iteration} iterations "
                f"(error={error_norm:.2e})"
            )
            logger.warning(messages[-1])
        else:
            messages.append(
                f"DLS converged in {iteration} iterations "
                f"(error={error_norm:.2e})"
            )

        return IKSolution(
            success=success,
            primary=JointSolution(values=q) if success else None,
            iterations=iteration,
            residual_error=error_norm,
            computation_time_ms=elapsed,
            solver_name=self.name,
            messages=messages,
        )
