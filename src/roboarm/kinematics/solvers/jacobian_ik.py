"""Jacobian pseudo-inverse iterative IK solver.

Computes joint updates via ``delta_q = J_pinv @ error`` at each
iteration.  Registered as ``"jacobian_pseudoinverse"``.
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


@IKSolverRegistry.register("jacobian_pseudoinverse")
class JacobianPseudoinverseSolver(IKSolverBase):
    """Iterative IK solver using the Moore-Penrose pseudo-inverse.

    At each iteration the solver computes the position error between the
    current FK result and the target, builds the geometric Jacobian, and
    applies the update ``delta_q = pinv(J) @ error``.

    Args:
        robot: The robot arm model.
        config: Optional :class:`IKConfig` with solver parameters.
        max_iterations: Override for ``config.max_iterations``.
        tolerance: Override for ``config.tolerance``.
        step_size: Override for ``config.step_size``.
    """

    def __init__(
        self,
        robot: RobotArm,
        *,
        config: Optional[IKConfig] = None,
        max_iterations: int = 500,
        tolerance: float = 1e-6,
        step_size: float = 1.0,
        **kwargs: object,
    ) -> None:
        super().__init__(robot, **kwargs)
        cfg = config or IKConfig()
        self._max_iter = max_iterations if config is None else cfg.max_iterations
        self._tol = tolerance if config is None else cfg.tolerance
        self._step = step_size if config is None else cfg.step_size
        self._jac = JacobianComputer(robot)

    def solve(
        self,
        target: EndEffectorPose,
        q0: Optional[Sequence[float]] = None,
    ) -> IKSolution:
        """Solve IK iteratively using the Jacobian pseudo-inverse.

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

        for iteration in range(1, self._max_iter + 1):
            fk = self._robot.forward_kinematics(q)
            current_pos = fk.position[:2] if is_planar else fk.position[:3]
            error = target_pos - current_pos
            error_norm = float(np.linalg.norm(error))

            if error_norm < self._tol:
                logger.debug(
                    "Converged at iteration %d (error=%.2e)", iteration, error_norm,
                )
                break

            J = self._jac.compute(q)
            J_pinv = np.linalg.pinv(J)
            delta_q = J_pinv @ error
            q = q + self._step * delta_q

        elapsed = (time.perf_counter() - t_start) * 1000.0
        success = error_norm < self._tol

        if not success:
            messages.append(
                f"Did not converge after {iteration} iterations "
                f"(error={error_norm:.2e})"
            )
            logger.warning(messages[-1])
        else:
            messages.append(
                f"Converged in {iteration} iterations (error={error_norm:.2e})"
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
