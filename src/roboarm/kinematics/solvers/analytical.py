"""Closed-form analytical IK solver for 2-link planar robots.

Uses the law of cosines to compute both elbow-up and elbow-down
solutions in constant time.  Registered as ``"analytical_2link"``.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence

import numpy as np

from roboarm.core.exceptions import ConfigurationError
from roboarm.core.robot import RobotArm
from roboarm.core.types import EndEffectorPose, IKSolution, JointSolution
from roboarm.kinematics.inverse import IKSolverBase
from roboarm.kinematics.solvers.registry import IKSolverRegistry

logger = logging.getLogger(__name__)


@IKSolverRegistry.register("analytical_2link")
class Analytical2LinkSolver(IKSolverBase):
    """Closed-form inverse kinematics for a 2-link planar (RR) arm.

    The solver extracts link lengths ``L1`` and ``L2`` from the DH ``a``
    parameters and applies the law of cosines to produce up to two
    solutions (elbow-up and elbow-down).

    Args:
        robot: A 2-DOF planar robot arm.

    Raises:
        ConfigurationError: If the robot does not have exactly 2 variable
            joints or is not planar.
    """

    def __init__(self, robot: RobotArm, **kwargs: object) -> None:
        super().__init__(robot, **kwargs)

        if robot.n_dof != 2:
            raise ConfigurationError(
                f"Analytical 2-link solver requires exactly 2 DOF, "
                f"got {robot.n_dof}"
            )

        joints = robot.joints
        self._l1 = abs(joints[0].dh_params.a)
        self._l2 = abs(joints[1].dh_params.a)

        if self._l1 <= 0 or self._l2 <= 0:
            raise ConfigurationError(
                "Both link lengths must be positive "
                f"(L1={self._l1}, L2={self._l2})"
            )

        logger.info(
            "Analytical2LinkSolver: L1=%.4f, L2=%.4f", self._l1, self._l2,
        )

    def solve(
        self,
        target: EndEffectorPose,
        q0: Sequence[float] | None = None,
    ) -> IKSolution:
        """Compute closed-form IK for the 2-link planar arm.

        Args:
            target: Desired end-effector pose (only ``x`` and ``y`` are
                used).
            q0: Ignored for the analytical solver.

        Returns:
            An :class:`IKSolution` with up to two solutions.  The
            elbow-down solution is returned as the primary result
            when both are valid.
        """
        t_start = time.perf_counter()

        x = float(target.position[0])
        y = float(target.position[1])
        l1, l2 = self._l1, self._l2

        dist_sq = x * x + y * y
        reach_max = l1 + l2
        reach_min = abs(l1 - l2)

        messages: list[str] = []

        # Check reachability
        dist = math.sqrt(dist_sq)
        if dist > reach_max + 1e-10 or dist < reach_min - 1e-10:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            messages.append(
                f"Target ({x:.4f}, {y:.4f}) is unreachable "
                f"(dist={dist:.4f}, range=[{reach_min:.4f}, {reach_max:.4f}])"
            )
            logger.warning(messages[-1])
            return IKSolution(
                success=False,
                iterations=0,
                residual_error=abs(dist - reach_max),
                computation_time_ms=elapsed,
                solver_name=self.name,
                messages=messages,
            )

        # Law of cosines: cos(q2) = (x^2 + y^2 - L1^2 - L2^2) / (2*L1*L2)
        cos_q2 = (dist_sq - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
        cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))

        sin_q2_pos = math.sqrt(1.0 - cos_q2 * cos_q2)

        solutions: list[JointSolution] = []

        for sign, label in [(1.0, "elbow-up"), (-1.0, "elbow-down")]:
            sin_q2 = sign * sin_q2_pos
            q2 = math.atan2(sin_q2, cos_q2)
            q1 = math.atan2(y, x) - math.atan2(
                l2 * sin_q2, l1 + l2 * cos_q2,
            )
            sol = JointSolution(values=np.array([q1, q2], dtype=np.float64))
            solutions.append(sol)
            logger.debug("  %s: q1=%.4f, q2=%.4f", label, q1, q2)

        # Compute residual for primary solution
        fk_check = self._robot.forward_kinematics(solutions[0].values)
        residual = float(np.linalg.norm(
            fk_check.position[:2] - target.position[:2],
        ))

        elapsed = (time.perf_counter() - t_start) * 1000.0
        messages.append(
            f"Found {len(solutions)} solution(s) in {elapsed:.3f} ms"
        )

        return IKSolution(
            success=True,
            primary=solutions[0],
            alternatives=solutions[1:],
            iterations=0,
            residual_error=residual,
            computation_time_ms=elapsed,
            solver_name=self.name,
            messages=messages,
        )
