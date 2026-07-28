"""Trajectory safety validation.

Checks a joint-space trajectory for joint-limit violations and
kinematic singularities before execution on hardware.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from roboarm.core.robot import RobotArm

logger = logging.getLogger(__name__)


@dataclass
class LimitViolation:
    """A single joint-limit violation in a trajectory.

    Attributes:
        step: Waypoint index (0-based).
        joint: Joint index (0-based).
        joint_name: Human-readable joint name.
        value: Actual angle value (radians).
        lower: Lower limit (radians).
        upper: Upper limit (radians).
    """
    step: int
    joint: int
    joint_name: str
    value: float
    lower: float
    upper: float


@dataclass
class SingularityWarning:
    """A near-singular waypoint in a trajectory.

    Attributes:
        step: Waypoint index (0-based).
        manipulability: Yoshikawa index at this configuration.
        threshold: The threshold below which the point is flagged.
    """
    step: int
    manipulability: float
    threshold: float


@dataclass
class TrajectoryReport:
    """Full validation report for a trajectory.

    Attributes:
        is_safe: ``True`` when no limit violations AND no singularities.
        limit_violations: All joint-limit violations found.
        singularities: All near-singular waypoints found.
        max_joint_step_rad: Maximum angle change between adjacent steps
            across all joints (useful for detecting velocity spikes).
        n_steps: Total number of waypoints checked.
        n_dof: Degrees of freedom.
    """
    is_safe: bool
    limit_violations: list[LimitViolation] = field(default_factory=list)
    singularities: list[SingularityWarning] = field(default_factory=list)
    max_joint_step_rad: float = 0.0
    n_steps: int = 0
    n_dof: int = 0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Trajectory validation ({self.n_steps} steps, {self.n_dof} DOF)",
            f"  Safe: {self.is_safe}",
            f"  Limit violations: {len(self.limit_violations)}",
            f"  Singular waypoints: {len(self.singularities)}",
            f"  Max joint step: {self.max_joint_step_rad:.4f} rad",
        ]
        for v in self.limit_violations[:5]:
            lines.append(
                f"    Violation step={v.step} joint={v.joint_name} "
                f"value={v.value:.4f} limits=[{v.lower:.4f},{v.upper:.4f}]"
            )
        if len(self.limit_violations) > 5:
            lines.append(f"    ... and {len(self.limit_violations) - 5} more")
        for s in self.singularities[:5]:
            lines.append(
                f"    Singularity step={s.step} mu={s.manipulability:.6e}"
            )
        if len(self.singularities) > 5:
            lines.append(f"    ... and {len(self.singularities) - 5} more")
        return "\n".join(lines)


class TrajectoryValidator:
    """Validates a joint-space trajectory for safety before execution.

    Args:
        robot: The robot arm model.
        singularity_threshold: Manipulability below which a configuration
            is flagged as singular.  Default ``1e-4``.
        check_singularities: If ``True`` (default), compute manipulability
            at every waypoint.  Disable for speed on long trajectories
            where singularity checking is not needed.

    Example::

        from roboarm.trajectory.validation import TrajectoryValidator

        validator = TrajectoryValidator(robot)
        report = validator.check(trajectory_array)
        if not report.is_safe:
            print(report.summary())
    """

    def __init__(
        self,
        robot: RobotArm,
        singularity_threshold: float = 1e-4,
        check_singularities: bool = True,
    ) -> None:
        self._robot = robot
        self._threshold = singularity_threshold
        self._check_sing = check_singularities

    def check(self, trajectory: np.ndarray) -> TrajectoryReport:
        """Validate every waypoint in *trajectory*.

        Args:
            trajectory: ``(n_steps, n_dof)`` joint angle array in radians.

        Returns:
            :class:`TrajectoryReport` with full validation details.
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != self._robot.n_dof:
            raise ValueError(
                f"Expected (n_steps, {self._robot.n_dof}) trajectory, "
                f"got {traj.shape}"
            )

        n_steps, n_dof = traj.shape
        limits = self._robot.joint_limits
        joint_names = self._robot.joint_names

        violations: list[LimitViolation] = []
        singularities: list[SingularityWarning] = []

        jac_computer = None
        if self._check_sing:
            from roboarm.kinematics.jacobian import JacobianComputer
            jac_computer = JacobianComputer(self._robot)

        for i in range(n_steps):
            q = traj[i]

            # Limit check
            for j, (lim, jname) in enumerate(zip(limits, joint_names)):
                if lim is not None:
                    if q[j] < lim.lower - 1e-9 or q[j] > lim.upper + 1e-9:
                        violations.append(LimitViolation(
                            step=i, joint=j, joint_name=jname,
                            value=float(q[j]),
                            lower=lim.lower, upper=lim.upper,
                        ))

            # Singularity check
            if jac_computer is not None:
                mu = jac_computer.manipulability(q)
                if mu < self._threshold:
                    singularities.append(SingularityWarning(
                        step=i, manipulability=mu, threshold=self._threshold,
                    ))

        # Max joint step between adjacent waypoints
        max_step = 0.0
        if n_steps > 1:
            diffs = np.abs(np.diff(traj, axis=0))
            max_step = float(np.max(diffs))

        is_safe = len(violations) == 0 and len(singularities) == 0
        report = TrajectoryReport(
            is_safe=is_safe,
            limit_violations=violations,
            singularities=singularities,
            max_joint_step_rad=max_step,
            n_steps=n_steps,
            n_dof=n_dof,
        )

        if is_safe:
            logger.info(
                "Trajectory validation PASSED (%d steps, %d DOF)",
                n_steps, n_dof,
            )
        else:
            logger.warning(
                "Trajectory validation FAILED: %d limit violations, "
                "%d singular waypoints",
                len(violations), len(singularities),
            )

        return report
