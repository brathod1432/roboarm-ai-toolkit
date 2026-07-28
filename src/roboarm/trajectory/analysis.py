"""Trajectory analysis — path length, joint speed, manipulability profile.

Computes diagnostic metrics over a complete joint-space trajectory to
help clients understand motion characteristics before or after execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from roboarm.core.robot import RobotArm

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMetrics:
    """Computed metrics for a joint-space trajectory.

    Attributes:
        cartesian_path_length_m: Total Euclidean distance travelled by
            the end-effector in metres.
        joint_speed_profile: ``(n_steps-1, n_dof)`` finite-difference
            joint velocity estimates in rad/s (requires *dt* > 0).
        max_joint_speed: ``(n_dof,)`` peak speed per joint in rad/s.
        manipulability_profile: ``(n_steps,)`` Yoshikawa index at every
            waypoint.
        min_manipulability: Smallest manipulability value and its index.
        min_manipulability_step: Waypoint index of minimum manipulability.
        smoothness: Mean squared second difference of joint angles
            (a proxy for jerk).  Lower is smoother.
        n_steps: Number of waypoints.
        n_dof: Degrees of freedom.
    """
    cartesian_path_length_m: float = 0.0
    joint_speed_profile: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0))
    )
    max_joint_speed: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )
    manipulability_profile: np.ndarray = field(
        default_factory=lambda: np.empty(0)
    )
    min_manipulability: float = float("inf")
    min_manipulability_step: int = -1
    smoothness: float = 0.0
    n_steps: int = 0
    n_dof: int = 0

    def summary(self, dt: float = 0.0) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Trajectory analysis ({self.n_steps} steps, {self.n_dof} DOF)",
            f"  Cartesian path length: {self.cartesian_path_length_m:.4f} m",
            f"  Smoothness (jerk proxy): {self.smoothness:.6f}",
            f"  Min manipulability: {self.min_manipulability:.6e} "
            f"(step {self.min_manipulability_step})",
        ]
        if self.max_joint_speed.size > 0:
            speeds = ", ".join(f"{v:.3f}" for v in self.max_joint_speed)
            lines.append(f"  Max joint speeds (rad/s): [{speeds}]")
        return "\n".join(lines)


class TrajectoryAnalyzer:
    """Compute diagnostic metrics for a joint-space trajectory.

    Args:
        robot: The robot arm model.
        dt: Time step in seconds between waypoints.  Set to 0 (default)
            to skip speed calculations.

    Example::

        from roboarm.trajectory.analysis import TrajectoryAnalyzer

        analyzer = TrajectoryAnalyzer(robot, dt=0.02)
        metrics = analyzer.analyze(trajectory_array)
        print(metrics.summary())
        print(f"Path length: {metrics.cartesian_path_length_m:.3f} m")
    """

    def __init__(self, robot: RobotArm, dt: float = 0.0) -> None:
        self._robot = robot
        self._dt = float(dt)

    def analyze(self, trajectory: np.ndarray) -> TrajectoryMetrics:
        """Compute metrics for *trajectory*.

        Args:
            trajectory: ``(n_steps, n_dof)`` joint angle array in radians.

        Returns:
            :class:`TrajectoryMetrics` with all computed values.
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != self._robot.n_dof:
            raise ValueError(
                f"Expected (n_steps, {self._robot.n_dof}) trajectory, "
                f"got {traj.shape}"
            )
        n_steps, n_dof = traj.shape

        # --- Cartesian path length ---
        positions = np.array([
            self._robot.forward_kinematics(traj[i]).position
            for i in range(n_steps)
        ])
        diffs = np.diff(positions, axis=0)
        path_length = float(np.sum(np.linalg.norm(diffs, axis=1)))

        # --- Joint speed profile ---
        speed_profile = np.empty((0, n_dof))
        max_speed = np.zeros(n_dof)
        if self._dt > 0 and n_steps > 1:
            speed_profile = np.abs(np.diff(traj, axis=0)) / self._dt
            max_speed = speed_profile.max(axis=0)

        # --- Manipulability profile ---
        from roboarm.kinematics.jacobian import JacobianComputer
        jc = JacobianComputer(self._robot)
        mu_profile = np.array([jc.manipulability(traj[i]) for i in range(n_steps)])
        min_mu_idx = int(np.argmin(mu_profile))
        min_mu = float(mu_profile[min_mu_idx])

        # --- Smoothness (mean squared second differences = jerk proxy) ---
        smoothness = 0.0
        if n_steps >= 3:
            second_diff = np.diff(traj, n=2, axis=0)
            smoothness = float(np.mean(second_diff ** 2))

        metrics = TrajectoryMetrics(
            cartesian_path_length_m=path_length,
            joint_speed_profile=speed_profile,
            max_joint_speed=max_speed,
            manipulability_profile=mu_profile,
            min_manipulability=min_mu,
            min_manipulability_step=min_mu_idx,
            smoothness=smoothness,
            n_steps=n_steps,
            n_dof=n_dof,
        )

        logger.info(
            "Trajectory analysis: path=%.4f m, min_mu=%.4e, smoothness=%.6f",
            path_length, min_mu, smoothness,
        )
        return metrics
