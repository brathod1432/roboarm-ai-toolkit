"""Workspace visualisation via random forward-kinematics sampling.

Provides :class:`WorkspaceVisualizer` to scatter-plot reachable
positions of a :class:`RobotArm`.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from roboarm.core.robot import RobotArm
from roboarm.core.transform import extract_position

logger = logging.getLogger(__name__)


class WorkspaceVisualizer:
    """Visualise the reachable workspace of a :class:`RobotArm`.

    Args:
        robot: The robot arm model whose workspace is to be plotted.

    Example::

        from roboarm.robots.two_link_planar import create_two_link_planar
        robot = create_two_link_planar()
        viz = WorkspaceVisualizer(robot)
        ax = viz.plot_reachable_workspace(n_samples=2000)
    """

    def __init__(self, robot: RobotArm) -> None:
        self._robot = robot
        logger.debug("WorkspaceVisualizer created for %s", robot.name)

    def plot_reachable_workspace(
        self,
        n_samples: int = 1000,
        ax: Optional[matplotlib.axes.Axes] = None,
    ) -> matplotlib.axes.Axes:
        """Sample random joint configurations and plot FK positions.

        For planar robots (all alpha == 0), the x-y plane is used.
        For spatial robots, the x-z projection is shown.

        Args:
            n_samples: Number of random configurations to evaluate.
            ax: Existing matplotlib axes, or ``None`` to create new ones.

        Returns:
            The matplotlib axes containing the scatter plot.
        """
        logger.info(
            "Sampling %d random configurations for workspace plot", n_samples
        )

        points = self._sample_points(n_samples)
        is_planar = self._is_planar()

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 8))

        if is_planar:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=1,
                alpha=0.4,
                color="steelblue",
                label="Reachable points",
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
        else:
            ax.scatter(
                points[:, 0],
                points[:, 2],
                s=1,
                alpha=0.4,
                color="steelblue",
                label="Reachable points",
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Z")

        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f"{self._robot.name} — Reachable Workspace ({n_samples} samples)")

        logger.debug("Workspace scatter plot complete with %d points", n_samples)
        return ax

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_points(self, n_samples: int) -> np.ndarray:
        """Generate end-effector positions from random joint angles.

        Returns:
            ``(n_samples, 3)`` array of x, y, z positions.
        """
        limits = self._robot.joint_limits
        n_dof = self._robot.n_dof
        rng = np.random.default_rng()

        positions = np.empty((n_samples, 3), dtype=np.float64)

        for i in range(n_samples):
            q = np.empty(n_dof, dtype=np.float64)
            for j in range(n_dof):
                lim = limits[j]
                if lim is not None:
                    q[j] = rng.uniform(lim.lower, lim.upper)
                else:
                    q[j] = rng.uniform(-np.pi, np.pi)
            pose = self._robot.forward_kinematics(q)
            positions[i] = pose.position

        return positions

    def _is_planar(self) -> bool:
        """Heuristic: all alpha values are zero -> planar robot."""
        for jc in self._robot.joints:
            if abs(jc.dh_params.alpha) > 1e-9:
                return False
        return True
