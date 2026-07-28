"""Workspace visualisation via random forward-kinematics sampling.

Provides :class:`WorkspaceVisualizer` to scatter-plot reachable
positions of a :class:`RobotArm`.
"""

from __future__ import annotations

import logging

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from roboarm.core.robot import RobotArm

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
        seed: int | None = None,
        ax: matplotlib.axes.Axes | None = None,
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

        points = self._sample_points(n_samples, seed=seed)
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

    def plot_manipulability_heatmap(
        self,
        n_samples: int = 2000,
        seed: int | None = None,
        ax: matplotlib.axes.Axes | None = None,
        colormap: str = "viridis",
    ) -> matplotlib.axes.Axes:
        """Scatter-plot the workspace coloured by manipulability index.

        Each sampled end-effector position is coloured by the Yoshikawa
        manipulability at that configuration.  Bright areas are dexterous;
        dark areas are near-singular.

        Args:
            n_samples: Number of random configurations to sample.
            seed: Optional RNG seed for reproducibility.
            ax: Existing axes, or ``None`` to create new ones.
            colormap: Matplotlib colormap name (default ``"viridis"``).

        Returns:
            The matplotlib axes containing the scatter plot.
        """
        from roboarm.kinematics.jacobian import JacobianComputer

        logger.info("Sampling %d configs for manipulability heatmap", n_samples)

        limits = self._robot.joint_limits
        rng = np.random.default_rng(seed)
        jc = JacobianComputer(self._robot)
        is_planar = self._is_planar()

        positions = np.empty((n_samples, 3), dtype=np.float64)
        mu_values = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            q = np.array([
                rng.uniform(lim.lower, lim.upper) if lim else rng.uniform(-np.pi, np.pi)
                for lim in limits
            ])
            pose = self._robot.forward_kinematics(q)
            positions[i] = pose.position
            mu_values[i] = jc.manipulability(q)

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(9, 8))

        xs = positions[:, 0]
        ys = positions[:, 1] if is_planar else positions[:, 2]
        sc = ax.scatter(xs, ys, c=mu_values, cmap=colormap, s=3, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Manipulability μ")

        ax.set_xlabel("X")
        ax.set_ylabel("Y" if is_planar else "Z")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.2)
        ax.set_title(
            f"{self._robot.name} — Manipulability Heatmap ({n_samples} samples)"
        )
        return ax

    def plot_3d(
        self,
        n_samples: int = 2000,
        seed: int | None = None,
        ax: object | None = None,
    ) -> object:
        """Scatter-plot the workspace in 3-D using mplot3d.

        Args:
            n_samples: Number of random FK evaluations.
            seed: Optional RNG seed.
            ax: Existing 3-D axes, or ``None`` to create new ones.

        Returns:
            The ``Axes3D`` instance.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        points = self._sample_points(n_samples, seed=seed)

        if ax is None:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(111, projection="3d")

        ax.scatter(  # type: ignore[union-attr]
            points[:, 0], points[:, 1], points[:, 2],
            s=1, alpha=0.4, color="steelblue",
        )
        ax.set_xlabel("X")  # type: ignore[union-attr]
        ax.set_ylabel("Y")  # type: ignore[union-attr]
        ax.set_zlabel("Z")  # type: ignore[union-attr]
        ax.set_title(  # type: ignore[union-attr]
            f"{self._robot.name} — 3-D Workspace ({n_samples} samples)"
        )
        return ax

    def plot_trajectory_overlay(
        self,
        trajectory: np.ndarray,
        n_samples: int = 1000,
        seed: int | None = None,
        ax: matplotlib.axes.Axes | None = None,
    ) -> matplotlib.axes.Axes:
        """Plot workspace point cloud with the Cartesian trajectory overlaid.

        Args:
            trajectory: ``(n_steps, n_dof)`` joint angle array.
            n_samples: Number of background workspace samples.
            seed: Optional RNG seed.
            ax: Existing axes or ``None``.

        Returns:
            The matplotlib axes.
        """
        is_planar = self._is_planar()
        points = self._sample_points(n_samples, seed=seed)

        traj = np.asarray(trajectory, dtype=np.float64)
        ee_positions = np.array([
            self._robot.forward_kinematics(traj[i]).position
            for i in range(len(traj))
        ])

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Background workspace
        xs_bg = points[:, 0]
        ys_bg = points[:, 1] if is_planar else points[:, 2]
        ax.scatter(xs_bg, ys_bg, s=1, alpha=0.2, color="lightsteelblue", label="Workspace")

        # Trajectory path
        xs_t = ee_positions[:, 0]
        ys_t = ee_positions[:, 1] if is_planar else ee_positions[:, 2]
        ax.plot(xs_t, ys_t, "r-", linewidth=2, label="EE trajectory")
        ax.plot(xs_t[0], ys_t[0], "go", markersize=8, label="Start")
        ax.plot(xs_t[-1], ys_t[-1], "rs", markersize=8, label="End")

        ax.set_xlabel("X")
        ax.set_ylabel("Y" if is_planar else "Z")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f"{self._robot.name} — Workspace + Trajectory")
        return ax

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_points(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        """Generate end-effector positions from random joint angles.

        Returns:
            ``(n_samples, 3)`` array of x, y, z positions.
        """
        limits = self._robot.joint_limits
        n_dof = self._robot.n_dof
        rng = np.random.default_rng(seed)

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
        """Delegate to the canonical ``RobotArm.is_planar`` property."""
        return self._robot.is_planar
