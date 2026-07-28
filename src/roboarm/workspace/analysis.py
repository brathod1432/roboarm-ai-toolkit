"""Workspace analysis utilities.

Provides :class:`WorkspaceAnalyzer` for approximate reachability
testing, workspace sampling, and bounding-box estimation via
Monte-Carlo forward kinematics.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from roboarm.core.robot import RobotArm

logger = logging.getLogger(__name__)


class WorkspaceAnalyzer:
    """Analyse the reachable workspace of a :class:`RobotArm`.

    All methods use randomised forward kinematics.  Results are
    approximate and improve with larger sample counts.

    Args:
        robot: The robot arm model to analyse.

    Example::

        from roboarm.robots.two_link_planar import create_two_link_planar
        analyzer = WorkspaceAnalyzer(create_two_link_planar())
        reachable = analyzer.is_reachable([1.5, 0.3, 0.0])
    """

    def __init__(self, robot: RobotArm) -> None:
        self._robot = robot
        logger.debug("WorkspaceAnalyzer created for %s", robot.name)

    # ------------------------------------------------------------------
    # Reachability check
    # ------------------------------------------------------------------

    def is_reachable(
        self,
        target_position: Sequence[float],
        n_samples: int = 500,
        tolerance: float = 0.05,
        seed: int | None = None,
    ) -> bool:
        """Quick Monte-Carlo reachability check.

        Samples random joint configurations, computes FK for each, and
        returns ``True`` if any sampled end-effector position is within
        *tolerance* of *target_position*.

        Args:
            target_position: Desired ``[x, y, z]`` position.
            n_samples: Number of random configurations to evaluate.
            tolerance: Maximum Euclidean distance to declare reachable.
            seed: Optional integer seed for reproducible sampling.

        Returns:
            ``True`` if the target is within the approximate workspace.
        """
        target = np.asarray(target_position, dtype=np.float64).ravel()[:3]
        points = self.sample_workspace(n_samples, seed=seed)

        distances = np.linalg.norm(points - target, axis=1)
        min_dist = float(np.min(distances))

        reachable = min_dist <= tolerance
        logger.info(
            "Reachability check: target=%s  min_dist=%.4f  reachable=%s",
            target.tolist(),
            min_dist,
            reachable,
        )
        return reachable

    # ------------------------------------------------------------------
    # Workspace sampling
    # ------------------------------------------------------------------

    def sample_workspace(
        self,
        n_samples: int = 1000,
        seed: int | None = None,
    ) -> np.ndarray:
        """Sample workspace points by random forward kinematics.

        Args:
            n_samples: Number of random configurations to evaluate.
            seed: Optional integer seed for reproducible sampling.
                Pass the same seed to get identical point clouds across
                calls.

        Returns:
            ``(n_samples, 3)`` array of ``[x, y, z]`` end-effector
            positions.
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

        logger.debug(
            "Sampled %d workspace points for %s", n_samples, self._robot.name
        )
        return positions

    # ------------------------------------------------------------------
    # Workspace bounds
    # ------------------------------------------------------------------

    def workspace_bounds(
        self,
        n_samples: int = 2000,
        seed: int | None = None,
    ) -> dict[str, tuple[float, float]]:
        """Compute approximate axis-aligned bounding box of the workspace.

        Args:
            n_samples: Number of random FK evaluations.
            seed: Optional integer seed for reproducible sampling.

        Returns:
            Dictionary with keys ``'x'``, ``'y'``, ``'z'`` mapping to
            ``(min, max)`` tuples.

        Example::

            bounds = analyzer.workspace_bounds(5000)
            x_min, x_max = bounds['x']
        """
        points = self.sample_workspace(n_samples, seed=seed)

        bounds: dict[str, tuple[float, float]] = {
            "x": (float(np.min(points[:, 0])), float(np.max(points[:, 0]))),
            "y": (float(np.min(points[:, 1])), float(np.max(points[:, 1]))),
            "z": (float(np.min(points[:, 2])), float(np.max(points[:, 2]))),
        }

        logger.info("Workspace bounds (%d samples): %s", n_samples, bounds)
        return bounds

    def plot(
        self,
        n_samples: int = 1000,
        seed: int | None = None,
        ax: object | None = None,
    ) -> object:
        """Sample the workspace and produce a scatter plot directly.

        Convenience method that combines :meth:`sample_workspace` with
        :class:`~roboarm.visualization.workspace_plot.WorkspaceVisualizer`
        so callers do not need to wire the two classes together manually.

        Args:
            n_samples: Number of random FK evaluations.
            seed: Optional integer seed for reproducible sampling.
            ax: Existing ``matplotlib.axes.Axes``, or ``None`` to create
                new ones.

        Returns:
            The ``matplotlib.axes.Axes`` containing the scatter plot.

        Example::

            from roboarm.robots.two_link_planar import create_two_link_planar
            from roboarm.workspace.analysis import WorkspaceAnalyzer

            analyzer = WorkspaceAnalyzer(create_two_link_planar())
            ax = analyzer.plot(n_samples=2000, seed=42)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for WorkspaceAnalyzer.plot(). "
                "Install it with: pip install matplotlib"
            ) from exc

        points = self.sample_workspace(n_samples, seed=seed)
        is_planar = self._robot.is_planar

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 8))

        if is_planar:
            ax.scatter(points[:, 0], points[:, 1], s=1, alpha=0.4,  # type: ignore[union-attr]
                       color="steelblue", label="Reachable points")
            ax.set_xlabel("X")  # type: ignore[union-attr]
            ax.set_ylabel("Y")  # type: ignore[union-attr]
        else:
            ax.scatter(points[:, 0], points[:, 2], s=1, alpha=0.4,  # type: ignore[union-attr]
                       color="steelblue", label="Reachable points")
            ax.set_xlabel("X")  # type: ignore[union-attr]
            ax.set_ylabel("Z")  # type: ignore[union-attr]

        ax.set_aspect("equal", adjustable="datalim")  # type: ignore[union-attr]
        ax.grid(True, alpha=0.3)  # type: ignore[union-attr]
        ax.legend(fontsize=8)  # type: ignore[union-attr]
        ax.set_title(f"{self._robot.name} — Reachable Workspace ({n_samples} samples)")  # type: ignore[union-attr]
        logger.info(
            "WorkspaceAnalyzer.plot: plotted %d points for %s",
            n_samples, self._robot.name,
        )
        return ax
