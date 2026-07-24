"""Workspace analysis utilities.

Provides :class:`WorkspaceAnalyzer` for approximate reachability
testing, workspace sampling, and bounding-box estimation via
Monte-Carlo forward kinematics.
"""

from __future__ import annotations

import logging
from typing import Dict, Sequence, Tuple

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
    ) -> bool:
        """Quick Monte-Carlo reachability check.

        Samples random joint configurations, computes FK for each, and
        returns ``True`` if any sampled end-effector position is within
        *tolerance* of *target_position*.

        Args:
            target_position: Desired ``[x, y, z]`` position.
            n_samples: Number of random configurations to evaluate.
            tolerance: Maximum Euclidean distance to declare reachable.

        Returns:
            ``True`` if the target is within the approximate workspace.
        """
        target = np.asarray(target_position, dtype=np.float64).ravel()[:3]
        points = self.sample_workspace(n_samples)

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

    def sample_workspace(self, n_samples: int = 1000) -> np.ndarray:
        """Sample workspace points by random forward kinematics.

        Args:
            n_samples: Number of random configurations to evaluate.

        Returns:
            ``(n_samples, 3)`` array of ``[x, y, z]`` end-effector
            positions.
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

        logger.debug(
            "Sampled %d workspace points for %s", n_samples, self._robot.name
        )
        return positions

    # ------------------------------------------------------------------
    # Workspace bounds
    # ------------------------------------------------------------------

    def workspace_bounds(
        self, n_samples: int = 2000
    ) -> Dict[str, Tuple[float, float]]:
        """Compute approximate axis-aligned bounding box of the workspace.

        Args:
            n_samples: Number of random FK evaluations.

        Returns:
            Dictionary with keys ``'x'``, ``'y'``, ``'z'`` mapping to
            ``(min, max)`` tuples.

        Example::

            bounds = analyzer.workspace_bounds(5000)
            x_min, x_max = bounds['x']
        """
        points = self.sample_workspace(n_samples)

        bounds: Dict[str, Tuple[float, float]] = {
            "x": (float(np.min(points[:, 0])), float(np.max(points[:, 0]))),
            "y": (float(np.min(points[:, 1])), float(np.max(points[:, 1]))),
            "z": (float(np.min(points[:, 2])), float(np.max(points[:, 2]))),
        }

        logger.info("Workspace bounds (%d samples): %s", n_samples, bounds)
        return bounds
