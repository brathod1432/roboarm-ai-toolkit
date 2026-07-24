"""Robot arm visualisation utilities.

Provides :class:`ArmVisualizer` for plotting the robot configuration
in 2-D and for visualising the joint-space configuration.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from roboarm.core.robot import RobotArm

logger = logging.getLogger(__name__)

# Use non-interactive backend when no display is available.
try:
    _display = matplotlib.get_backend()
    if _display.lower() in ("agg",):
        pass  # already non-interactive
except Exception:  # pragma: no cover
    matplotlib.use("Agg")


class ArmVisualizer:
    """Visualise a :class:`RobotArm` configuration.

    Args:
        robot: The robot arm model to visualise.

    Example::

        from roboarm.robots.two_link_planar import create_two_link_planar
        robot = create_two_link_planar()
        viz = ArmVisualizer(robot)
        ax = viz.plot_2d([0.5, -0.3])
    """

    def __init__(self, robot: RobotArm) -> None:
        self._robot = robot
        logger.debug("ArmVisualizer created for %s", robot.name)

    # ------------------------------------------------------------------
    # 2-D configuration plot
    # ------------------------------------------------------------------

    def plot_2d(
        self,
        q: Sequence[float],
        ax: matplotlib.axes.Axes | None = None,
        show_workspace: bool = False,
        title: str | None = None,
    ) -> matplotlib.axes.Axes:
        """Plot a 2-D view of the robot in configuration *q*.

        For planar robots the x-y plane is used.  For spatial (3-D)
        robots the x-z projection is shown instead.

        Args:
            q: Joint angles in radians (length == ``robot.n_dof``).
            ax: Existing matplotlib axes, or ``None`` to create new ones.
            show_workspace: If ``True``, draw the approximate workspace
                boundary as a dashed circle.
            title: Optional plot title.

        Returns:
            The matplotlib axes containing the plot.
        """
        positions = self._robot.joint_positions(q)
        is_planar = self._is_planar()

        if is_planar:
            xs = positions[:, 0]
            ys = positions[:, 1]
            x_label, y_label = "X", "Y"
        else:
            xs = positions[:, 0]
            ys = positions[:, 2]
            x_label, y_label = "X", "Z"

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Draw links as thick lines
        ax.plot(xs, ys, "o-", color="steelblue", linewidth=3, markersize=4,
                label="Links")

        # Mark joints with circles
        ax.plot(xs[:-1], ys[:-1], "o", color="navy", markersize=8,
                zorder=5, label="Joints")

        # Mark end-effector with a star
        ax.plot(xs[-1], ys[-1], "*", color="crimson", markersize=14,
                zorder=6, label="End-Effector")

        if show_workspace:
            self._draw_workspace_boundary(ax, is_planar)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title(f"{self._robot.name} — 2-D View")

        logger.debug("Plotted 2-D view for q=%s", list(np.round(q, 3)))
        return ax

    # ------------------------------------------------------------------
    # Configuration-space bar chart
    # ------------------------------------------------------------------

    def plot_configuration_space(
        self,
        q: Sequence[float],
        ax: matplotlib.axes.Axes | None = None,
    ) -> matplotlib.axes.Axes:
        """Plot the joint-space configuration as a bar chart.

        Each bar represents one joint angle.  If joint limits are
        available they are overlaid as horizontal range markers.

        Args:
            q: Joint angles in radians (length == ``robot.n_dof``).
            ax: Existing matplotlib axes, or ``None`` to create new ones.

        Returns:
            The matplotlib axes containing the plot.
        """
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        names = self._robot.joint_names
        limits = self._robot.joint_limits

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        indices = np.arange(len(q_arr))
        ax.bar(indices, np.degrees(q_arr), color="steelblue", alpha=0.8,
               label="Current angle")

        # Overlay joint limits as error-bar style markers
        for idx, lim in enumerate(limits):
            if lim is not None:
                ax.plot(
                    [idx, idx],
                    [np.degrees(lim.lower), np.degrees(lim.upper)],
                    color="grey",
                    linewidth=2,
                    alpha=0.5,
                )

        ax.set_xticks(indices)
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("Angle (degrees)")
        ax.set_title(f"{self._robot.name} — Configuration Space")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

        logger.debug("Plotted configuration space for q=%s", list(np.round(q, 3)))
        return ax

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_planar(self) -> bool:
        """Heuristic: all alpha values are zero -> planar robot."""
        for jc in self._robot.joints:
            if abs(jc.dh_params.alpha) > 1e-9:
                return False
        return True

    def _draw_workspace_boundary(
        self,
        ax: matplotlib.axes.Axes,
        is_planar: bool,
    ) -> None:
        """Draw an approximate workspace boundary circle.

        The radius is the sum of all link lengths (maximum reach).
        """
        total_reach = 0.0
        for jc in self._robot.joints:
            total_reach += abs(jc.dh_params.a) + abs(jc.dh_params.d)

        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        cx = total_reach * np.cos(theta)
        cy = total_reach * np.sin(theta)
        ax.plot(cx, cy, "--", color="grey", alpha=0.4, linewidth=1,
                label="Workspace boundary")
