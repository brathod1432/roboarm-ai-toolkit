"""Example 01: Two-link planar robot forward kinematics.

Demonstrates creating a 2-link planar robot, computing FK at several
joint configurations, and optionally plotting the arm pose.
"""

from __future__ import annotations

import logging
import math
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available; skipping plot generation")

import numpy as np

from roboarm.robots import create_two_link_planar
from roboarm.utils.angle_utils import rad2deg


def main() -> None:
    """Run the two-link FK demonstration."""
    logger.info("=== Two-Link Planar FK Demo ===")

    # Create robot with equal link lengths
    robot = create_two_link_planar(link1=1.0, link2=1.0)
    logger.info("Created robot: %s", robot.name)
    logger.info("  DOF: %d, Joints: %d", robot.n_dof, robot.n_joints)
    logger.info("  Joint names: %s", robot.joint_names)

    # Define several test configurations
    configurations = [
        ([0.0, 0.0], "Fully extended along X"),
        ([math.pi / 2, 0.0], "Shoulder at 90 deg"),
        ([math.pi / 4, -math.pi / 4], "Diagonal reach"),
        ([0.0, math.pi / 2], "Elbow at 90 deg"),
        ([math.pi / 3, -math.pi / 6], "Arbitrary config"),
        ([0.0, math.pi], "Folded back"),
    ]

    logger.info("")
    logger.info("%-35s %10s %10s %10s", "Configuration", "X", "Y", "Z")
    logger.info("-" * 70)

    for angles, description in configurations:
        pose = robot.forward_kinematics(angles)
        angles_deg = [rad2deg(a) for a in angles]
        logger.info(
            "%-35s %10.4f %10.4f %10.4f",
            f"{description} ({angles_deg[0]:.0f}, {angles_deg[1]:.0f} deg)",
            pose.x,
            pose.y,
            pose.z,
        )

    # Optional: create a visualization
    if HAS_MATPLOTLIB:
        logger.info("")
        logger.info("Generating arm configuration plot...")

        try:
            from roboarm.visualization.arm_plot import ArmVisualizer

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            viz = ArmVisualizer(robot)

            for idx, (angles, description) in enumerate(configurations):
                row, col = divmod(idx, 3)
                ax = axes[row][col]
                viz.plot_2d(angles, ax=ax, title=description)

            fig.suptitle("Two-Link Planar Robot - FK Configurations", fontsize=14)
            fig.tight_layout()
            fig.savefig("two_link_fk_configs.png", dpi=100)
            logger.info("Plot saved to two_link_fk_configs.png")
        except Exception as exc:
            logger.warning("Could not generate plot: %s", exc)

    logger.info("=== Demo complete ===")


if __name__ == "__main__":
    main()
