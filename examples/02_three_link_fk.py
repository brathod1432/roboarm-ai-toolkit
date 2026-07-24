"""Example 02: Three-link planar robot forward kinematics.

Demonstrates creating a 3-link planar (redundant) robot, computing FK
at multiple configurations, and showing redundancy by finding different
joint configurations that reach the same workspace region.
"""

from __future__ import annotations

import logging
import math

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

from roboarm.robots import create_three_link_planar
from roboarm.utils.angle_utils import rad2deg


def main() -> None:
    """Run the three-link FK and redundancy demonstration."""
    logger.info("=== Three-Link Planar FK Demo ===")

    # Create robot
    robot = create_three_link_planar(link1=1.0, link2=1.0, link3=0.5)
    logger.info("Created robot: %s", robot.name)
    logger.info("  DOF: %d, Joints: %d", robot.n_dof, robot.n_joints)
    logger.info("  Joint names: %s", robot.joint_names)

    # Part 1: FK at several configurations
    configurations = [
        ([0.0, 0.0, 0.0], "Fully extended"),
        ([math.pi / 2, 0.0, 0.0], "Shoulder at 90 deg"),
        ([0.0, math.pi / 2, 0.0], "Elbow at 90 deg"),
        ([0.0, 0.0, math.pi / 2], "Wrist at 90 deg"),
        ([math.pi / 4, -math.pi / 4, math.pi / 6], "Mixed config"),
    ]

    logger.info("")
    logger.info("--- FK Results ---")
    logger.info("%-30s %10s %10s %10s", "Configuration", "X", "Y", "Z")
    logger.info("-" * 65)

    for angles, description in configurations:
        pose = robot.forward_kinematics(angles)
        logger.info(
            "%-30s %10.4f %10.4f %10.4f",
            description, pose.x, pose.y, pose.z,
        )

    # Part 2: Show redundancy -- multiple configs reaching similar position
    logger.info("")
    logger.info("--- Redundancy Demonstration ---")
    logger.info("Finding different configurations that reach similar regions:")
    logger.info("")

    redundant_configs = [
        ([0.5, 0.3, -0.2], "Config A"),
        ([0.3, 0.7, -0.4], "Config B"),
        ([0.7, -0.1, 0.2], "Config C"),
        ([0.1, 0.9, -0.4], "Config D"),
    ]

    logger.info(
        "%-12s %8s %8s %8s   |  %-30s",
        "Config", "X", "Y", "Z", "Joint angles (deg)",
    )
    logger.info("-" * 75)

    for angles, label in redundant_configs:
        pose = robot.forward_kinematics(angles)
        angles_deg = [f"{rad2deg(a):.1f}" for a in angles]
        logger.info(
            "%-12s %8.4f %8.4f %8.4f   |  [%s]",
            label, pose.x, pose.y, pose.z,
            ", ".join(angles_deg),
        )

    # Part 3: Demonstrate 3-link reach vs 2-link
    logger.info("")
    logger.info("--- Workspace Comparison ---")
    max_reach = sum(
        abs(j.dh_params.a) for j in robot.joints
    )
    logger.info("Maximum reach (sum of link lengths): %.2f", max_reach)

    # Optional: create visualization
    if HAS_MATPLOTLIB:
        logger.info("")
        logger.info("Generating configuration comparison plot...")

        try:
            from roboarm.visualization.arm_plot import ArmVisualizer

            fig, axes = plt.subplots(1, len(redundant_configs), figsize=(20, 5))
            viz = ArmVisualizer(robot)

            for idx, (angles, label) in enumerate(redundant_configs):
                pose = robot.forward_kinematics(angles)
                ax = axes[idx]
                viz.plot_2d(
                    angles, ax=ax,
                    title=f"{label}: ({pose.x:.2f}, {pose.y:.2f})",
                )

            fig.suptitle(
                "3-Link Planar Robot - Redundant Configurations",
                fontsize=14,
            )
            fig.tight_layout()
            fig.savefig("three_link_redundancy.png", dpi=100)
            logger.info("Plot saved to three_link_redundancy.png")
        except Exception as exc:
            logger.warning("Could not generate plot: %s", exc)

    logger.info("=== Demo complete ===")


if __name__ == "__main__":
    main()
