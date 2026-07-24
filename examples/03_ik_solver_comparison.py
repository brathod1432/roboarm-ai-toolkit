"""Example 03: IK solver comparison.

Demonstrates creating a 2-link planar robot, picking a target position,
and solving the IK problem with every registered solver.  Results are
presented in a comparison table.
"""

from __future__ import annotations

import logging
import time

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
from roboarm.core.types import EndEffectorPose
from roboarm.kinematics.solvers.registry import IKSolverRegistry

# Trigger auto-registration of all built-in solvers
import roboarm.kinematics.solvers  # noqa: F401


def _make_target(x: float, y: float, z: float = 0.0) -> EndEffectorPose:
    """Build a target EndEffectorPose from Cartesian coordinates."""
    position = np.array([x, y, z], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    return EndEffectorPose(
        position=position,
        rotation=np.eye(3, dtype=np.float64),
        transform=transform,
    )


def main() -> None:
    """Run the IK solver comparison demonstration."""
    logger.info("=== IK Solver Comparison Demo ===")

    # Create robot
    robot = create_two_link_planar(link1=1.0, link2=1.0)
    logger.info("Robot: %s", robot.name)

    # List available solvers
    available = IKSolverRegistry.available()
    logger.info("Available IK solvers: %s", available)

    # Define target positions to test
    targets = [
        (1.0, 0.5, "Easy target"),
        (0.5, 1.2, "Mid-range target"),
        (1.5, 0.3, "Near workspace edge"),
        (0.3, 0.3, "Close to base"),
    ]

    for target_x, target_y, description in targets:
        logger.info("")
        logger.info("=" * 80)
        logger.info(
            "Target: (%s, %s) -- %s", target_x, target_y, description,
        )
        logger.info("=" * 80)

        target = _make_target(target_x, target_y)

        # Table header
        logger.info(
            "  %-28s %-10s %-14s %-12s %-12s",
            "Solver", "Success", "Residual", "Time (ms)", "Iterations",
        )
        logger.info("  %s", "-" * 76)

        for solver_name in available:
            try:
                solver = IKSolverRegistry.create(solver_name, robot)
                t0 = time.perf_counter()
                result = solver.solve(target)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                status = "Yes" if result.success else "No"
                logger.info(
                    "  %-28s %-10s %-14.6e %-12.2f %-12d",
                    solver_name,
                    status,
                    result.residual_error,
                    elapsed_ms,
                    result.iterations,
                )

                if result.success and result.primary is not None:
                    angles = result.primary.values
                    logger.info(
                        "    -> Solution (rad): [%s]",
                        ", ".join(f"{v:.4f}" for v in angles),
                    )
            except Exception as exc:
                logger.error("  %-28s ERROR: %s", solver_name, exc)

    # Test an unreachable target
    logger.info("")
    logger.info("=" * 80)
    logger.info("Unreachable target: (5.0, 5.0)")
    logger.info("=" * 80)

    target = _make_target(5.0, 5.0)
    for solver_name in available:
        try:
            solver = IKSolverRegistry.create(solver_name, robot)
            result = solver.solve(target)
            status = "Yes" if result.success else "No"
            logger.info(
                "  %-28s %-10s residual=%.6e",
                solver_name, status, result.residual_error,
            )
        except Exception as exc:
            logger.error("  %-28s ERROR: %s", solver_name, exc)

    logger.info("")
    logger.info("=== Demo complete ===")


if __name__ == "__main__":
    main()
