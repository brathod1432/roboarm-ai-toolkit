"""Example 05: AI agent demo.

Demonstrates using the RoboticsCoordinator to process natural-language
queries about a 2-link planar robot.  The coordinator routes queries
to specialist FK and IK agents.
"""

from __future__ import annotations

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    pass

from roboarm.robots import create_two_link_planar

try:
    from roboarm.agents import RoboticsCoordinator
    HAS_AGENTS = True
except ImportError:
    HAS_AGENTS = False
    logger.warning("Agent layer not available; demo will be limited")


def main() -> None:
    """Run the AI agent demonstration."""
    logger.info("=== AI Agent Demo ===")

    # Create robot
    robot = create_two_link_planar(link1=1.0, link2=1.0)
    logger.info("Robot: %s", robot.name)

    if not HAS_AGENTS:
        logger.error(
            "RoboticsCoordinator is not available. "
            "Ensure the agents package is properly installed."
        )
        return

    # Create coordinator
    coordinator = RoboticsCoordinator(robot)
    logger.info("RoboticsCoordinator initialised.")

    # Define queries to process
    queries = [
        "Describe the robot",
        "Compute FK for angles [0.5, -0.3]",
        "Solve IK for x=1.0, y=0.5",
        "Compare all IK solvers for x=0.8, y=0.6",
    ]

    for query in queries:
        logger.info("")
        logger.info("=" * 70)
        logger.info("USER QUERY: %s", query)
        logger.info("=" * 70)

        try:
            response = coordinator.process(query)
            logger.info("")
            logger.info("AGENT RESPONSE:")
            for line in response.split("\n"):
                logger.info("  %s", line)
        except Exception as exc:
            logger.error("Error processing query: %s", exc)

    logger.info("")
    logger.info("=== Demo complete ===")


if __name__ == "__main__":
    main()
