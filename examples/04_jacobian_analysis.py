"""Example 04: Jacobian analysis.

Demonstrates computing the geometric Jacobian at various configurations,
calculating the manipulability index, and detecting singularities for a
2-link planar robot.
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

from roboarm.robots import create_two_link_planar
from roboarm.kinematics.jacobian import JacobianComputer
from roboarm.utils.angle_utils import rad2deg


def main() -> None:
    """Run the Jacobian analysis demonstration."""
    logger.info("=== Jacobian Analysis Demo ===")

    # Create robot
    robot = create_two_link_planar(link1=1.0, link2=1.0)
    jc = JacobianComputer(robot)
    logger.info("Robot: %s (planar=%s)", robot.name, jc.is_planar)

    # Part 1: Jacobian at several configurations
    configurations = [
        ([0.0, 0.0], "Fully extended (singularity!)"),
        ([math.pi / 4, math.pi / 4], "Both joints at 45 deg"),
        ([math.pi / 2, 0.0], "Shoulder at 90 deg"),
        ([0.5, -0.3], "Arbitrary config"),
        ([0.0, math.pi], "Folded back (singularity!)"),
        ([math.pi / 4, -math.pi / 2], "Right-angle elbow"),
    ]

    logger.info("")
    logger.info("--- Jacobian Matrices and Manipulability ---")

    for angles, description in configurations:
        J = jc.compute(angles)
        J_num = jc.compute_numerical(angles)
        mu = jc.manipulability(angles)
        singular = jc.is_singular(angles)

        logger.info("")
        logger.info("Config: %s", description)
        logger.info(
            "  q = [%s] rad = [%s] deg",
            ", ".join(f"{a:.4f}" for a in angles),
            ", ".join(f"{rad2deg(a):.1f}" for a in angles),
        )
        logger.info("  Jacobian (geometric):")
        for row_idx in range(J.shape[0]):
            row_str = "  ".join(f"{v:+.6f}" for v in J[row_idx])
            logger.info("    [%s]", row_str)

        logger.info("  Manipulability: %.6e", mu)
        logger.info("  Near singularity: %s", singular)

        # Verify geometric vs numerical Jacobian agreement
        J_num_trimmed = J_num[:J.shape[0], :]
        max_diff = float(np.max(np.abs(J - J_num_trimmed)))
        logger.info(
            "  Geometric vs. numerical max diff: %.2e", max_diff,
        )

    # Part 2: Manipulability sweep
    logger.info("")
    logger.info("--- Manipulability Sweep (q2 from -pi to pi) ---")

    q2_values = np.linspace(-math.pi, math.pi, 37)
    mu_values = []

    for q2 in q2_values:
        q = [math.pi / 4, float(q2)]
        mu = jc.manipulability(q)
        mu_values.append(mu)

    # Report min and max
    min_mu = min(mu_values)
    max_mu = max(mu_values)
    min_idx = mu_values.index(min_mu)
    max_idx = mu_values.index(max_mu)

    logger.info(
        "  Min manipulability: %.6e at q2=%.2f deg",
        min_mu, rad2deg(float(q2_values[min_idx])),
    )
    logger.info(
        "  Max manipulability: %.6e at q2=%.2f deg",
        max_mu, rad2deg(float(q2_values[max_idx])),
    )

    # Part 3: Singularity detection at fully extended
    logger.info("")
    logger.info("--- Singularity Detection ---")
    extended_q = [0.0, 0.0]
    logger.info(
        "  Fully extended [0, 0]: singular=%s, mu=%.6e",
        jc.is_singular(extended_q),
        jc.manipulability(extended_q),
    )
    folded_q = [0.0, math.pi]
    logger.info(
        "  Folded back [0, pi]: singular=%s, mu=%.6e",
        jc.is_singular(folded_q),
        jc.manipulability(folded_q),
    )
    normal_q = [0.5, -1.0]
    logger.info(
        "  Normal config [0.5, -1.0]: singular=%s, mu=%.6e",
        jc.is_singular(normal_q),
        jc.manipulability(normal_q),
    )

    # Optional: plot manipulability
    if HAS_MATPLOTLIB:
        logger.info("")
        logger.info("Generating manipulability plot...")
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(
                np.degrees(q2_values), mu_values,
                "b-", linewidth=2,
            )
            ax.set_xlabel("q2 (degrees)")
            ax.set_ylabel("Manipulability")
            ax.set_title(
                "Manipulability vs. q2 (q1 fixed at 45 deg)"
            )
            ax.grid(True, alpha=0.3)
            ax.axhline(y=1e-4, color="r", linestyle="--", alpha=0.5,
                       label="Singularity threshold")
            ax.legend()
            fig.tight_layout()
            fig.savefig("jacobian_manipulability.png", dpi=100)
            logger.info("Plot saved to jacobian_manipulability.png")
        except Exception as exc:
            logger.warning("Could not generate plot: %s", exc)

    logger.info("=== Demo complete ===")


if __name__ == "__main__":
    main()
