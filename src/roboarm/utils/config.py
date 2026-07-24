"""Configuration loader and default parameter sets.

Provides sensible defaults for inverse-kinematics solvers and a
helper to merge user overrides into those defaults.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------

DEFAULT_IK_CONFIG: dict[str, Any] = {
    "max_iterations": 500,
    "tolerance": 1e-6,
    "damping": 0.5,
    "step_size": 1.0,
}
"""Default parameters for iterative IK solvers."""

DEFAULT_TRAJECTORY_CONFIG: dict[str, Any] = {
    "n_steps": 100,
    "t_total": 2.0,
    "interpolation": "cubic",
}
"""Default parameters for trajectory generation."""

DEFAULT_VISUALIZATION_CONFIG: dict[str, Any] = {
    "figure_size": (8, 8),
    "link_color": "steelblue",
    "joint_color": "navy",
    "ee_color": "crimson",
    "grid_alpha": 0.3,
}
"""Default parameters for visualisation."""


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------


def get_config(
    overrides: dict[str, Any] | None = None,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge user overrides with a base configuration.

    The base defaults to :data:`DEFAULT_IK_CONFIG` if not provided.
    A deep copy of the base is made before merging so the module-level
    defaults are never mutated.

    Args:
        overrides: Key-value pairs to override in the base config.
            ``None`` values are ignored.
        base: Base configuration dictionary.  Defaults to
            :data:`DEFAULT_IK_CONFIG`.

    Returns:
        Merged configuration dictionary.

    Example::

        cfg = get_config({"tolerance": 1e-4, "max_iterations": 1000})
        # cfg == {"max_iterations": 1000, "tolerance": 1e-4,
        #         "damping": 0.5, "step_size": 1.0}
    """
    if base is None:
        base = DEFAULT_IK_CONFIG

    config = copy.deepcopy(base)

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value

    logger.debug("Resolved configuration: %s", config)
    return config
