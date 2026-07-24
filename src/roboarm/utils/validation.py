"""Input validation helpers for the roboarm toolkit."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

from roboarm.core.exceptions import ValidationError
from roboarm.core.types import JointConfig

logger = logging.getLogger(__name__)


def validate_joint_angles(
    q: Sequence[float],
    n_dof: int,
    joints: Optional[List[JointConfig]] = None,
) -> np.ndarray:
    """Validate and convert joint angles to a numpy array.

    Args:
        q: Joint angle values (radians).
        n_dof: Expected number of degrees of freedom.
        joints: Optional joint configs for limit checking.

    Returns:
        Validated 1-D float array.

    Raises:
        ValidationError: If length mismatch or limits exceeded.
    """
    arr = np.asarray(q, dtype=np.float64).ravel()
    if arr.size != n_dof:
        raise ValidationError(
            f"Expected {n_dof} joint values, got {arr.size}"
        )
    if joints is not None:
        variable_joints = [j for j in joints if j.is_variable]
        for i, jc in enumerate(variable_joints):
            if jc.limits is not None:
                if arr[i] < jc.limits.lower - 1e-9 or arr[i] > jc.limits.upper + 1e-9:
                    logger.warning(
                        "Joint %d (%s) angle %.4f outside limits [%.4f, %.4f]",
                        i, jc.name, arr[i], jc.limits.lower, jc.limits.upper,
                    )
    return arr


def validate_position_target(
    target: Sequence[float],
    dimensions: int = 3,
) -> np.ndarray:
    """Validate a Cartesian position target.

    Args:
        target: Target position ``[x, y, ...]``.
        dimensions: Expected dimensionality (2 or 3).

    Returns:
        Validated position array.

    Raises:
        ValidationError: If dimensionality is wrong or values are not finite.
    """
    arr = np.asarray(target, dtype=np.float64).ravel()
    if arr.size != dimensions:
        raise ValidationError(
            f"Expected {dimensions}-D position, got {arr.size} values"
        )
    if not np.all(np.isfinite(arr)):
        raise ValidationError("Position target contains non-finite values")
    return arr
