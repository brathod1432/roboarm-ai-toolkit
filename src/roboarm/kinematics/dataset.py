"""Labeled dataset generation for ML training and research.

Generates (configuration, FK_pose, Jacobian, manipulability) tuples
in batches, stored as a dictionary of numpy arrays for easy downstream use.
"""

from __future__ import annotations

import logging

import numpy as np

from roboarm.core.robot import RobotArm

logger = logging.getLogger(__name__)


def generate_dataset(
    robot: RobotArm,
    n_samples: int = 1000,
    seed: int | None = None,
    include_jacobian: bool = True,
    include_manipulability: bool = True,
    q_range: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Generate a labeled dataset of robot configurations and FK results.

    Samples *n_samples* random joint configurations (uniformly within each
    joint's limits, or within *q_range* if set), computes FK at each, and
    optionally adds Jacobian and manipulability values.

    The returned dictionary has consistent first-axis length *n_samples* and
    can be saved directly with ``np.savez_compressed("dataset.npz", **data)``.

    Args:
        robot: The robot arm model.
        n_samples: Number of samples to generate.
        seed: Optional RNG seed for reproducibility.
        include_jacobian: If ``True`` (default), include the flattened
            Jacobian matrix at each configuration.
        include_manipulability: If ``True`` (default), include the Yoshikawa
            manipulability index at each configuration.
        q_range: If given, sample all joints uniformly from
            ``(q_range[0], q_range[1])`` instead of their individual limits.
            Useful for robots without explicit joint limits.

    Returns:
        Dictionary with the following keys (all ``float64`` arrays):

        * ``"q"`` — ``(n_samples, n_dof)`` joint configurations.
        * ``"position"`` — ``(n_samples, 3)`` end-effector positions [x,y,z].
        * ``"rotation_flat"`` — ``(n_samples, 9)`` flattened 3×3 rotation matrices.
        * ``"jacobian_flat"`` — ``(n_samples, rows*n_dof)`` flattened Jacobians
          (only present if *include_jacobian* is ``True``).
        * ``"manipulability"`` — ``(n_samples,)`` Yoshikawa indices (only present
          if *include_manipulability* is ``True``).

    Example::

        data = generate_dataset(robot, n_samples=10_000, seed=42)
        np.savez_compressed("training_data.npz", **data)
        # Load: arc = np.load("training_data.npz"); q = arc["q"]
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")

    rng = np.random.default_rng(seed)
    n_dof = robot.n_dof
    limits = robot.joint_limits

    # Pre-allocate arrays
    q_data = np.empty((n_samples, n_dof), dtype=np.float64)
    pos_data = np.empty((n_samples, 3), dtype=np.float64)
    rot_data = np.empty((n_samples, 9), dtype=np.float64)

    jac_rows = None
    jac_data = None
    mu_data = None

    jac_computer = None
    if include_jacobian or include_manipulability:
        from roboarm.kinematics.jacobian import JacobianComputer
        jac_computer = JacobianComputer(robot)

    for i in range(n_samples):
        # Sample configuration
        q = np.empty(n_dof, dtype=np.float64)
        for j in range(n_dof):
            if q_range is not None:
                q[j] = rng.uniform(q_range[0], q_range[1])
            else:
                lim = limits[j]
                lo = lim.lower if lim is not None else -np.pi
                hi = lim.upper if lim is not None else np.pi
                q[j] = rng.uniform(lo, hi)

        q_data[i] = q

        # FK
        pose = robot.forward_kinematics(q)
        pos_data[i] = pose.position
        rot_data[i] = pose.rotation.ravel()

        # Jacobian / manipulability
        if jac_computer is not None:
            J = jac_computer.compute(q)
            if include_jacobian:
                if jac_rows is None:
                    jac_rows = J.shape[0]
                    jac_data = np.empty(
                        (n_samples, jac_rows * n_dof), dtype=np.float64
                    )
                jac_data[i] = J.ravel()
            if include_manipulability:
                if mu_data is None:
                    mu_data = np.empty(n_samples, dtype=np.float64)
                mu_data[i] = jac_computer.manipulability(q)

    result: dict[str, np.ndarray] = {
        "q": q_data,
        "position": pos_data,
        "rotation_flat": rot_data,
    }
    if jac_data is not None:
        result["jacobian_flat"] = jac_data
    if mu_data is not None:
        result["manipulability"] = mu_data

    logger.info(
        "Generated dataset: %d samples, %d DOF, keys=%s",
        n_samples, n_dof, sorted(result.keys()),
    )
    return result
