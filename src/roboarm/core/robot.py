"""Serial-link robot arm model with forward kinematics.

Defines :class:`RobotArm`, the central model class that holds a kinematic
chain and computes forward kinematics by chaining DH transforms.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from roboarm.core.exceptions import ValidationError
from roboarm.core.transform import (
    chain_transforms,
    extract_position,
    extract_rotation,
    transform_from_dh_params,
)
from roboarm.core.types import (
    EndEffectorPose,
    JointConfig,
    JointLimits,
)

logger = logging.getLogger(__name__)


class RobotArm:
    """Serial-link robot arm built from a chain of DH-parameterised joints.

    Example::

        from roboarm.core.types import DHParams, JointConfig
        j1 = JointConfig(DHParams(alpha=0, a=1.0, d=0, theta=0))
        j2 = JointConfig(DHParams(alpha=0, a=1.0, d=0, theta=0))
        robot = RobotArm([j1, j2], name="2-Link Planar")
        pose = robot.forward_kinematics([0.5, -0.3])
    """

    def __init__(
        self,
        joints: list[JointConfig],
        name: str = "Robot",
    ) -> None:
        """Initialise the robot arm.

        Args:
            joints: Ordered list of joint configurations (base to tip).
            name: Human-readable robot name.
        """
        if not joints:
            raise ValidationError("Robot must have at least one joint")
        self._joints = list(joints)
        self.name = name

    @property
    def joints(self) -> list[JointConfig]:
        """All joints in the chain."""
        return list(self._joints)

    @property
    def n_joints(self) -> int:
        """Total number of joints (including fixed offsets)."""
        return len(self._joints)

    @property
    def n_dof(self) -> int:
        """Number of actuated (variable) degrees of freedom."""
        return sum(1 for j in self._joints if j.is_variable)

    @property
    def joint_names(self) -> list[str]:
        """Names of all variable joints."""
        return [j.name or f"J{i}" for i, j in enumerate(self._joints) if j.is_variable]

    @property
    def joint_limits(self) -> list[JointLimits | None]:
        """Limits for each variable joint."""
        return [j.limits for j in self._joints if j.is_variable]

    def forward_kinematics(self, q: Sequence[float]) -> EndEffectorPose:
        """Compute end-effector pose from joint angles.

        Args:
            q: Joint angles in radians (length must equal :attr:`n_dof`).

        Returns:
            End-effector position and orientation.

        Raises:
            ValidationError: If ``q`` has the wrong length.
        """
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        if q_arr.size != self.n_dof:
            raise ValidationError(
                f"Expected {self.n_dof} joint angles, got {q_arr.size}"
            )

        transforms = self._compute_link_transforms(q_arr)
        T = chain_transforms(transforms)
        return EndEffectorPose(
            position=extract_position(T),
            rotation=extract_rotation(T),
            transform=T,
        )

    def joint_transforms(self, q: Sequence[float]) -> list[np.ndarray]:
        """Compute the cumulative transform up to each joint frame.

        The returned list has ``n_joints + 1`` entries: ``[T_base, T_01,
        T_02, ..., T_0n]`` where ``T_base`` is the identity.

        Args:
            q: Joint angles in radians.

        Returns:
            List of 4x4 cumulative transforms.
        """
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        if q_arr.size != self.n_dof:
            raise ValidationError(
                f"Expected {self.n_dof} joint angles, got {q_arr.size}"
            )

        link_transforms = self._compute_link_transforms(q_arr)
        cumulative = [np.eye(4, dtype=np.float64)]
        T = np.eye(4, dtype=np.float64)
        for lt in link_transforms:
            T = T @ lt
            cumulative.append(T.copy())
        return cumulative

    def joint_positions(self, q: Sequence[float]) -> np.ndarray:
        """Compute the 3-D position of every joint origin.

        Args:
            q: Joint angles in radians.

        Returns:
            ``(n_joints + 1, 3)`` array of positions (base through TCP).
        """
        cumulative = self.joint_transforms(q)
        return np.array([extract_position(T) for T in cumulative])

    def _compute_link_transforms(self, q: np.ndarray) -> list[np.ndarray]:
        """Build per-link transforms inserting variable joint angles.

        Args:
            q: Array of variable joint angles (length == n_dof).

        Returns:
            List of 4x4 per-link transforms.
        """
        transforms: list[np.ndarray] = []
        q_idx = 0
        for jc in self._joints:
            if jc.is_variable:
                T = transform_from_dh_params(jc.dh_params, q[q_idx])
                q_idx += 1
            else:
                T = transform_from_dh_params(jc.dh_params, 0.0)
            transforms.append(T)
        return transforms

    def __repr__(self) -> str:
        return f"RobotArm(name={self.name!r}, n_joints={self.n_joints}, n_dof={self.n_dof})"
