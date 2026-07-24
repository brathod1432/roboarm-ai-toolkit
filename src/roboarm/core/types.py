"""Core data types for robot arm kinematics.

Provides dataclasses for DH parameters, joint configuration, end-effector
pose, and inverse kinematics solutions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DHParams:
    """Denavit-Hartenberg parameters for a single link.

    Attributes:
        alpha: Twist angle between joint axes (radians).
        a: Link length along common perpendicular.
        d: Link offset along joint axis.
        theta: Joint angle (radians) — variable for revolute joints.
        convention: DH convention, ``'standard'`` or ``'modified'``.

    Example::

        link = DHParams(alpha=0.0, a=1.0, d=0.0, theta=0.0)
    """

    alpha: float
    a: float
    d: float
    theta: float
    convention: str = "standard"


@dataclass(frozen=True)
class JointLimits:
    """Mechanical limits for a single joint.

    Attributes:
        lower: Minimum angle (radians).
        upper: Maximum angle (radians).
        velocity_max: Maximum angular velocity (rad/s), or ``None``.
        acceleration_max: Maximum angular acceleration (rad/s^2), or ``None``.
    """

    lower: float
    upper: float
    velocity_max: Optional[float] = None
    acceleration_max: Optional[float] = None


@dataclass
class JointConfig:
    """Full configuration for one joint in a serial chain.

    Attributes:
        dh_params: DH parameters describing the link geometry.
        limits: Optional mechanical limits.
        name: Human-readable joint name.
        is_variable: ``True`` for actuated joints, ``False`` for fixed offsets.
    """

    dh_params: DHParams
    limits: Optional[JointLimits] = None
    name: str = ""
    is_variable: bool = True


@dataclass
class EndEffectorPose:
    """End-effector position and orientation.

    Attributes:
        position: 3-element array ``[x, y, z]``.
        rotation: 3x3 rotation matrix.
        transform: 4x4 homogeneous transformation matrix.
    """

    position: np.ndarray
    rotation: np.ndarray
    transform: np.ndarray

    @property
    def x(self) -> float:
        """X coordinate of the end-effector."""
        return float(self.position[0])

    @property
    def y(self) -> float:
        """Y coordinate of the end-effector."""
        return float(self.position[1])

    @property
    def z(self) -> float:
        """Z coordinate of the end-effector."""
        return float(self.position[2])


@dataclass
class JointSolution:
    """A single set of joint angles.

    Attributes:
        values: Joint angles in radians as a numpy array.
    """

    values: np.ndarray


@dataclass
class IKSolution:
    """Result from an inverse kinematics solver.

    Attributes:
        success: Whether the solver converged to a valid solution.
        primary: Best joint-angle solution, or ``None`` on failure.
        alternatives: Other valid solutions found by the solver.
        iterations: Number of iterations used.
        residual_error: Final position error norm.
        computation_time_ms: Wall-clock solve time in milliseconds.
        solver_name: Name of the solver that produced this result.
        messages: Informational or warning messages from the solver.
    """

    success: bool
    primary: Optional[JointSolution] = None
    alternatives: List[JointSolution] = field(default_factory=list)
    iterations: int = 0
    residual_error: float = float("inf")
    computation_time_ms: float = 0.0
    solver_name: str = ""
    messages: List[str] = field(default_factory=list)


def list_to_array(values: Sequence[float]) -> np.ndarray:
    """Convert a sequence of floats to a 1-D numpy array.

    Args:
        values: Input sequence.

    Returns:
        1-D ``np.float64`` array.
    """
    return np.asarray(values, dtype=np.float64).ravel()
