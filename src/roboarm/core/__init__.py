"""Core mathematics and robot model.

Re-exports key classes for convenient access::

    from roboarm.core import RobotArm, DHParams, EndEffectorPose
"""

from __future__ import annotations

from roboarm.core.builder import RobotBuilder
from roboarm.core.exceptions import (
    ConfigurationError,
    ConvergenceError,
    JointLimitError,
    KinematicsError,
    RobotArmError,
    ValidationError,
    WorkspaceError,
)
from roboarm.core.robot import RobotArm
from roboarm.core.types import (
    DHParams,
    EndEffectorPose,
    IKSolution,
    JointConfig,
    JointLimits,
    JointSolution,
)

__all__ = [
    "RobotArm",
    "RobotBuilder",
    "DHParams",
    "EndEffectorPose",
    "IKSolution",
    "JointConfig",
    "JointLimits",
    "JointSolution",
    "RobotArmError",
    "KinematicsError",
    "ConvergenceError",
    "ConfigurationError",
    "JointLimitError",
    "ValidationError",
    "WorkspaceError",
]
