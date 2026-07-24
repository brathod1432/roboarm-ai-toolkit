"""Custom exception hierarchy for the roboarm toolkit."""

from __future__ import annotations


class RobotArmError(Exception):
    """Base exception for all roboarm errors."""


class KinematicsError(RobotArmError):
    """Error during a kinematics computation (FK or IK)."""


class JointLimitError(RobotArmError):
    """A joint angle exceeds its mechanical limits."""


class WorkspaceError(RobotArmError):
    """A target position is outside the reachable workspace."""


class ConvergenceError(KinematicsError):
    """An iterative solver failed to converge."""


class ConfigurationError(RobotArmError):
    """Invalid robot or solver configuration."""


class ValidationError(RobotArmError):
    """Input validation failure."""
