"""roboarm — Modular robot arm kinematics toolkit.

Primary classes are importable directly from the top-level package::

    from roboarm import RobotArm, RobotBuilder
    from roboarm import JacobianComputer, batch_fk, batch_ik
    from roboarm import TrajectoryValidator, TrajectoryAnalyzer
    from roboarm import RoboticsCoordinator

For the full public API see the subpackage documentation:
- :mod:`roboarm.core`         — RobotArm, types, exceptions
- :mod:`roboarm.kinematics`   — Jacobian, IK solvers, batch ops
- :mod:`roboarm.trajectory`   — planning, validation, analysis, I/O
- :mod:`roboarm.agents`       — AI coordinator, tools, audit log
- :mod:`roboarm.workspace`    — reachability analysis
- :mod:`roboarm.visualization` — 2-D/3-D/animation plots
"""

from __future__ import annotations

__version__ = "0.2.0"

# Core model
# Agents
from roboarm.agents.coordinator import RoboticsCoordinator
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

# Kinematics
from roboarm.kinematics.batch import batch_fk, batch_ik
from roboarm.kinematics.jacobian import JacobianComputer

# Trajectory
from roboarm.trajectory.analysis import TrajectoryAnalyzer, TrajectoryMetrics
from roboarm.trajectory.validation import TrajectoryReport, TrajectoryValidator

__all__ = [
    # Core
    "RobotArm",
    "RobotBuilder",
    "DHParams",
    "EndEffectorPose",
    "IKSolution",
    "JointConfig",
    "JointLimits",
    "JointSolution",
    # Exceptions
    "RobotArmError",
    "KinematicsError",
    "ConvergenceError",
    "ConfigurationError",
    "JointLimitError",
    "ValidationError",
    "WorkspaceError",
    # Kinematics
    "JacobianComputer",
    "batch_fk",
    "batch_ik",
    # Trajectory
    "TrajectoryAnalyzer",
    "TrajectoryMetrics",
    "TrajectoryReport",
    "TrajectoryValidator",
    # Agents
    "RoboticsCoordinator",
    # Package version
    "__version__",
]
