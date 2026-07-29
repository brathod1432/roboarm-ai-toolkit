"""Trajectory planning utilities.

Re-exports all trajectory functions for convenient access::

    from roboarm.trajectory import (
        TrajectoryAnalyzer,
        TrajectoryValidator,
        cartesian_trajectory,
        cubic_interpolation,
        linear_interpolation,
        lspb,
        multi_joint_lspb,
        quintic_interpolation,
        save_trajectory_csv,
        load_trajectory_csv,
        via_point_trajectory,
    )
"""

from __future__ import annotations

from roboarm.trajectory.analysis import TrajectoryAnalyzer, TrajectoryMetrics
from roboarm.trajectory.cartesian import cartesian_trajectory
from roboarm.trajectory.interpolation import (
    cubic_interpolation,
    linear_interpolation,
    quintic_interpolation,
)
from roboarm.trajectory.io import (
    load_trajectory_csv,
    load_trajectory_npz,
    save_trajectory_csv,
    save_trajectory_npz,
)
from roboarm.trajectory.lspb import lspb, multi_joint_lspb
from roboarm.trajectory.multipoint import via_point_trajectory
from roboarm.trajectory.validation import (
    AccelerationViolation,
    LimitViolation,
    SafeZoneViolation,
    SingularityWarning,
    TrajectoryReport,
    TrajectoryValidator,
)

__all__ = [
    # Analysis
    "TrajectoryAnalyzer",
    "TrajectoryMetrics",
    # Validation
    "AccelerationViolation",
    "LimitViolation",
    "SafeZoneViolation",
    "SingularityWarning",
    "TrajectoryReport",
    "TrajectoryValidator",
    # Planning
    "cartesian_trajectory",
    "via_point_trajectory",
    # Interpolation
    "cubic_interpolation",
    "linear_interpolation",
    "lspb",
    "multi_joint_lspb",
    "quintic_interpolation",
    # I/O
    "load_trajectory_csv",
    "load_trajectory_npz",
    "save_trajectory_csv",
    "save_trajectory_npz",
]
