"""Kinematics sub-package for forward/inverse kinematics and Jacobians.

Provides:

* :func:`compute_fk` -- convenience wrapper for forward kinematics.
* :class:`JacobianComputer` -- geometric and numerical Jacobian.
* :class:`IKSolverBase` / :class:`IKConfig` -- IK solver abstractions.
* :class:`IKSolverRegistry` -- solver discovery and creation by name.
"""

from __future__ import annotations

from roboarm.kinematics.forward import compute_fk
from roboarm.kinematics.inverse import IKConfig, IKSolverBase
from roboarm.kinematics.jacobian import JacobianComputer
from roboarm.kinematics.solvers.registry import IKSolverRegistry

__all__ = [
    "IKConfig",
    "IKSolverBase",
    "IKSolverRegistry",
    "JacobianComputer",
    "compute_fk",
]
