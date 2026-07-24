"""IK solver implementations.

Importing this package triggers auto-registration of all built-in
solvers with the :class:`IKSolverRegistry`.
"""

from __future__ import annotations

# Import each solver module so the ``@IKSolverRegistry.register``
# decorators execute and populate the registry.
from roboarm.kinematics.solvers import (  # noqa: F401
    analytical,
    ccd,
    damped_least_squares,
    fabrik,
    jacobian_ik,
)
from roboarm.kinematics.solvers.registry import IKSolverRegistry

__all__ = ["IKSolverRegistry"]
