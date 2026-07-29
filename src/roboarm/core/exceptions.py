"""Custom exception hierarchy for the roboarm toolkit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roboarm.core.types import JointSolution


class RobotArmError(Exception):
    """Base exception for all roboarm errors."""


class KinematicsError(RobotArmError):
    """Error during a kinematics computation (FK or IK)."""


class IKFailedError(KinematicsError):
    """Raised by :meth:`~roboarm.core.robot.RobotArm.ik` when the solver
    fails to converge.

    Unlike ``KinematicsError``, this subclass carries the solver's
    **best attempt** — the closest joint configuration it reached before
    giving up.  Callers that can accept an approximate solution may inspect
    :attr:`best_attempt` before deciding to fail.

    Attributes:
        residual_error: Final position-error norm at the failed configuration.
        best_attempt: Closest :class:`~roboarm.core.types.JointSolution`
            reached, or ``None`` if no iteration was performed.
        solver_name: Name of the IK solver that was used.
    """

    def __init__(
        self,
        message: str,
        residual_error: float = float("inf"),
        best_attempt: JointSolution | None = None,
        solver_name: str = "",
    ) -> None:
        super().__init__(message)
        self.residual_error = residual_error
        self.best_attempt = best_attempt
        self.solver_name = solver_name


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
