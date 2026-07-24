"""Abstract base class and configuration for inverse kinematics solvers.

Defines :class:`IKSolverBase` -- the interface every IK solver must
implement -- and :class:`IKConfig`, a shared configuration container for
common solver parameters such as tolerances and iteration limits.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from roboarm.core.robot import RobotArm
from roboarm.core.types import EndEffectorPose, IKSolution

logger = logging.getLogger(__name__)


class IKSolverBase(ABC):
    """Abstract base class for inverse kinematics solvers.

    Subclasses must implement :meth:`solve` and may override
    :attr:`name` if the default (the class name) is not descriptive
    enough.

    Args:
        robot: The robot arm model to solve for.
        **kwargs: Solver-specific keyword arguments.
    """

    def __init__(self, robot: RobotArm, **kwargs: object) -> None:
        self._robot = robot
        logger.debug(
            "Initialised IK solver %s for %s", self.name, robot.name,
        )

    @abstractmethod
    def solve(
        self,
        target: EndEffectorPose,
        q0: Sequence[float] | None = None,
    ) -> IKSolution:
        """Solve the inverse kinematics problem.

        Args:
            target: Desired end-effector pose.
            q0: Optional initial joint-angle guess.  When ``None`` the
                solver should choose a reasonable default (e.g. zeros).

        Returns:
            An :class:`IKSolution` describing success/failure, the best
            joint-angle solution, and solver diagnostics.
        """

    @property
    def name(self) -> str:
        """Human-readable solver name (defaults to the class name)."""
        return self.__class__.__name__


@dataclass
class IKConfig:
    """Common configuration shared by iterative IK solvers.

    Attributes:
        max_iterations: Maximum number of solver iterations.
        tolerance: Convergence tolerance on the position-error norm.
        damping: Damping factor for damped-least-squares methods.
        step_size: Scaling factor applied to each update step.
        joint_limit_margin: Safety margin (radians) to keep away from
            mechanical joint limits.
    """

    max_iterations: int = 500
    tolerance: float = 1e-6
    damping: float = 0.01
    step_size: float = 1.0
    joint_limit_margin: float = 0.01
