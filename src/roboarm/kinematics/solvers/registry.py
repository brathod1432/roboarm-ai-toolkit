"""IK solver registry for dynamic solver discovery and instantiation.

Provides :class:`IKSolverRegistry`, a class-level registry that maps
human-readable names to :class:`IKSolverBase` subclasses so that solvers
can be created by name at runtime.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Type

from roboarm.core.exceptions import ConfigurationError
from roboarm.core.robot import RobotArm
from roboarm.kinematics.inverse import IKSolverBase

logger = logging.getLogger(__name__)


class IKSolverRegistry:
    """Central registry that maps solver names to their classes.

    Solver modules register themselves on import using the
    :meth:`register` decorator::

        @IKSolverRegistry.register("my_solver")
        class MySolver(IKSolverBase):
            ...

    Callers can then create solvers by name::

        solver = IKSolverRegistry.create("my_solver", robot)
    """

    _registry: Dict[str, Type[IKSolverBase]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[IKSolverBase]], Type[IKSolverBase]]:
        """Decorator that registers a solver class under *name*.

        Args:
            name: Unique solver identifier (e.g. ``"damped_least_squares"``).

        Returns:
            A class decorator that records the solver in the registry.

        Raises:
            ConfigurationError: If *name* is already registered.
        """

        def decorator(solver_cls: Type[IKSolverBase]) -> Type[IKSolverBase]:
            if name in cls._registry:
                raise ConfigurationError(
                    f"IK solver '{name}' is already registered "
                    f"({cls._registry[name].__name__})"
                )
            cls._registry[name] = solver_cls
            logger.debug("Registered IK solver '%s' -> %s", name, solver_cls.__name__)
            return solver_cls

        return decorator

    @classmethod
    def create(cls, name: str, robot: RobotArm, **kwargs: object) -> IKSolverBase:
        """Instantiate a registered solver by name.

        Args:
            name: Registered solver identifier.
            robot: Robot arm model to pass to the solver constructor.
            **kwargs: Additional keyword arguments forwarded to the
                solver constructor.

        Returns:
            A ready-to-use :class:`IKSolverBase` instance.

        Raises:
            ConfigurationError: If *name* is not registered.
        """
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry)) or "(none)"
            raise ConfigurationError(
                f"Unknown IK solver '{name}'. Available: {available}"
            )
        solver_cls = cls._registry[name]
        logger.info("Creating IK solver '%s' for %s", name, robot.name)
        return solver_cls(robot, **kwargs)

    @classmethod
    def available(cls) -> List[str]:
        """Return a sorted list of all registered solver names."""
        return sorted(cls._registry)
