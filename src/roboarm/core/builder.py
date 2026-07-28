"""Fluent robot builder DSL.

Provides :class:`RobotBuilder` for constructing a :class:`RobotArm` from
DH parameters using a readable method-chain API instead of directly
instantiating dataclasses.

Example::

    from roboarm.core.builder import RobotBuilder

    robot = (
        RobotBuilder("My 4-DOF Arm")
        .add_revolute(a=0.4, limits=(-180, 180), name="Base")
        .add_revolute(a=0.3, limits=(-90, 90),   name="Shoulder")
        .add_revolute(a=0.2, limits=(-90, 90),   name="Elbow")
        .add_revolute(a=0.1, limits=(-45, 45),   name="Wrist")
        .build()
    )
"""

from __future__ import annotations

import logging
import math

from roboarm.core.robot import RobotArm
from roboarm.core.types import DHParams, JointConfig, JointLimits

logger = logging.getLogger(__name__)


class RobotBuilder:
    """Fluent builder for :class:`~roboarm.core.robot.RobotArm`.

    Args:
        name: Human-readable robot name.

    Example::

        robot = (
            RobotBuilder("2-DOF Planar")
            .add_revolute(a=1.0, name="J1")
            .add_revolute(a=0.8, name="J2")
            .build()
        )
    """

    def __init__(self, name: str = "Robot") -> None:
        self._name = name
        self._joints: list[JointConfig] = []

    def add_revolute(
        self,
        a: float = 0.0,
        d: float = 0.0,
        alpha: float = 0.0,
        theta_offset: float = 0.0,
        convention: str = "standard",
        limits: tuple[float, float] | None = None,
        velocity_max: float | None = None,
        name: str = "",
        limits_in_degrees: bool = True,
    ) -> RobotBuilder:
        """Append a revolute (rotary) joint to the chain.

        Args:
            a: Link length along the common perpendicular (metres).
            d: Link offset along the joint axis (metres).
            alpha: Twist angle between joint axes (degrees if
                *limits_in_degrees* else radians).
            theta_offset: Fixed angular offset on the joint angle
                (degrees if *limits_in_degrees* else radians).
            convention: DH convention — ``"standard"`` or ``"modified"``.
            limits: ``(lower, upper)`` joint angle limits.  Values are
                in degrees when *limits_in_degrees* is ``True`` (default).
                Pass ``None`` for unlimited joints.
            velocity_max: Maximum angular velocity in rad/s, or ``None``.
            name: Human-readable joint name.
            limits_in_degrees: If ``True`` (default), *limits* and
                *alpha* / *theta_offset* are interpreted as degrees and
                converted to radians internally.

        Returns:
            *self* for method chaining.
        """
        def to_rad(v: float) -> float:
            return math.radians(v) if limits_in_degrees else v

        dh = DHParams(
            alpha=to_rad(alpha),
            a=float(a),
            d=float(d),
            theta=to_rad(theta_offset),
            convention=convention,
        )
        lim: JointLimits | None = None
        if limits is not None:
            lower, upper = limits
            lim = JointLimits(
                lower=to_rad(lower),
                upper=to_rad(upper),
                velocity_max=velocity_max,
            )
        self._joints.append(
            JointConfig(dh_params=dh, limits=lim, name=name, is_variable=True)
        )
        return self

    def add_fixed(
        self,
        a: float = 0.0,
        d: float = 0.0,
        alpha: float = 0.0,
        theta: float = 0.0,
        convention: str = "standard",
        name: str = "TCP",
        in_degrees: bool = True,
    ) -> RobotBuilder:
        """Append a fixed (non-actuated) link — typically a TCP offset.

        Args:
            a: Link length (metres).
            d: Link offset (metres).
            alpha: Twist angle (degrees if *in_degrees* else radians).
            theta: Fixed joint angle (degrees if *in_degrees* else radians).
            convention: DH convention.
            name: Link name.
            in_degrees: If ``True``, angular values are degrees.

        Returns:
            *self* for method chaining.
        """
        def to_rad(v: float) -> float:
            return math.radians(v) if in_degrees else v

        dh = DHParams(
            alpha=to_rad(alpha),
            a=float(a),
            d=float(d),
            theta=to_rad(theta),
            convention=convention,
        )
        self._joints.append(
            JointConfig(dh_params=dh, limits=None, name=name, is_variable=False)
        )
        return self

    def build(self) -> RobotArm:
        """Construct and return the :class:`~roboarm.core.robot.RobotArm`.

        Returns:
            A fully configured :class:`RobotArm` instance.

        Raises:
            ValueError: If no joints have been added.
        """
        if not self._joints:
            raise ValueError("Add at least one joint before calling build().")
        robot = RobotArm(list(self._joints), name=self._name)
        logger.info("RobotBuilder: built %r (%d DOF)", self._name, robot.n_dof)
        return robot

    def __repr__(self) -> str:
        return f"RobotBuilder(name={self._name!r}, joints={len(self._joints)})"
