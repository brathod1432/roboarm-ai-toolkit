"""Three-link planar (redundant) robot arm factory."""

from __future__ import annotations

import math

from roboarm.core.robot import RobotArm
from roboarm.core.types import DHParams, JointConfig, JointLimits


def create_three_link_planar(
    link1: float = 1.0,
    link2: float = 1.0,
    link3: float = 0.5,
) -> RobotArm:
    """Create a 3-DOF planar RRR (redundant) robot arm.

    A planar arm with three revolute joints — redundant for 2-D positioning
    since it has more DOF than the task space dimension.

    Args:
        link1: Length of the first link.
        link2: Length of the second link.
        link3: Length of the third link.

    Returns:
        Configured :class:`RobotArm` with 3 revolute joints.

    Example::

        robot = create_three_link_planar()
        pose = robot.forward_kinematics([0.3, -0.5, 0.2])
    """
    joints = [
        JointConfig(
            dh_params=DHParams(alpha=0.0, a=link1, d=0.0, theta=0.0),
            limits=JointLimits(lower=-math.pi, upper=math.pi),
            name="J1",
        ),
        JointConfig(
            dh_params=DHParams(alpha=0.0, a=link2, d=0.0, theta=0.0),
            limits=JointLimits(lower=-math.pi, upper=math.pi),
            name="J2",
        ),
        JointConfig(
            dh_params=DHParams(alpha=0.0, a=link3, d=0.0, theta=0.0),
            limits=JointLimits(lower=-math.pi, upper=math.pi),
            name="J3",
        ),
    ]
    return RobotArm(joints, name=f"3-Link Planar (L={link1},{link2},{link3})")
