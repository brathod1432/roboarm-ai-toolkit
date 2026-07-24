"""Two-link planar robot arm factory."""

from __future__ import annotations

import math

from roboarm.core.robot import RobotArm
from roboarm.core.types import DHParams, JointConfig, JointLimits


def create_two_link_planar(
    link1: float = 1.0,
    link2: float = 1.0,
) -> RobotArm:
    """Create a 2-DOF planar RR robot arm.

    Uses standard DH parameters with all joints rotating in the XY plane.

    Args:
        link1: Length of the first link.
        link2: Length of the second link.

    Returns:
        Configured :class:`RobotArm` with 2 revolute joints.

    Example::

        robot = create_two_link_planar(link1=1.0, link2=0.8)
        pose = robot.forward_kinematics([0.5, -0.3])
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
    ]
    return RobotArm(joints, name=f"2-Link Planar (L1={link1}, L2={link2})")
