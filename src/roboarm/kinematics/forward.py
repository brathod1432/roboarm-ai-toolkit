"""Forward kinematics convenience wrapper.

Re-exports the forward kinematics computation from :class:`RobotArm` so that
callers can use a functional interface without coupling to the model class.
"""

from __future__ import annotations

import logging
from typing import Sequence

from roboarm.core.robot import RobotArm
from roboarm.core.types import EndEffectorPose

logger = logging.getLogger(__name__)


def compute_fk(robot: RobotArm, q: Sequence[float]) -> EndEffectorPose:
    """Compute forward kinematics for a robot arm.

    This is a thin convenience wrapper around
    :meth:`RobotArm.forward_kinematics` that provides a functional API.

    Args:
        robot: The robot arm model.
        q: Joint angles in radians (length must equal ``robot.n_dof``).

    Returns:
        End-effector position and orientation.

    Raises:
        ValidationError: If *q* has the wrong length.

    Example::

        from roboarm.robots.two_link_planar import create_two_link_planar
        robot = create_two_link_planar()
        pose = compute_fk(robot, [0.5, -0.3])
        print(pose.x, pose.y)
    """
    logger.debug(
        "Computing FK for %s with q=%s", robot.name, list(q),
    )
    return robot.forward_kinematics(q)
