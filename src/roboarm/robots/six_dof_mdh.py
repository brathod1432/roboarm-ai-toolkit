"""Six-DOF robot arm factory using Modified DH (Craig) convention.

Builds a serial-link robot with six revolute joints plus a fixed
tool-centre-point (TCP) offset, for a total of seven links and six
degrees of freedom.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from roboarm.core.robot import RobotArm
from roboarm.core.types import DHParams, JointConfig, JointLimits

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------
_DEG2RAD: float = math.pi / 180.0
_CM2M: float = 1.0 / 100.0

# ---------------------------------------------------------------------------
# Modified DH parameter table (Craig convention)
#
# | Link  | alpha(i-1) [deg] | a(i-1) [cm] | theta(i) [deg] | d(i) [cm] |
# |-------|------------------:|------------:|---------------:|----------:|
# | 0->1  |  0                |  0          | theta1         |  0        |
# | 1->2  | 90                |  0          | theta2         |  0        |
# | 2->3  |  0                | 15          | theta3         |  0        |
# | 3->4  | 90                |  7.2        | theta4         |  0        |
# | 4->5  | 90                |  0          | theta5         | 13.2      |
# | 5->6  | 90                |  0          | theta6         |  3        |
# | 6->TCP|  0                |  0          |  0 (fixed)     |  7.5      |
#
# Joint limits (degrees):
# J1: [-150, 150]  J2: [5, 175]  J3: [-90, 90]
# J4: [-90, 90]    J5: [90, 270]  J6: [-90, 90]
#
# Home pose (degrees): [0, 90, 0, 0, 180, 0]
# ---------------------------------------------------------------------------

_MDH_TABLE: list[dict] = [
    {"alpha_deg": 0.0, "a_cm": 0.0, "theta_deg": 0.0, "d_cm": 0.0},
    {"alpha_deg": 90.0, "a_cm": 0.0, "theta_deg": 0.0, "d_cm": 0.0},
    {"alpha_deg": 0.0, "a_cm": 15.0, "theta_deg": 0.0, "d_cm": 0.0},
    {"alpha_deg": 90.0, "a_cm": 7.2, "theta_deg": 0.0, "d_cm": 0.0},
    {"alpha_deg": 90.0, "a_cm": 0.0, "theta_deg": 0.0, "d_cm": 13.2},
    {"alpha_deg": 90.0, "a_cm": 0.0, "theta_deg": 0.0, "d_cm": 3.0},
    {"alpha_deg": 0.0, "a_cm": 0.0, "theta_deg": 0.0, "d_cm": 7.5},
]

_JOINT_LIMITS_DEG: list[dict] = [
    {"lower": -150.0, "upper": 150.0},
    {"lower": 5.0, "upper": 175.0},
    {"lower": -90.0, "upper": 90.0},
    {"lower": -90.0, "upper": 90.0},
    {"lower": 90.0, "upper": 270.0},
    {"lower": -90.0, "upper": 90.0},
]

HOME_POSE_DEG: list[float] = [0.0, 90.0, 0.0, 0.0, 180.0, 0.0]
"""Default home configuration in degrees."""

HOME_POSE_RAD: np.ndarray = np.array(
    [deg * _DEG2RAD for deg in HOME_POSE_DEG], dtype=np.float64
)
HOME_POSE_RAD.flags.writeable = False
"""Default home configuration in radians."""


def create_six_dof_mdh() -> RobotArm:
    """Create a 6-DOF robot arm using Modified DH parameters.

    The kinematic chain consists of seven links (six variable revolute
    joints plus one fixed TCP offset).  The ``n_joints`` property of the
    returned robot is 7 while ``n_dof`` is 6.

    Returns:
        Configured :class:`RobotArm` with 6 variable joints and 1 fixed
        TCP link.

    Example::

        robot = create_six_dof_mdh()
        home = [0.0, math.pi / 2, 0.0, 0.0, math.pi, 0.0]
        pose = robot.forward_kinematics(home)
    """
    joints: list[JointConfig] = []

    for idx, row in enumerate(_MDH_TABLE):
        alpha = row["alpha_deg"] * _DEG2RAD
        a = row["a_cm"] * _CM2M
        theta = row["theta_deg"] * _DEG2RAD
        d = row["d_cm"] * _CM2M

        dh = DHParams(
            alpha=alpha,
            a=a,
            d=d,
            theta=theta,
            convention="modified",
        )

        # Last link (6->TCP) is a fixed offset
        is_variable = idx < 6

        limits = None
        if is_variable:
            lim = _JOINT_LIMITS_DEG[idx]
            limits = JointLimits(
                lower=lim["lower"] * _DEG2RAD,
                upper=lim["upper"] * _DEG2RAD,
            )

        name = f"J{idx + 1}" if is_variable else "TCP"

        joints.append(
            JointConfig(
                dh_params=dh,
                limits=limits,
                name=name,
                is_variable=is_variable,
            )
        )

    logger.info(
        "Created 6-DOF MDH robot with %d joints (%d variable)",
        len(joints),
        sum(1 for j in joints if j.is_variable),
    )

    robot = RobotArm(joints, name="6-DOF MDH Robot")
    robot.save_pose("home", HOME_POSE_RAD)
    return robot
