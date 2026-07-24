"""Integration tests for the 6-DOF Modified-DH robot model.

Covers structural properties, forward kinematics at known configurations,
Jacobian correctness, and input validation for the 6-DOF serial-link arm
built from Modified DH (Craig convention) parameters.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from roboarm.core.transform import is_valid_transform
from roboarm.core.types import EndEffectorPose
from roboarm.kinematics.jacobian import JacobianComputer
from roboarm.robots.six_dof_mdh import create_six_dof_mdh


# ======================================================================
# 6-DOF robot structural properties
# ======================================================================


class TestSixDOFProperties:
    """6-DOF robot model structural properties."""

    def test_joint_count(self) -> None:
        """Total joints: 6 revolute + 1 fixed TCP = 7."""
        robot = create_six_dof_mdh()
        assert robot.n_joints == 7

    def test_dof_count(self) -> None:
        """Actuated degrees of freedom must be 6."""
        robot = create_six_dof_mdh()
        assert robot.n_dof == 6

    def test_has_7_joint_configs(self) -> None:
        """The joints list must contain exactly 7 entries."""
        robot = create_six_dof_mdh()
        assert len(robot.joints) == 7

    def test_last_joint_is_fixed(self) -> None:
        """The 7th joint (TCP offset) must be marked as fixed."""
        robot = create_six_dof_mdh()
        assert robot.joints[-1].is_variable is False

    def test_first_6_joints_are_variable(self) -> None:
        """Joints 1-6 must all be variable (actuated)."""
        robot = create_six_dof_mdh()
        for idx, jc in enumerate(robot.joints[:6]):
            assert jc.is_variable is True, (
                f"Joint {idx} should be variable"
            )

    def test_all_joints_use_mdh(self) -> None:
        """Every joint must use the 'modified' DH convention."""
        robot = create_six_dof_mdh()
        for idx, jc in enumerate(robot.joints):
            assert jc.dh_params.convention == "modified", (
                f"Joint {idx} convention is {jc.dh_params.convention!r}"
            )


# ======================================================================
# 6-DOF forward kinematics at known configurations
# ======================================================================


class TestSixDOFFK:
    """6-DOF FK at known configurations."""

    def test_home_pose_position(self) -> None:
        """FK at home pose [0, 90, 0, 0, 180, 0] deg must produce a
        finite position within reasonable reach (~0.4 m max)."""
        robot = create_six_dof_mdh()
        home_rad = np.radians([0, 90, 0, 0, 180, 0]).tolist()
        pose = robot.forward_kinematics(home_rad)

        assert len(pose.position) == 3
        assert np.all(np.isfinite(pose.position))
        assert np.linalg.norm(pose.position) < 1.0  # less than 1 metre

    def test_all_zeros_position(self) -> None:
        """FK at all-zero angles must produce a finite position."""
        robot = create_six_dof_mdh()
        pose = robot.forward_kinematics([0.0] * 6)
        assert np.all(np.isfinite(pose.position))

    def test_valid_transform_at_home(self) -> None:
        """FK transform at home pose must be a valid SE(3) matrix."""
        robot = create_six_dof_mdh()
        home_rad = np.radians([0, 90, 0, 0, 180, 0]).tolist()
        pose = robot.forward_kinematics(home_rad)
        assert is_valid_transform(pose.transform), (
            "Home-pose transform is not valid SE(3)"
        )

    def test_valid_transform_at_random_configs(self) -> None:
        """FK transforms at 20 random configs must all be valid SE(3)."""
        robot = create_six_dof_mdh()
        rng = np.random.default_rng(seed=42)
        for trial in range(20):
            q = rng.uniform(-1.5, 1.5, size=6).tolist()
            pose = robot.forward_kinematics(q)
            assert is_valid_transform(pose.transform), (
                f"Invalid SE(3) at trial {trial}, q={q}"
            )

    def test_joint_positions_shape(self) -> None:
        """joint_positions must return (8, 3) for a 7-joint chain
        (7 joints + 1 base frame = 8 frames)."""
        robot = create_six_dof_mdh()
        home_rad = np.radians([0, 90, 0, 0, 180, 0]).tolist()
        positions = robot.joint_positions(home_rad)
        assert positions.shape == (8, 3), (
            f"Expected (8, 3), got {positions.shape}"
        )

    def test_base_at_origin(self) -> None:
        """The first frame (base) must be at the origin."""
        robot = create_six_dof_mdh()
        positions = robot.joint_positions([0.0] * 6)
        np.testing.assert_allclose(
            positions[0], [0.0, 0.0, 0.0], atol=1e-10,
        )

    def test_home_and_zeros_differ(self) -> None:
        """Home pose and all-zeros must produce different positions."""
        robot = create_six_dof_mdh()
        pose_home = robot.forward_kinematics(
            np.radians([0, 90, 0, 0, 180, 0]).tolist(),
        )
        pose_zero = robot.forward_kinematics([0.0] * 6)
        assert not np.allclose(
            pose_home.position, pose_zero.position, atol=1e-6,
        ), "Home and zero poses should differ"


# ======================================================================
# 6-DOF Jacobian tests
# ======================================================================


class TestSixDOFJacobian:
    """6-DOF Jacobian shape and numerical agreement."""

    def test_jacobian_shape(self) -> None:
        """Geometric Jacobian of a 6-DOF arm must be (6, 6)."""
        robot = create_six_dof_mdh()
        jac = JacobianComputer(robot)
        jacobian = jac.compute([0.0] * 6)
        assert jacobian.shape == (6, 6), (
            f"Expected (6, 6), got {jacobian.shape}"
        )

    def test_numerical_jacobian_velocity_prediction(self) -> None:
        """Numerical Jacobian-predicted position change must match FK
        finite differences for small joint perturbations.

        The numerical Jacobian is computed via central finite differences
        on FK, so a forward-difference check with a fresh perturbation
        validates internal consistency end-to-end.
        """
        robot = create_six_dof_mdh()
        jac = JacobianComputer(robot)
        home_rad = np.radians([0, 90, 0, 0, 180, 0]).tolist()

        j_num = jac.compute_numerical(home_rad)  # (3, 6)

        delta = 1e-5
        rng = np.random.default_rng(seed=31)
        dq = rng.uniform(-delta, delta, size=6)

        # Predicted displacement via numerical Jacobian
        dp_pred = j_num @ dq

        # Actual displacement via FK
        q_arr = np.array(home_rad, dtype=np.float64)
        pos_base = robot.forward_kinematics(q_arr.tolist()).position
        pos_pert = robot.forward_kinematics((q_arr + dq).tolist()).position
        dp_actual = pos_pert - pos_base

        np.testing.assert_allclose(
            dp_pred, dp_actual, atol=1e-7,
            err_msg="Numerical Jacobian velocity prediction mismatch",
        )

    def test_manipulability_positive_at_home(self) -> None:
        """Manipulability must be positive at the home configuration,
        indicating the arm is not in a singular posture."""
        robot = create_six_dof_mdh()
        jac = JacobianComputer(robot)
        home_rad = np.radians([0, 90, 0, 0, 180, 0]).tolist()
        mu = jac.manipulability(home_rad)
        assert mu > 0.0, f"Manipulability at home is {mu} (expected > 0)"

    def test_jacobian_finite_at_random_configs(self) -> None:
        """Jacobian entries must be finite for 10 random configs."""
        robot = create_six_dof_mdh()
        jac = JacobianComputer(robot)
        rng = np.random.default_rng(seed=77)
        for _ in range(10):
            q = rng.uniform(-1.0, 1.0, size=6).tolist()
            jacobian = jac.compute(q)
            assert np.all(np.isfinite(jacobian)), (
                f"Non-finite Jacobian at q={q}"
            )


# ======================================================================
# Wrong-input validation
# ======================================================================


class TestSixDOFWrongInputs:
    """Input validation for the 6-DOF robot."""

    def test_wrong_number_of_angles(self) -> None:
        """5 angles for a 6-DOF robot must raise an error."""
        robot = create_six_dof_mdh()
        with pytest.raises(Exception):
            robot.forward_kinematics([0.0] * 5)

    def test_too_many_angles(self) -> None:
        """7 angles for a 6-DOF robot must raise an error."""
        robot = create_six_dof_mdh()
        with pytest.raises(Exception):
            robot.forward_kinematics([0.0] * 7)
