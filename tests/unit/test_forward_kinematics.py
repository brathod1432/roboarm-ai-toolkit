"""Unit tests for forward kinematics computations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roboarm.core.exceptions import ValidationError
from roboarm.core.robot import RobotArm
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.robots.three_link_planar import create_three_link_planar


class TestTwoLinkFK:
    """Forward kinematics tests for a 2-link planar robot."""

    def test_zero_angles(self, two_link_robot):
        """At q=[0,0] the end-effector should be at (2.0, 0.0, 0.0)."""
        pose = two_link_robot.forward_kinematics([0.0, 0.0])
        np.testing.assert_array_almost_equal(
            pose.position, [2.0, 0.0, 0.0], decimal=10,
        )

    def test_shoulder_90(self, two_link_robot):
        """At q=[pi/2, 0] the end-effector should be at (0.0, 2.0, 0.0)."""
        pose = two_link_robot.forward_kinematics([math.pi / 2, 0.0])
        np.testing.assert_array_almost_equal(
            pose.position, [0.0, 2.0, 0.0], decimal=10,
        )

    def test_pi4_minus_pi6(self, two_link_robot):
        """At q=[pi/4, -pi/6] verify numerically computed position."""
        q1 = math.pi / 4
        q2 = -math.pi / 6
        # Analytically: x = cos(q1) + cos(q1+q2), y = sin(q1) + sin(q1+q2)
        x_expected = math.cos(q1) + math.cos(q1 + q2)
        y_expected = math.sin(q1) + math.sin(q1 + q2)
        pose = two_link_robot.forward_kinematics([q1, q2])
        assert pose.x == pytest.approx(x_expected, abs=1e-10)
        assert pose.y == pytest.approx(y_expected, abs=1e-10)
        assert pose.z == pytest.approx(0.0, abs=1e-10)

    def test_fully_extended(self, two_link_robot):
        """Fully extended arm should reach maximum distance."""
        pose = two_link_robot.forward_kinematics([0.0, 0.0])
        distance = math.sqrt(pose.x ** 2 + pose.y ** 2)
        assert distance == pytest.approx(2.0, abs=1e-10)

    def test_folded_back(self, two_link_robot):
        """At q=[0, pi] the arm folds back to the origin."""
        pose = two_link_robot.forward_kinematics([0.0, math.pi])
        distance = math.sqrt(pose.x ** 2 + pose.y ** 2)
        assert distance == pytest.approx(0.0, abs=1e-10)

    def test_unequal_links(self, two_link_unequal):
        """Unequal-length arms should extend to L1+L2."""
        pose = two_link_unequal.forward_kinematics([0.0, 0.0])
        assert pose.x == pytest.approx(1.8, abs=1e-10)
        assert pose.y == pytest.approx(0.0, abs=1e-10)


class TestThreeLinkFK:
    """Forward kinematics tests for a 3-link planar robot."""

    def test_zero_angles(self, three_link_robot):
        """At q=[0,0,0] the end-effector should be at (2.5, 0.0, 0.0)."""
        pose = three_link_robot.forward_kinematics([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(
            pose.position, [2.5, 0.0, 0.0], decimal=10,
        )

    def test_all_90(self, three_link_robot):
        """At q=[pi/2, 0, 0] the arm should point along Y."""
        pose = three_link_robot.forward_kinematics([math.pi / 2, 0.0, 0.0])
        assert pose.x == pytest.approx(0.0, abs=1e-10)
        assert pose.y == pytest.approx(2.5, abs=1e-10)


class TestRobotProperties:
    """Tests for RobotArm metadata properties."""

    def test_n_joints(self, two_link_robot):
        """A 2-link robot should have 2 joints."""
        assert two_link_robot.n_joints == 2

    def test_n_dof(self, two_link_robot):
        """A 2-link robot should have 2 DOF."""
        assert two_link_robot.n_dof == 2

    def test_joint_names(self, two_link_robot):
        """Joint names should be returned as a list."""
        names = two_link_robot.joint_names
        assert isinstance(names, list)
        assert len(names) == 2

    def test_three_link_n_dof(self, three_link_robot):
        """A 3-link robot should have 3 DOF."""
        assert three_link_robot.n_dof == 3


class TestFKValidation:
    """Tests for FK input validation."""

    def test_wrong_number_of_angles(self, two_link_robot):
        """Passing 3 angles to a 2-DOF robot should raise ValidationError."""
        with pytest.raises(ValidationError):
            two_link_robot.forward_kinematics([0.0, 0.0, 0.0])

    def test_too_few_angles(self, two_link_robot):
        """Passing 1 angle to a 2-DOF robot should raise ValidationError."""
        with pytest.raises(ValidationError):
            two_link_robot.forward_kinematics([0.0])

    def test_empty_angles(self, two_link_robot):
        """Passing empty list to a 2-DOF robot should raise ValidationError."""
        with pytest.raises(ValidationError):
            two_link_robot.forward_kinematics([])


class TestRobotArmRepr:
    """Tests for RobotArm string representation."""

    def test_repr_contains_name(self, two_link_robot):
        """The repr should include the robot name."""
        r = repr(two_link_robot)
        assert "RobotArm" in r
        assert "n_joints=2" in r
        assert "n_dof=2" in r
