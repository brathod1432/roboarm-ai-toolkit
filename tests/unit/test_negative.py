"""Negative and error handling tests for the roboarm toolkit.

Every bad input must be handled gracefully. These tests verify that invalid
joint angles, robot configurations, IK targets, solver names, transform
inputs, and rotation inputs all produce clear errors instead of crashes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import roboarm.kinematics.solvers  # noqa: F401 — triggers auto-registration
from roboarm.core.exceptions import (
    ConfigurationError,
    RobotArmError,
    ValidationError,
)
from roboarm.core.robot import RobotArm
from roboarm.core.rotations import (
    is_valid_rotation,
    quaternion_to_rotation,
)
from roboarm.core.transform import (
    inverse_transform,
    is_valid_transform,
)
from roboarm.core.types import EndEffectorPose, IKSolution
from roboarm.kinematics.solvers.registry import IKSolverRegistry
from roboarm.robots.two_link_planar import create_two_link_planar


def _make_target(x: float, y: float, z: float = 0.0) -> EndEffectorPose:
    """Build a minimal IK target from Cartesian coordinates."""
    position = np.array([x, y, z], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    return EndEffectorPose(
        position=position,
        rotation=np.eye(3, dtype=np.float64),
        transform=transform,
    )


# ------------------------------------------------------------------ #
#  Invalid joint angles
# ------------------------------------------------------------------ #


class TestInvalidJointAngles:
    """Invalid joint angle inputs."""

    def test_nan_angles(self, two_link_robot):
        """NaN joint angles should raise or produce NaN position (not crash)."""
        try:
            pose = two_link_robot.forward_kinematics([float("nan"), 0.0])
            # If it returns, position components should be NaN
            assert np.any(np.isnan(pose.position)), (
                "NaN input should propagate to NaN output"
            )
        except (ValueError, ValidationError):
            pass  # raising is also acceptable

    def test_inf_angles(self, two_link_robot):
        """Inf angles should not crash."""
        try:
            pose = two_link_robot.forward_kinematics([float("inf"), 0.0])
            # If it returns, check it did not crash; NaN output is acceptable
            assert pose.position is not None
        except (ValueError, ValidationError):
            pass  # raising is also acceptable

    def test_none_input(self, two_link_robot):
        """None input should raise TypeError or ValidationError."""
        with pytest.raises((TypeError, ValidationError)):
            two_link_robot.forward_kinematics(None)

    def test_string_input(self, two_link_robot):
        """String input should raise error."""
        with pytest.raises((TypeError, ValidationError, ValueError)):
            two_link_robot.forward_kinematics("hello")

    def test_dict_input(self, two_link_robot):
        """Dict input should raise error."""
        with pytest.raises((TypeError, ValidationError, ValueError)):
            two_link_robot.forward_kinematics({"q1": 0.5})

    def test_empty_list(self, two_link_robot):
        """Empty list should raise ValidationError."""
        with pytest.raises(ValidationError):
            two_link_robot.forward_kinematics([])

    def test_too_many_angles(self, two_link_robot):
        """Too many angles should raise ValidationError."""
        with pytest.raises(ValidationError):
            two_link_robot.forward_kinematics([0.1, 0.2, 0.3])

    def test_single_angle(self, two_link_robot):
        """One angle for 2-DOF should raise ValidationError."""
        with pytest.raises(ValidationError):
            two_link_robot.forward_kinematics([0.5])

    def test_very_large_angles(self, two_link_robot):
        """Very large angles (1000 rad) should not crash."""
        pose = two_link_robot.forward_kinematics([1000.0, -1000.0])
        assert np.all(np.isfinite(pose.position))

    def test_very_small_angles(self, two_link_robot):
        """Very small angles (1e-15) should work fine."""
        pose = two_link_robot.forward_kinematics([1e-15, 1e-15])
        assert np.all(np.isfinite(pose.position))

    def test_negative_pi_boundary(self, two_link_robot):
        """Angles at exactly -pi should produce finite results."""
        pose = two_link_robot.forward_kinematics([-math.pi, -math.pi])
        assert np.all(np.isfinite(pose.position))

    def test_boolean_list_input(self, two_link_robot):
        """Boolean list should be coerced to floats (0.0/1.0) without crash."""
        # numpy treats bools as 0/1 integers, should not crash
        pose = two_link_robot.forward_kinematics([True, False])
        assert np.all(np.isfinite(pose.position))


# ------------------------------------------------------------------ #
#  Invalid robot configuration
# ------------------------------------------------------------------ #


class TestInvalidRobotConfig:
    """Invalid robot construction."""

    def test_empty_joints_list(self):
        """Empty joints list should raise ValidationError."""
        with pytest.raises(ValidationError):
            RobotArm([], name="Empty")

    def test_zero_length_links(self):
        """Robot with zero-length links should work (degenerate but valid)."""
        robot = create_two_link_planar(link1=0.0, link2=0.0)
        pose = robot.forward_kinematics([0.5, 0.3])
        assert abs(pose.x) < 1e-10 and abs(pose.y) < 1e-10

    def test_negative_link_lengths(self):
        """Negative link lengths should work mathematically (direction reversal)."""
        robot = create_two_link_planar(link1=-1.0, link2=1.0)
        pose = robot.forward_kinematics([0.0, 0.0])
        assert np.all(np.isfinite(pose.position))

    def test_very_long_links(self):
        """Very long links (1e6) should not overflow."""
        robot = create_two_link_planar(link1=1e6, link2=1e6)
        pose = robot.forward_kinematics([0.0, 0.0])
        assert abs(pose.x - 2e6) < 1.0

    def test_very_short_links(self):
        """Very short links (1e-12) should produce finite results."""
        robot = create_two_link_planar(link1=1e-12, link2=1e-12)
        pose = robot.forward_kinematics([0.5, -0.3])
        assert np.all(np.isfinite(pose.position))

    def test_mismatched_link_ratio(self):
        """Extremely mismatched link ratio should still produce valid FK."""
        robot = create_two_link_planar(link1=1000.0, link2=0.001)
        pose = robot.forward_kinematics([0.0, 0.0])
        assert abs(pose.x - 1000.001) < 0.01


# ------------------------------------------------------------------ #
#  Invalid IK targets
# ------------------------------------------------------------------ #


class TestInvalidIKTargets:
    """Invalid IK target inputs."""

    def test_unreachable_far(self):
        """Target far outside workspace should fail gracefully."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        target = _make_target(100.0, 100.0)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        result = solver.solve(target)
        assert isinstance(result, IKSolution)

    def test_unreachable_origin(self):
        """Target at exact origin (0,0) for non-folding arm."""
        robot = create_two_link_planar(link1=1.0, link2=0.5)
        target = _make_target(0.0, 0.0)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        result = solver.solve(target)
        assert isinstance(result, IKSolution)

    def test_nan_target(self):
        """NaN target should not crash solver."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        target = _make_target(float("nan"), 0.5)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        try:
            result = solver.solve(target)
            # If it returns, it should indicate failure or have NaN residual
            assert not result.success or not np.isfinite(result.residual_error)
        except (ValueError, RobotArmError):
            pass  # raising is acceptable

    def test_negative_coordinates(self):
        """Negative coordinates should be valid if reachable."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        target = _make_target(-1.0, 0.5)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        result = solver.solve(target)
        assert isinstance(result, IKSolution)

    def test_target_just_barely_reachable(self):
        """Target at exact max reach (L1+L2) edge case."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        target = _make_target(2.0, 0.0)  # exactly at max reach
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        result = solver.solve(target)
        assert isinstance(result, IKSolution)

    def test_target_at_large_negative(self):
        """Very large negative coordinates should not crash."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        target = _make_target(-1e6, -1e6)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        result = solver.solve(target)
        assert isinstance(result, IKSolution)


# ------------------------------------------------------------------ #
#  Invalid solver config
# ------------------------------------------------------------------ #


class TestInvalidSolverConfig:
    """Invalid solver registry usage."""

    def test_unknown_solver_name(self):
        """Requesting non-existent solver should raise."""
        robot = create_two_link_planar()
        with pytest.raises((KeyError, ConfigurationError, RobotArmError)):
            IKSolverRegistry.create("nonexistent_solver", robot)

    def test_empty_solver_name(self):
        """Empty string solver name should raise."""
        robot = create_two_link_planar()
        with pytest.raises((KeyError, ConfigurationError, RobotArmError)):
            IKSolverRegistry.create("", robot)

    def test_none_solver_name(self):
        """None as solver name should raise."""
        robot = create_two_link_planar()
        with pytest.raises((KeyError, ConfigurationError, RobotArmError, TypeError)):
            IKSolverRegistry.create(None, robot)


# ------------------------------------------------------------------ #
#  Invalid transform inputs
# ------------------------------------------------------------------ #


class TestInvalidTransformInputs:
    """Invalid inputs to transform functions."""

    def test_is_valid_transform_wrong_shape(self):
        """3x3 matrix should fail is_valid_transform."""
        assert not is_valid_transform(np.eye(3))

    def test_is_valid_transform_5x5(self):
        """5x5 matrix should fail."""
        assert not is_valid_transform(np.eye(5))

    def test_is_valid_transform_non_orthogonal(self):
        """Non-orthogonal 4x4 should fail."""
        T = np.eye(4)
        T[0, 1] = 0.5  # break orthogonality
        assert not is_valid_transform(T)

    def test_inverse_of_non_rotation(self):
        """inverse_transform on non-rotation should still return a matrix."""
        T = np.eye(4)
        T[0, 0] = 2.0  # not a proper rotation
        result = inverse_transform(T)
        assert result.shape == (4, 4)

    def test_is_valid_transform_bad_bottom_row(self):
        """4x4 with bad bottom row should fail."""
        T = np.eye(4)
        T[3, 0] = 1.0  # corrupt bottom row
        assert not is_valid_transform(T)

    def test_is_valid_transform_negative_det(self):
        """4x4 with reflection (det=-1) should fail."""
        T = np.eye(4)
        T[0, 0] = -1.0  # reflection
        assert not is_valid_transform(T)

    def test_is_valid_transform_1d_array(self):
        """1-D array should fail is_valid_transform."""
        assert not is_valid_transform(np.array([1, 0, 0, 0]))


# ------------------------------------------------------------------ #
#  Invalid rotation inputs
# ------------------------------------------------------------------ #


class TestInvalidRotationInputs:
    """Invalid rotation inputs."""

    def test_is_valid_rotation_wrong_shape(self):
        """2x2 should fail."""
        assert not is_valid_rotation(np.eye(2))

    def test_is_valid_rotation_reflection(self):
        """det = -1 (reflection) should fail."""
        R = np.diag([1.0, 1.0, -1.0])
        assert not is_valid_rotation(R)

    def test_quaternion_zero_vector(self):
        """Zero quaternion should be handled (normalized or error)."""
        try:
            R = quaternion_to_rotation(np.array([0.0, 0.0, 0.0, 0.0]))
            # If it returns, it should be a 3x3 matrix
            assert R.shape == (3, 3)
        except (ValueError, ZeroDivisionError, FloatingPointError):
            pass  # raising is acceptable

    def test_is_valid_rotation_4x4(self):
        """4x4 identity should fail is_valid_rotation."""
        assert not is_valid_rotation(np.eye(4))

    def test_is_valid_rotation_1d(self):
        """1-D array should fail is_valid_rotation."""
        assert not is_valid_rotation(np.array([1.0, 0.0, 0.0]))

    def test_is_valid_rotation_scaled_matrix(self):
        """Scaled rotation (det != 1) should fail."""
        R = 2.0 * np.eye(3)  # det = 8, not SO(3)
        assert not is_valid_rotation(R)
