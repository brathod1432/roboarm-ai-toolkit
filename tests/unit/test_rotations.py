"""Unit tests for SO(3) rotation utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roboarm.core.rotations import (
    axis_angle_to_rotation,
    euler_to_rotation,
    is_valid_rotation,
    quaternion_to_rotation,
    rotation_to_axis_angle,
    rotation_to_euler,
    rotation_to_quaternion,
    rotx,
    roty,
    rotz,
)


class TestBasicRotations:
    """Tests for rotx, roty, rotz at known angles."""

    def test_rotx_90(self):
        """Rotation about X by 90 degrees should map Y to Z."""
        R = rotx(math.pi / 2)
        y_axis = np.array([0.0, 1.0, 0.0])
        result = R @ y_axis
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 1.0])
        assert is_valid_rotation(R)

    def test_roty_90(self):
        """Rotation about Y by 90 degrees should map Z to X."""
        R = roty(math.pi / 2)
        z_axis = np.array([0.0, 0.0, 1.0])
        result = R @ z_axis
        np.testing.assert_array_almost_equal(result, [1.0, 0.0, 0.0])
        assert is_valid_rotation(R)

    def test_rotz_90(self):
        """Rotation about Z by 90 degrees should map X to Y."""
        R = rotz(math.pi / 2)
        x_axis = np.array([1.0, 0.0, 0.0])
        result = R @ x_axis
        np.testing.assert_array_almost_equal(result, [0.0, 1.0, 0.0])
        assert is_valid_rotation(R)

    def test_rotx_zero_is_identity(self):
        """Rotation by zero should be the identity."""
        R = rotx(0.0)
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_rotz_180(self):
        """Rotation about Z by 180 degrees should negate X and Y."""
        R = rotz(math.pi)
        x_axis = np.array([1.0, 0.0, 0.0])
        result = R @ x_axis
        np.testing.assert_array_almost_equal(result, [-1.0, 0.0, 0.0])


class TestEulerRoundtrip:
    """Tests for euler_to_rotation and rotation_to_euler."""

    @pytest.mark.parametrize("roll,pitch,yaw", [
        (0.0, 0.0, 0.0),
        (0.3, -0.2, 0.5),
        (math.pi / 4, math.pi / 6, -math.pi / 3),
        (1.0, 0.5, -1.0),
    ])
    def test_roundtrip(self, roll, pitch, yaw):
        """Converting Euler -> rotation -> Euler should recover angles."""
        R = euler_to_rotation(roll, pitch, yaw)
        assert is_valid_rotation(R)
        r2, p2, y2 = rotation_to_euler(R)
        R2 = euler_to_rotation(r2, p2, y2)
        np.testing.assert_array_almost_equal(R, R2, decimal=10)


class TestAxisAngleRoundtrip:
    """Tests for axis_angle_to_rotation and rotation_to_axis_angle."""

    def test_roundtrip_arbitrary(self):
        """axis_angle -> rotation -> axis_angle should roundtrip."""
        axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
        angle = math.pi / 3
        R = axis_angle_to_rotation(axis, angle)
        assert is_valid_rotation(R)
        axis_out, angle_out = rotation_to_axis_angle(R)
        R2 = axis_angle_to_rotation(axis_out, angle_out)
        np.testing.assert_array_almost_equal(R, R2, decimal=10)

    def test_zero_angle(self):
        """Zero rotation angle should give identity."""
        R = axis_angle_to_rotation(np.array([0.0, 0.0, 1.0]), 0.0)
        np.testing.assert_array_almost_equal(R, np.eye(3))


class TestQuaternionRoundtrip:
    """Tests for quaternion_to_rotation and rotation_to_quaternion."""

    def test_identity_quaternion(self):
        """The identity quaternion [1,0,0,0] should give identity rotation."""
        R = quaternion_to_rotation(np.array([1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_roundtrip(self):
        """quaternion -> rotation -> quaternion should roundtrip."""
        q = np.array([0.7071, 0.0, 0.7071, 0.0])
        q = q / np.linalg.norm(q)
        R = quaternion_to_rotation(q)
        assert is_valid_rotation(R)
        q_out = rotation_to_quaternion(R)
        R2 = quaternion_to_rotation(q_out)
        np.testing.assert_array_almost_equal(R, R2, decimal=10)

    def test_90deg_about_z(self):
        """Quaternion for 90-deg rotation about Z should match rotz."""
        angle = math.pi / 2
        q = np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])
        R_quat = quaternion_to_rotation(q)
        R_direct = rotz(angle)
        np.testing.assert_array_almost_equal(R_quat, R_direct, decimal=10)


class TestIsValidRotation:
    """Tests for SO(3) validation."""

    def test_identity_is_valid(self):
        """Identity matrix is a valid rotation."""
        assert is_valid_rotation(np.eye(3)) == True  # noqa: E712

    def test_wrong_shape(self):
        """A 4x4 matrix should fail SO(3) validation."""
        assert is_valid_rotation(np.eye(4)) == False  # noqa: E712

    def test_non_orthogonal(self):
        """Scaled identity should fail SO(3) validation."""
        assert is_valid_rotation(2.0 * np.eye(3)) == False  # noqa: E712

    def test_reflection_is_invalid(self):
        """A reflection matrix (det=-1) should not be valid SO(3)."""
        R = np.diag([1.0, 1.0, -1.0])
        assert is_valid_rotation(R) == False  # noqa: E712
