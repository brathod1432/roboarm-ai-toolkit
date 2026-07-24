"""Unit tests for core data types."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roboarm.core.types import (
    DHParams,
    EndEffectorPose,
    IKSolution,
    JointConfig,
    JointLimits,
    JointSolution,
    list_to_array,
)


class TestDHParams:
    """Tests for the DHParams frozen dataclass."""

    def test_creation_with_defaults(self):
        """DHParams should default to standard convention."""
        params = DHParams(alpha=0.0, a=1.0, d=0.0, theta=0.0)
        assert params.alpha == 0.0
        assert params.a == 1.0
        assert params.d == 0.0
        assert params.theta == 0.0
        assert params.convention == "standard"

    def test_modified_convention(self):
        """DHParams should accept a modified convention string."""
        params = DHParams(
            alpha=math.pi / 2, a=0.5, d=0.1, theta=0.0, convention="modified",
        )
        assert params.convention == "modified"
        assert params.alpha == pytest.approx(math.pi / 2)

    def test_frozen_immutability(self):
        """DHParams instances should be immutable (frozen dataclass)."""
        params = DHParams(alpha=0.0, a=1.0, d=0.0, theta=0.0)
        with pytest.raises(AttributeError):
            params.a = 2.0  # type: ignore[misc]


class TestJointLimits:
    """Tests for the JointLimits frozen dataclass."""

    def test_basic_limits(self):
        """JointLimits should store lower and upper bounds."""
        limits = JointLimits(lower=-math.pi, upper=math.pi)
        assert limits.lower == pytest.approx(-math.pi)
        assert limits.upper == pytest.approx(math.pi)

    def test_optional_velocity_and_acceleration(self):
        """Velocity and acceleration limits should default to None."""
        limits = JointLimits(lower=-1.0, upper=1.0)
        assert limits.velocity_max is None
        assert limits.acceleration_max is None

    def test_full_limits(self):
        """JointLimits should accept all four fields."""
        limits = JointLimits(
            lower=-2.0, upper=2.0, velocity_max=5.0, acceleration_max=10.0,
        )
        assert limits.velocity_max == 5.0
        assert limits.acceleration_max == 10.0


class TestJointConfig:
    """Tests for the JointConfig dataclass."""

    def test_default_joint_config(self):
        """JointConfig should default to variable with no limits."""
        dh = DHParams(alpha=0.0, a=1.0, d=0.0, theta=0.0)
        jc = JointConfig(dh_params=dh)
        assert jc.is_variable is True
        assert jc.limits is None
        assert jc.name == ""

    def test_named_fixed_joint(self):
        """A fixed joint config should store is_variable=False."""
        dh = DHParams(alpha=0.0, a=0.0, d=0.5, theta=0.0)
        jc = JointConfig(dh_params=dh, name="TCP", is_variable=False)
        assert jc.is_variable is False
        assert jc.name == "TCP"


class TestEndEffectorPose:
    """Tests for EndEffectorPose including convenience properties."""

    def test_xyz_properties(self):
        """The x, y, z properties should return individual coordinates."""
        pos = np.array([1.5, 2.5, 3.5])
        rot = np.eye(3)
        tf = np.eye(4)
        tf[:3, 3] = pos
        pose = EndEffectorPose(position=pos, rotation=rot, transform=tf)
        assert pose.x == pytest.approx(1.5)
        assert pose.y == pytest.approx(2.5)
        assert pose.z == pytest.approx(3.5)

    def test_position_array(self):
        """Position should be a 3-element array."""
        pos = np.array([0.0, 0.0, 0.0])
        pose = EndEffectorPose(
            position=pos, rotation=np.eye(3), transform=np.eye(4),
        )
        assert pose.position.shape == (3,)


class TestIKSolution:
    """Tests for the IKSolution dataclass."""

    def test_default_failure(self):
        """An IKSolution with success=False should have sensible defaults."""
        sol = IKSolution(success=False)
        assert sol.success is False
        assert sol.primary is None
        assert sol.alternatives == []
        assert sol.iterations == 0
        assert sol.residual_error == float("inf")
        assert sol.solver_name == ""
        assert sol.messages == []

    def test_successful_solution(self):
        """A successful IKSolution should carry a primary JointSolution."""
        js = JointSolution(values=np.array([0.5, -0.3]))
        sol = IKSolution(
            success=True,
            primary=js,
            iterations=42,
            residual_error=1e-8,
            solver_name="test_solver",
        )
        assert sol.success is True
        assert sol.primary is not None
        np.testing.assert_array_almost_equal(sol.primary.values, [0.5, -0.3])
        assert sol.iterations == 42


class TestListToArray:
    """Tests for the list_to_array utility."""

    def test_converts_list(self):
        """list_to_array should produce a 1-D float64 array."""
        arr = list_to_array([1.0, 2.0, 3.0])
        assert arr.dtype == np.float64
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_converts_tuple(self):
        """list_to_array should accept tuples."""
        arr = list_to_array((4.0, 5.0))
        assert arr.shape == (2,)
