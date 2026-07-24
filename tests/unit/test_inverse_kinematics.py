"""Unit tests for inverse kinematics solvers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roboarm.core.types import EndEffectorPose, IKSolution
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.kinematics.solvers.registry import IKSolverRegistry

# Importing the solvers package triggers auto-registration of all solvers.
import roboarm.kinematics.solvers  # noqa: F401


def _make_target(x: float, y: float, z: float = 0.0) -> EndEffectorPose:
    """Build a target EndEffectorPose from Cartesian coordinates."""
    position = np.array([x, y, z], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    return EndEffectorPose(
        position=position,
        rotation=np.eye(3, dtype=np.float64),
        transform=transform,
    )


class TestFKIKRoundtrip:
    """FK -> IK -> FK roundtrip tests."""

    def test_dls_roundtrip(self, two_link_robot):
        """Damped-least-squares should recover joint angles via FK->IK->FK."""
        q_original = [0.5, -0.3]
        pose = two_link_robot.forward_kinematics(q_original)
        target = _make_target(pose.x, pose.y)

        solver = IKSolverRegistry.create("damped_least_squares", two_link_robot)
        result = solver.solve(target)

        assert result.success is True
        assert result.primary is not None

        # Verify FK of the IK solution matches the target
        recovered = two_link_robot.forward_kinematics(result.primary.values)
        assert recovered.x == pytest.approx(pose.x, abs=1e-4)
        assert recovered.y == pytest.approx(pose.y, abs=1e-4)

    def test_jacobian_pseudoinverse_roundtrip(self, two_link_robot):
        """Jacobian pseudo-inverse should solve a reachable target."""
        q_original = [0.3, 0.6]
        pose = two_link_robot.forward_kinematics(q_original)
        target = _make_target(pose.x, pose.y)

        solver = IKSolverRegistry.create(
            "jacobian_pseudoinverse", two_link_robot,
        )
        result = solver.solve(target)

        assert result.success is True
        assert result.residual_error < 1e-4


class TestAnalyticalSolver:
    """Tests specific to the analytical 2-link IK solver."""

    def test_analytical_finds_solution(self, two_link_robot):
        """Analytical solver should find a solution for a reachable target."""
        target = _make_target(1.0, 1.0)
        solver = IKSolverRegistry.create("analytical_2link", two_link_robot)
        result = solver.solve(target)

        assert result.success is True
        assert result.primary is not None
        assert result.residual_error < 1e-6

    def test_analytical_returns_alternatives(self, two_link_robot):
        """Analytical solver should find elbow-up and elbow-down solutions."""
        target = _make_target(1.0, 1.0)
        solver = IKSolverRegistry.create("analytical_2link", two_link_robot)
        result = solver.solve(target)

        assert result.success is True
        assert len(result.alternatives) >= 1  # at least one alternative


class TestAllSolversReachableTarget:
    """Verify all registered solvers can solve a reachable target."""

    def test_all_solvers_solve_reachable(self, two_link_robot):
        """Every registered solver should converge for a reachable point."""
        target = _make_target(1.0, 0.5)
        available = IKSolverRegistry.available()
        assert len(available) > 0, "No IK solvers registered"

        for solver_name in available:
            solver = IKSolverRegistry.create(solver_name, two_link_robot)
            result = solver.solve(target)
            assert result.success is True, (
                f"Solver {solver_name!r} failed for reachable target "
                f"(residual={result.residual_error:.2e})"
            )


class TestUnreachableTarget:
    """Tests for targets outside the workspace."""

    def test_unreachable_analytical(self, two_link_robot):
        """Analytical solver should report failure for unreachable target."""
        target = _make_target(5.0, 5.0)  # well beyond reach of L1+L2=2.0
        solver = IKSolverRegistry.create("analytical_2link", two_link_robot)
        result = solver.solve(target)
        assert result.success is False

    def test_unreachable_dls_large_residual(self, two_link_robot):
        """DLS solver should have large residual for unreachable target."""
        target = _make_target(5.0, 5.0)
        solver = IKSolverRegistry.create(
            "damped_least_squares", two_link_robot,
        )
        result = solver.solve(target, q0=[0.0, 0.0])
        # Either success is False or residual is large
        if result.success:
            assert result.residual_error > 0.1
        else:
            assert result.residual_error > 0.1


class TestSolverRegistry:
    """Tests for the IKSolverRegistry itself."""

    def test_available_returns_non_empty(self):
        """Registry should have at least one solver after imports."""
        available = IKSolverRegistry.available()
        assert len(available) > 0

    def test_known_solvers_registered(self):
        """Essential solvers should be registered."""
        available = IKSolverRegistry.available()
        assert "damped_least_squares" in available
        assert "analytical_2link" in available
        assert "jacobian_pseudoinverse" in available

    def test_available_returns_sorted(self):
        """The available() list should be sorted alphabetically."""
        available = IKSolverRegistry.available()
        assert available == sorted(available)
