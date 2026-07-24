"""Security, injection-prevention, and unsupported-operation tests.

Validates that the agent layer, tool registry, and IK solvers handle
malicious, malformed, and unsupported inputs gracefully without
crashing or producing dangerous outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from roboarm.agents.coordinator import RoboticsCoordinator
from roboarm.agents.robotics_tools import build_robotics_tools
from roboarm.agents.tools import ToolRegistry
from roboarm.core.exceptions import ConfigurationError, RobotArmError
from roboarm.core.types import EndEffectorPose
from roboarm.kinematics.solvers.registry import IKSolverRegistry
from roboarm.robots.two_link_planar import create_two_link_planar

import roboarm.kinematics.solvers  # noqa: F401  -- triggers auto-registration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(x: float, y: float, z: float = 0.0) -> EndEffectorPose:
    """Build an ``EndEffectorPose`` from Cartesian coordinates."""
    position = np.array([x, y, z], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    return EndEffectorPose(
        position=position,
        rotation=np.eye(3, dtype=np.float64),
        transform=transform,
    )


# ===================================================================
# 1. Agent injection / fuzzing tests
# ===================================================================

class TestAgentInjection:
    """Verify agents handle malicious or unusual inputs safely."""

    def test_sql_injection_string(self) -> None:
        """SQL-injection-like input must not crash the coordinator."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("'; DROP TABLE robots; --")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_script_injection(self) -> None:
        """HTML / JS injection must not crash the coordinator."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("<script>alert('xss')</script>")
        assert isinstance(response, str)

    def test_path_traversal(self) -> None:
        """Path-traversal attempt must not crash the coordinator."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("../../../etc/passwd")
        assert isinstance(response, str)

    def test_very_long_input(self) -> None:
        """A 10 KB input string must not crash or hang."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("A" * 10_000)
        assert isinstance(response, str)

    def test_empty_input(self) -> None:
        """An empty string must not crash the coordinator."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("")
        assert isinstance(response, str)

    def test_null_bytes(self) -> None:
        """Null bytes embedded in input must not crash the coordinator."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("compute FK\x00for angles [0.5, -0.3]")
        assert isinstance(response, str)

    def test_unicode_input(self) -> None:
        """Unicode characters must not crash the coordinator."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process(
            "\u2603 Compute FK for angles [0.5, -0.3] \u2764"
        )
        assert isinstance(response, str)

    def test_special_chars_in_angles(self) -> None:
        """Non-numeric characters where angles are expected must not crash."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("FK for angles [abc, def]")
        assert isinstance(response, str)

    def test_negative_number_injection(self) -> None:
        """Extreme negative numbers must not crash the coordinator."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("Solve IK for x=-1e308, y=-1e308")
        assert isinstance(response, str)

    def test_format_string_attack(self) -> None:
        """Python format-string patterns must not crash the coordinator."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("{0.__class__.__mro__}")
        assert isinstance(response, str)

    def test_repeated_special_characters(self) -> None:
        """Repeated special characters must not cause regex catastrophe."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("((((((((((((((((((((")
        assert isinstance(response, str)


# ===================================================================
# 2. Unsupported-operation tests
# ===================================================================

class TestUnsupportedOperations:
    """Verify unsupported operations are properly rejected."""

    def test_unsupported_command(self) -> None:
        """An unrelated command must produce a non-trivial help or error."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("make coffee")
        assert isinstance(response, str)
        # Should contain help or error, not a stub
        assert len(response) > 10

    def test_dynamics_not_supported(self) -> None:
        """Dynamics / torque requests must not crash."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("compute torques for joint 1")
        assert isinstance(response, str)

    def test_collision_not_supported(self) -> None:
        """Collision-detection requests must not crash."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("check for collisions")
        assert isinstance(response, str)

    def test_trajectory_planning_not_supported(self) -> None:
        """High-level trajectory-planning commands must not crash."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        response = coordinator.process("plan a trajectory from A to B")
        assert isinstance(response, str)


# ===================================================================
# 3. Tool-registry security
# ===================================================================

class TestToolRegistrySecurity:
    """Tool-registry access-control and error handling."""

    def test_execute_nonexistent_tool(self) -> None:
        """Executing a non-existent tool must raise ``KeyError``."""
        robot = create_two_link_planar()
        tools = build_robotics_tools(robot)
        with pytest.raises(KeyError):
            tools.execute("nonexistent_tool")

    def test_tool_with_wrong_args(self) -> None:
        """Calling a tool without required arguments must raise cleanly."""
        robot = create_two_link_planar()
        tools = build_robotics_tools(robot)
        # ``compute_fk`` expects ``angles=...``; omitting it should raise.
        with pytest.raises((TypeError, KeyError, RobotArmError)):
            tools.execute("compute_fk")

    def test_registry_available_returns_list(self) -> None:
        """``IKSolverRegistry.available()`` must return a list of strings."""
        result = IKSolverRegistry.available()
        assert isinstance(result, list)
        for name in result:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_registry_create_unknown_solver(self) -> None:
        """Creating an unknown solver must raise ``ConfigurationError``."""
        robot = create_two_link_planar()
        with pytest.raises(ConfigurationError):
            IKSolverRegistry.create("totally_fake_solver", robot)

    def test_tool_list_is_sorted(self) -> None:
        """``list_tools()`` must return a sorted list of tool names."""
        robot = create_two_link_planar()
        tools = build_robotics_tools(robot)
        names = tools.list_tools()
        assert names == sorted(names)
        assert len(names) >= 4  # at least describe, fk, ik, jacobian


# ===================================================================
# 4. Solver output safety-bounds
# ===================================================================

class TestSolverSecurityBounds:
    """Verify solvers never produce dangerous or invalid outputs."""

    def test_solver_output_finite(self) -> None:
        """All solver outputs must have finite joint angles."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        target = _make_target(1.0, 0.5)
        for name in IKSolverRegistry.available():
            solver = IKSolverRegistry.create(name, robot)
            result = solver.solve(target)
            if result.success and result.primary is not None:
                assert np.all(np.isfinite(result.primary.values)), (
                    f"Solver {name!r} produced non-finite angles"
                )

    def test_solver_residual_non_negative(self) -> None:
        """Residual error must be non-negative for every solver."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        target = _make_target(1.0, 0.5)
        for name in IKSolverRegistry.available():
            solver = IKSolverRegistry.create(name, robot)
            result = solver.solve(target)
            assert result.residual_error >= 0, (
                f"Solver {name!r} produced negative residual"
            )

    def test_solver_computation_time_non_negative(self) -> None:
        """Computation time must be non-negative for every solver."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        target = _make_target(1.0, 0.5)
        for name in IKSolverRegistry.available():
            solver = IKSolverRegistry.create(name, robot)
            result = solver.solve(target)
            assert result.computation_time_ms >= 0, (
                f"Solver {name!r} reported negative computation time"
            )

    def test_solver_unreachable_target_does_not_crash(self) -> None:
        """An unreachable target must not crash any solver."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        unreachable = _make_target(100.0, 100.0)
        for name in IKSolverRegistry.available():
            solver = IKSolverRegistry.create(name, robot)
            result = solver.solve(unreachable)
            # May succeed or fail, but must never crash
            assert isinstance(result.success, bool)
            assert result.residual_error >= 0
