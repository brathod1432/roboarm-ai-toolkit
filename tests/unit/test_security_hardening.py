"""Comprehensive security hardening tests for roboarm-ai-toolkit.

Test classes and scenarios covered
------------------------------------
S1 – TestDHParamInjection     : NaN/Inf in DH parameters rejected at every entry point.
S2 – TestJointCountDoS        : Joint count limits prevent resource exhaustion.
S3 – TestCoordinateBounds     : Non-finite / astronomically large IK targets rejected.
S4 – TestLogInjection         : User input is sanitized before reaching log records.
S5 – TestCSVFormulaInjection  : Spreadsheet formula prefixes neutralised in CSV output.
S6 – TestAuditLogIntegrity    : Returned audit-log entries are immutable copies.
S8 – TestJointVelocitySaturation : joint_velocities() saturates and guards NaN output.
R1 – TestForwardKinematicsNaNGuard : forward_kinematics() raises on non-finite transforms.
"""

from __future__ import annotations

import logging
import os
import tempfile

import numpy as np
import pytest

import roboarm.kinematics.solvers  # noqa: F401 — triggers auto-registration of all solvers
from roboarm.agents._request_context import request_context
from roboarm.agents.coordinator import RoboticsCoordinator
from roboarm.agents.tools import ToolDefinition, ToolRegistry
from roboarm.core.builder import RobotBuilder
from roboarm.core.exceptions import ValidationError
from roboarm.core.robot import RobotArm
from roboarm.core.types import DHParams, JointConfig
from roboarm.kinematics.jacobian import JacobianComputer
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.trajectory.io import _safe_csv_cell, load_trajectory_csv, save_trajectory_csv
from roboarm.utils.log_event import sanitize_for_log

# ---------------------------------------------------------------------------
# Module-level helpers shared across test classes
# ---------------------------------------------------------------------------

_MAX_JOINTS: int = 32  # mirrors roboarm.core.robot._MAX_JOINTS


def _simple_joint(
    alpha: float = 0.0,
    a: float = 1.0,
    d: float = 0.0,
    theta: float = 0.0,
    convention: str = "standard",
) -> JointConfig:
    """Return a single revolute :class:`JointConfig` with the given DH params.

    Constructs the :class:`DHParams` dataclass directly, bypassing all
    ``from_dict`` validation.  Use this only when testing the FK NaN guard
    or other post-construction safety nets.
    """
    return JointConfig(
        dh_params=DHParams(alpha=alpha, a=a, d=d, theta=theta, convention=convention)
    )


def _robot_dict(
    alpha: float = 0.0,
    a: float = 1.0,
    d: float = 0.0,
    theta: float = 0.0,
    convention: str = "standard",
    n_joints: int = 1,
) -> dict:
    """Return a minimal ``RobotArm.from_dict()``-compatible dict.

    All joints share the same DH parameter values, which makes it easy to
    inject bad values (NaN/Inf/bad-convention) into ``from_dict``.
    """
    joint_entry = {
        "dh_params": {
            "alpha": alpha,
            "a": a,
            "d": d,
            "theta": theta,
            "convention": convention,
        }
    }
    return {
        "name": "TestRobot",
        "joints": [joint_entry] * n_joints,
    }


# ============================================================================
# S1 — DH Parameter Injection
# ============================================================================


class TestDHParamInjection:
    """S1: NaN and Inf must be rejected at every DH-parameter entry point.

    Attack vector: an attacker crafts a JSON robot file where one DH field
    contains NaN or Infinity.  If accepted, NaN would silently propagate
    through all downstream maths (FK, Jacobian, IK) producing garbage results
    without any visible error.
    """

    def test_nan_alpha_rejected(self) -> None:
        """from_dict with alpha=NaN must raise ValidationError citing 'finite'."""
        data = _robot_dict(alpha=float("nan"))
        with pytest.raises(ValidationError, match="finite"):
            RobotArm.from_dict(data)

    def test_inf_a_rejected(self) -> None:
        """from_dict with a=+Infinity must raise ValidationError citing 'finite'."""
        data = _robot_dict(a=float("inf"))
        with pytest.raises(ValidationError, match="finite"):
            RobotArm.from_dict(data)

    def test_nan_in_builder(self) -> None:
        """RobotBuilder.add_revolute(a=NaN) must raise ValidationError immediately."""
        with pytest.raises(ValidationError):
            RobotBuilder("BadRobot").add_revolute(a=float("nan"))

    def test_inf_in_builder(self) -> None:
        """RobotBuilder.add_revolute(d=+Inf) must raise ValidationError immediately."""
        with pytest.raises(ValidationError):
            RobotBuilder("BadRobot").add_revolute(d=float("inf"))

    def test_valid_params_accepted(self) -> None:
        """Finite DH parameters must produce a valid robot with finite FK output."""
        data = _robot_dict(alpha=0.0, a=1.0, d=0.0, theta=0.0)
        robot = RobotArm.from_dict(data)
        assert robot.n_dof == 1
        pose = robot.forward_kinematics([0.0])
        assert np.all(np.isfinite(pose.position)), "FK position must be finite"

    def test_convention_whitelist(self) -> None:
        """Convention string 'evil' must be rejected with a 'convention' error."""
        data = _robot_dict(convention="evil")
        with pytest.raises(ValidationError, match="convention"):
            RobotArm.from_dict(data)

    def test_nan_propagation_blocked(self) -> None:
        """A robot constructed directly with NaN alpha must raise in forward_kinematics.

        ``from_dict`` validation is bypassed by constructing ``DHParams`` and
        ``RobotArm`` directly.  The FK NaN guard must still catch the bad value.
        """
        jc = _simple_joint(alpha=float("nan"), a=1.0)
        robot = RobotArm([jc])  # __init__ checks only joint count, not DH values
        with pytest.raises(ValidationError, match="non-finite"):
            robot.forward_kinematics([0.0])


# ============================================================================
# S2 — Joint Count DoS
# ============================================================================


class TestJointCountDoS:
    """S2: Excessively large joint counts must be rejected at every entry point.

    Attack vector: a crafted JSON robot file with thousands of joints triggers
    O(N) computation in FK/Jacobian loops, potentially exhausting CPU/memory.
    """

    @staticmethod
    def _joint_list(n: int) -> list[JointConfig]:
        """Return n identical JointConfig objects."""
        return [_simple_joint() for _ in range(n)]

    def test_too_many_joints_in_json(self) -> None:
        """A robot dict with 33 joints must raise ValidationError."""
        data = _robot_dict(n_joints=_MAX_JOINTS + 1)
        with pytest.raises(ValidationError, match="[Mm]ax"):
            RobotArm.from_dict(data)

    def test_max_joints_accepted(self) -> None:
        """Exactly 32 joints must be accepted by from_dict without error."""
        data = _robot_dict(n_joints=_MAX_JOINTS)
        robot = RobotArm.from_dict(data)
        assert robot.n_joints == _MAX_JOINTS

    def test_builder_joint_limit(self) -> None:
        """Builder with 33 revolute joints must raise ValueError on build()."""
        builder = RobotBuilder("TooMany")
        for _ in range(_MAX_JOINTS + 1):
            builder.add_revolute(a=0.1)
        with pytest.raises(ValueError, match="[Mm]ax"):
            builder.build()

    def test_direct_init_limit(self) -> None:
        """RobotArm([...33 joints...]) must raise ValidationError."""
        joints = self._joint_list(_MAX_JOINTS + 1)
        with pytest.raises(ValidationError):
            RobotArm(joints)


# ============================================================================
# S3 — Coordinate Bounds
# ============================================================================


class TestCoordinateBounds:
    """S3: Non-finite or astronomically large IK target coordinates must be rejected.

    Attack vector: a crafted request supplies NaN, Inf, or a number like 1e308
    as a target coordinate.  Without bounds checking this would reach the IK
    solver and produce NaN joint angles that could command physical hardware to
    unsafe positions.
    """

    def setup_method(self) -> None:
        """Create a shared 2-link planar robot for every test in this class."""
        self.robot = create_two_link_planar(link1=1.0, link2=0.8)

    def test_inf_coordinate_rejected(self) -> None:
        """solve_ik([Inf, 0]) must raise ValidationError."""
        with pytest.raises(ValidationError):
            self.robot.solve_ik([float("inf"), 0.0])

    def test_nan_coordinate_rejected(self) -> None:
        """solve_ik([NaN, 0]) must raise ValidationError."""
        with pytest.raises(ValidationError):
            self.robot.solve_ik([float("nan"), 0.0])

    def test_huge_coordinate_rejected(self) -> None:
        """solve_ik([1e7, 0]) must raise ValidationError (exceeds 1e6 m limit)."""
        with pytest.raises(ValidationError, match="1e6"):
            self.robot.solve_ik([1e7, 0.0])

    def test_valid_coordinate_accepted(self) -> None:
        """solve_ik([1.0, 0.5]) must return an IKSolution without raising."""
        result = self.robot.solve_ik([1.0, 0.5])
        assert result is not None
        assert result.residual_error >= 0.0

    def test_agent_extreme_coord(self) -> None:
        """Coordinator with x=1e308 must return an error string and not crash."""
        coordinator = RoboticsCoordinator(self.robot)
        response = coordinator.process("Solve IK for x=1e308, y=1e308")
        assert isinstance(response, str), "Response must be a string"
        assert len(response) > 0, "Response must not be empty"
        assert "error" in response.lower(), (
            f"Expected error indication in response, got: {response!r}"
        )


# ============================================================================
# S4 — Log Injection
# ============================================================================


class TestLogInjection:
    """S4: User-supplied input must be sanitized before reaching log records.

    Attack vector: a newline or carriage-return in user input causes a log
    management system (SIEM, Splunk, ELK) to interpret the injected text as a
    separate, attacker-controlled log line, potentially hiding or spoofing
    legitimate security events.
    """

    def test_newline_sanitized(self) -> None:
        r"""sanitize_for_log: '\n' must be replaced with the visible ⏎ marker."""
        result = sanitize_for_log("hello\nworld")
        assert "\n" not in result, "Raw newline must not appear in sanitized output"
        assert "⏎" in result, "Expected ⏎ marker in place of newline"
        assert "hello" in result
        assert "world" in result

    def test_cr_sanitized(self) -> None:
        r"""sanitize_for_log: '\r' must be replaced with the ⏎ marker."""
        result = sanitize_for_log("hello\rworld")
        assert "\r" not in result
        assert "⏎" in result

    def test_crlf_sanitized(self) -> None:
        r"""sanitize_for_log: '\r\n' must produce a single ⏎ marker (not two)."""
        result = sanitize_for_log("line1\r\nline2")
        assert "\r" not in result
        assert "\n" not in result
        assert "⏎" in result

    def test_null_byte_removed(self) -> None:
        r"""sanitize_for_log: '\x00' must be stripped entirely from the output."""
        result = sanitize_for_log("hello\x00world")
        assert "\x00" not in result, "Null byte must not appear in sanitized output"
        assert "hello" in result
        assert "world" in result

    def test_long_input_truncated(self) -> None:
        """sanitize_for_log: a 600-character string must be truncated to ~500 chars."""
        long_input = "A" * 600
        result = sanitize_for_log(long_input)
        assert len(result) < len(long_input), (
            "Sanitized output must be shorter than the 600-char input"
        )
        assert result.endswith("…"), (
            "Truncated output must end with the '…' ellipsis marker"
        )

    def test_coordinator_sanitizes_input(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No raw newlines must appear in log records when user injects them."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        malicious = "solve IK\nfor x=1.0, y=0.5\nINJECTED-FAKE-LOG-LINE"

        with caplog.at_level(logging.DEBUG):
            coordinator.process(malicious)

        for record in caplog.records:
            msg = record.getMessage()
            assert "\n" not in msg, (
                f"Raw newline found in log record from '{record.name}': {msg!r}"
            )
            assert "\r" not in msg, (
                f"Raw carriage-return found in log record from '{record.name}': {msg!r}"
            )


# ============================================================================
# S5 — CSV Formula Injection
# ============================================================================


class TestCSVFormulaInjection:
    """S5: Joint names starting with formula characters must be tab-prefixed in CSV.

    Attack vector: a joint named ``=HYPERLINK("http://evil.com","click")``
    becomes an executable formula when the CSV is opened in Excel/LibreOffice,
    potentially exfiltrating data.  The OWASP mitigation is to prepend a tab.
    """

    def test_equals_prefix_blocked(self) -> None:
        """'=SUM(A1)' must be tab-prefixed so spreadsheets treat it as text."""
        result = _safe_csv_cell("=SUM(A1)")
        assert isinstance(result, str)
        assert result.startswith("\t"), "Formula cell must be prefixed with a tab"
        assert "=SUM(A1)" in result, "Original content must be preserved"

    def test_plus_prefix_blocked(self) -> None:
        """'+evil' must be tab-prefixed."""
        result = _safe_csv_cell("+evil")
        assert isinstance(result, str)
        assert result.startswith("\t")
        assert "+evil" in result

    def test_at_prefix_blocked(self) -> None:
        """'@evil' must be tab-prefixed."""
        result = _safe_csv_cell("@evil")
        assert isinstance(result, str)
        assert result.startswith("\t")
        assert "@evil" in result

    def test_normal_name_unchanged(self) -> None:
        """'J1_rad' must pass through _safe_csv_cell without modification."""
        assert _safe_csv_cell("J1_rad") == "J1_rad"

    def test_csv_roundtrip_with_formula_name(self) -> None:
        """Save/load round-trip with a formula joint name must preserve data exactly."""
        original_traj = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64
        )
        original_ts = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        formula_name = "=SUM(A1)"

        fd, csv_path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        try:
            save_trajectory_csv(
                csv_path,
                original_traj,
                timestamps=original_ts,
                joint_names=[formula_name, "J2_rad"],
            )
            loaded_traj, loaded_ts, _ = load_trajectory_csv(csv_path)
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

        np.testing.assert_allclose(
            loaded_traj, original_traj, rtol=1e-12,
            err_msg="Joint angle data must survive a CSV round-trip with formula names",
        )
        np.testing.assert_allclose(
            loaded_ts, original_ts, rtol=1e-12,
            err_msg="Timestamps must survive a CSV round-trip",
        )


# ============================================================================
# S6 — Audit Log Integrity
# ============================================================================


class TestAuditLogIntegrity:
    """S6: Callers must not be able to mutate the internal audit log.

    Attack surface: if ``get_audit_log()`` returned references to the internal
    dicts, a caller could silently rewrite past audit entries, destroying the
    integrity of the security event log.
    """

    @staticmethod
    def _registry_with_call() -> ToolRegistry:
        """Return a ToolRegistry that has already executed one tool call."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="dummy",
            description="No-op tool for testing.",
            function=lambda: "ok",
        ))
        registry.execute("dummy")
        return registry

    def test_get_audit_log_returns_copies(self) -> None:
        """Mutating a returned audit entry must not affect the internal log."""
        registry = self._registry_with_call()
        log_first = registry.get_audit_log()
        assert len(log_first) == 1
        original_tool_name = log_first[0]["tool"]

        # Attempt to tamper with the returned entry
        log_first[0]["tool"] = "HACKED"

        # A fresh retrieval must still show the original value
        log_second = registry.get_audit_log()
        assert log_second[0]["tool"] == original_tool_name, (
            "Internal audit entry must not be mutable from outside the registry"
        )

    def test_multiple_calls_independent(self) -> None:
        """Two consecutive get_audit_log() calls return independent copy lists."""
        registry = self._registry_with_call()

        log_a = registry.get_audit_log()
        log_b = registry.get_audit_log()

        # Tamper with the first copy
        log_a[0]["status"] = "TAMPERED"

        # The second copy must be unaffected
        assert log_b[0]["status"] != "TAMPERED", (
            "Two calls to get_audit_log() must return independent copies"
        )

    def test_request_id_captured(self) -> None:
        """Audit entries must record the correct request_id from request_context."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="tracked_tool",
            description="Tool used to verify request_id capture.",
            function=lambda: "result",
        ))

        fixed_rid = "security-test-rid-abc123"
        with request_context(fixed_rid):
            registry.execute("tracked_tool")

        log = registry.get_audit_log()
        assert len(log) == 1, "Exactly one audit entry expected"
        assert log[0]["request_id"] == fixed_rid, (
            f"Expected request_id={fixed_rid!r}, got {log[0]['request_id']!r}"
        )

    def test_clear_audit_log_works(self) -> None:
        """After clear_audit_log(), get_audit_log() must return an empty list."""
        registry = self._registry_with_call()
        assert len(registry.get_audit_log()) >= 1, "Pre-condition: log must have entries"

        registry.clear_audit_log()

        assert registry.get_audit_log() == [], (
            "Audit log must be empty after clear_audit_log()"
        )


# ============================================================================
# S8 — Joint Velocity Saturation
# ============================================================================


class TestJointVelocitySaturation:
    """S8: joint_velocities() must saturate output and raise on NaN inputs.

    Safety relevance: unsaturated or NaN joint velocities sent to physical
    motors cause hardware damage.  The safety guard must prevent both cases.
    """

    def setup_method(self) -> None:
        """Create a 2-DOF planar robot and JacobianComputer shared by all tests."""
        self.robot = create_two_link_planar(link1=1.0, link2=0.8)
        self.jac = JacobianComputer(self.robot)
        # A non-singular joint configuration
        self.q_good = [0.5, 0.3]

    def test_saturation_clamps_output(self) -> None:
        """With max_joint_velocity=0.1, all joint velocities must stay in [-0.1, +0.1]."""
        limit = 0.1
        dq = self.jac.joint_velocities(
            self.q_good,
            ee_velocity=[2.0, 2.0],
            max_joint_velocity=limit,
        )
        assert np.all(np.abs(dq) <= limit + 1e-12), (
            f"joint_velocities() exceeded saturation limit {limit}: {dq}"
        )

    def test_no_saturation_by_default(self) -> None:
        """Without max_joint_velocity, a large EE velocity produces unsaturated output."""
        tight_limit = 0.01
        dq_sat = self.jac.joint_velocities(
            self.q_good, ee_velocity=[10.0, 10.0], max_joint_velocity=tight_limit,
        )
        dq_raw = self.jac.joint_velocities(
            self.q_good, ee_velocity=[10.0, 10.0],
        )
        # Saturated output must respect the limit
        assert np.all(np.abs(dq_sat) <= tight_limit + 1e-12), (
            "Saturated velocities must not exceed the tight limit"
        )
        # Unsaturated output must be larger than the tight limit (for ee=[10,10])
        assert np.any(np.abs(dq_raw) > tight_limit), (
            "Unsaturated velocities with ee_velocity=[10,10] should exceed 0.01 rad/s"
        )

    def test_nan_raises_on_degenerate_input(self) -> None:
        """joint_velocities with NaN in ee_velocity must raise ValueError."""
        with pytest.raises(ValueError, match="non-finite"):
            self.jac.joint_velocities(
                self.q_good,
                ee_velocity=[float("nan"), 0.0],
            )

    def test_valid_velocity_returned(self) -> None:
        """A non-singular configuration must yield a finite (n_dof,) velocity array."""
        dq = self.jac.joint_velocities(
            self.q_good,
            ee_velocity=[0.1, 0.05],
        )
        assert dq.shape == (self.robot.n_dof,), (
            f"Expected shape ({self.robot.n_dof},), got {dq.shape}"
        )
        assert np.all(np.isfinite(dq)), f"Joint velocities must be finite: {dq}"


# ============================================================================
# R1 — Forward Kinematics NaN Guard
# ============================================================================


class TestForwardKinematicsNaNGuard:
    """R1: forward_kinematics() must raise immediately when DH params are non-finite.

    This is a defence-in-depth guard: even if the ``from_dict`` validation is
    somehow bypassed (e.g. by direct dataclass construction), the FK output
    check must still surface the bad parameter before NaN propagates to
    downstream modules (IK solver, Jacobian, trajectory planner).
    """

    def test_nan_dh_param_caught_at_fk(self) -> None:
        """Robot with alpha=NaN (bypassed from_dict) must raise in forward_kinematics."""
        jc = _simple_joint(alpha=float("nan"), a=1.0)
        robot = RobotArm([jc])  # __init__ checks only joint count
        with pytest.raises(ValidationError, match="non-finite"):
            robot.forward_kinematics([0.0])

    def test_inf_dh_param_caught_at_fk(self) -> None:
        """Robot with a=+Inf (bypassed from_dict) must raise in forward_kinematics."""
        jc = _simple_joint(alpha=0.0, a=float("inf"))
        robot = RobotArm([jc])
        with pytest.raises(ValidationError, match="non-finite"):
            robot.forward_kinematics([0.0])

    def test_valid_fk_works(self) -> None:
        """Normal finite DH parameters must return a fully finite EndEffectorPose."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        pose = robot.forward_kinematics([0.5, -0.3])
        assert np.all(np.isfinite(pose.position)), "FK position must be finite"
        assert np.all(np.isfinite(pose.rotation)), "FK rotation matrix must be finite"
        assert pose.transform.shape == (4, 4), "Transform must be 4×4"
        assert np.all(np.isfinite(pose.transform)), "Full transform matrix must be finite"
