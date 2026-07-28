"""Tests for the observability improvements: metrics, audit log, and structured logging.

Covers:
- A2: ToolRegistry audit call log
- A5: ToolRegistry per-tool metrics
- L3: log_event helper and structured logging
- F12: compare_solvers uses solver's own computation_time_ms
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from roboarm.agents._request_context import request_context
from roboarm.agents.tools import ToolDefinition, ToolMetrics, ToolRegistry
from roboarm.utils.log_event import log_event

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_registry() -> ToolRegistry:
    """A ToolRegistry with two trivial tools for testing."""
    reg = ToolRegistry()

    reg.register(ToolDefinition(
        name="add",
        description="Add two numbers",
        parameters={"a": {"type": "number"}, "b": {"type": "number"}},
        function=lambda a, b: a + b,
    ))

    def _fail(**_kw: object) -> None:
        raise RuntimeError("deliberate failure")

    reg.register(ToolDefinition(
        name="broken",
        description="Always raises",
        parameters={},
        function=_fail,
    ))
    return reg


# ---------------------------------------------------------------------------
# A5: Per-tool metrics
# ---------------------------------------------------------------------------

class TestToolMetrics:
    """ToolRegistry.get_metrics() — per-tool call counters."""

    def test_metrics_exist_after_registration(self, simple_registry: ToolRegistry) -> None:
        metrics = simple_registry.get_metrics()
        assert "add" in metrics
        assert "broken" in metrics

    def test_zero_counts_before_execution(self, simple_registry: ToolRegistry) -> None:
        m = simple_registry.get_metrics()["add"]
        assert m.calls == 0
        assert m.successes == 0
        assert m.failures == 0
        assert m.total_duration_ms == 0.0

    def test_success_increments_counters(self, simple_registry: ToolRegistry) -> None:
        simple_registry.execute("add", a=1, b=2)
        m = simple_registry.get_metrics()["add"]
        assert m.calls == 1
        assert m.successes == 1
        assert m.failures == 0
        assert m.total_duration_ms >= 0.0

    def test_multiple_successes_accumulate(self, simple_registry: ToolRegistry) -> None:
        for _ in range(5):
            simple_registry.execute("add", a=1, b=1)
        m = simple_registry.get_metrics()["add"]
        assert m.calls == 5
        assert m.successes == 5

    def test_failure_increments_failure_counter(self, simple_registry: ToolRegistry) -> None:
        with pytest.raises(RuntimeError):
            simple_registry.execute("broken")
        m = simple_registry.get_metrics()["broken"]
        assert m.calls == 1
        assert m.failures == 1
        assert m.successes == 0

    def test_success_rate_is_fraction(self, simple_registry: ToolRegistry) -> None:
        simple_registry.execute("add", a=1, b=1)
        simple_registry.execute("add", a=2, b=2)
        m = simple_registry.get_metrics()["add"]
        assert m.success_rate == pytest.approx(1.0)

    def test_success_rate_nan_before_calls(self, simple_registry: ToolRegistry) -> None:
        m = simple_registry.get_metrics()["add"]
        assert math.isnan(m.success_rate)

    def test_avg_duration_nan_before_calls(self, simple_registry: ToolRegistry) -> None:
        m = simple_registry.get_metrics()["add"]
        assert math.isnan(m.avg_duration_ms)

    def test_avg_duration_positive_after_call(self, simple_registry: ToolRegistry) -> None:
        simple_registry.execute("add", a=1, b=1)
        m = simple_registry.get_metrics()["add"]
        assert m.avg_duration_ms >= 0.0

    def test_reset_metrics_clears_counters(self, simple_registry: ToolRegistry) -> None:
        simple_registry.execute("add", a=1, b=1)
        simple_registry.reset_metrics()
        m = simple_registry.get_metrics()["add"]
        assert m.calls == 0
        assert m.total_duration_ms == 0.0

    def test_reset_keeps_tools_registered(self, simple_registry: ToolRegistry) -> None:
        simple_registry.reset_metrics()
        # Tools should still be callable
        assert simple_registry.execute("add", a=3, b=4) == 7

    def test_metrics_type(self, simple_registry: ToolRegistry) -> None:
        m = simple_registry.get_metrics()
        for v in m.values():
            assert isinstance(v, ToolMetrics)

    def test_mixed_success_and_failure(self, simple_registry: ToolRegistry) -> None:
        simple_registry.execute("add", a=1, b=1)
        with pytest.raises(RuntimeError):
            simple_registry.execute("broken")
        add_m = simple_registry.get_metrics()["add"]
        broken_m = simple_registry.get_metrics()["broken"]
        assert add_m.successes == 1 and add_m.failures == 0
        assert broken_m.successes == 0 and broken_m.failures == 1


# ---------------------------------------------------------------------------
# A2: Audit call log
# ---------------------------------------------------------------------------

class TestAuditLog:
    """ToolRegistry.get_audit_log() — invocation history."""

    def test_empty_before_any_call(self, simple_registry: ToolRegistry) -> None:
        assert simple_registry.get_audit_log() == []

    def test_entry_added_on_success(self, simple_registry: ToolRegistry) -> None:
        simple_registry.execute("add", a=1, b=2)
        log = simple_registry.get_audit_log()
        assert len(log) == 1
        assert log[0]["tool"] == "add"
        assert log[0]["status"] == "ok"

    def test_entry_added_on_failure(self, simple_registry: ToolRegistry) -> None:
        with pytest.raises(RuntimeError):
            simple_registry.execute("broken")
        log = simple_registry.get_audit_log()
        assert len(log) == 1
        assert log[0]["status"] == "error"
        assert log[0]["error"] == "RuntimeError"

    def test_entries_are_chronological(self, simple_registry: ToolRegistry) -> None:
        simple_registry.execute("add", a=1, b=1)
        simple_registry.execute("add", a=2, b=2)
        log = simple_registry.get_audit_log()
        assert len(log) == 2
        # Timestamps should be non-decreasing
        assert log[0]["timestamp"] <= log[1]["timestamp"]

    def test_last_n_returns_tail(self, simple_registry: ToolRegistry) -> None:
        for i in range(5):
            simple_registry.execute("add", a=i, b=i)
        log = simple_registry.get_audit_log(last_n=3)
        assert len(log) == 3

    def test_last_n_none_returns_all(self, simple_registry: ToolRegistry) -> None:
        for i in range(4):
            simple_registry.execute("add", a=i, b=i)
        log = simple_registry.get_audit_log(last_n=None)
        assert len(log) == 4

    def test_entry_has_duration_ms(self, simple_registry: ToolRegistry) -> None:
        simple_registry.execute("add", a=1, b=1)
        entry = simple_registry.get_audit_log()[0]
        assert "duration_ms" in entry
        assert isinstance(entry["duration_ms"], float)
        assert entry["duration_ms"] >= 0.0

    def test_entry_has_timestamp(self, simple_registry: ToolRegistry) -> None:
        simple_registry.execute("add", a=1, b=1)
        entry = simple_registry.get_audit_log()[0]
        assert "timestamp" in entry
        # ISO-8601 strings contain 'T'
        assert "T" in entry["timestamp"]

    def test_request_id_captured_inside_context(self, simple_registry: ToolRegistry) -> None:
        with request_context("test-audit-rid") as rid:
            simple_registry.execute("add", a=1, b=1)
        entry = simple_registry.get_audit_log()[0]
        assert entry["request_id"] == rid

    def test_request_id_none_outside_context(self, simple_registry: ToolRegistry) -> None:
        simple_registry.execute("add", a=1, b=1)
        entry = simple_registry.get_audit_log()[0]
        assert entry["request_id"] is None

    def test_clear_audit_log(self, simple_registry: ToolRegistry) -> None:
        simple_registry.execute("add", a=1, b=1)
        assert len(simple_registry.get_audit_log()) == 1
        simple_registry.clear_audit_log()
        assert simple_registry.get_audit_log() == []

    def test_audit_log_is_copy(self, simple_registry: ToolRegistry) -> None:
        """Mutating the returned list must not affect the internal log."""
        simple_registry.execute("add", a=1, b=1)
        log = simple_registry.get_audit_log()
        log.clear()
        # Internal log should still have 1 entry
        assert len(simple_registry.get_audit_log()) == 1


# ---------------------------------------------------------------------------
# L3: log_event helper
# ---------------------------------------------------------------------------

class TestLogEvent:
    """log_event() emits structured key=value lines."""

    def test_emits_event_key(self, caplog: pytest.LogCaptureFixture) -> None:
        test_logger = logging.getLogger("test.log_event")
        with caplog.at_level(logging.INFO, logger="test.log_event"):
            log_event(test_logger, logging.INFO, "my_event", foo="bar")
        assert any("event=my_event" in r.message for r in caplog.records)

    def test_includes_string_field(self, caplog: pytest.LogCaptureFixture) -> None:
        test_logger = logging.getLogger("test.log_event.str")
        with caplog.at_level(logging.INFO, logger="test.log_event.str"):
            log_event(test_logger, logging.INFO, "test", solver="dls")
        assert any("solver='dls'" in r.message for r in caplog.records)

    def test_includes_int_field(self, caplog: pytest.LogCaptureFixture) -> None:
        test_logger = logging.getLogger("test.log_event.int")
        with caplog.at_level(logging.INFO, logger="test.log_event.int"):
            log_event(test_logger, logging.INFO, "test", iterations=8)
        assert any("iterations=8" in r.message for r in caplog.records)

    def test_includes_float_field(self, caplog: pytest.LogCaptureFixture) -> None:
        test_logger = logging.getLogger("test.log_event.float")
        with caplog.at_level(logging.INFO, logger="test.log_event.float"):
            log_event(test_logger, logging.INFO, "test", error=2.74e-7)
        assert any("error=" in r.message for r in caplog.records)

    def test_suppressed_below_level(self, caplog: pytest.LogCaptureFixture) -> None:
        test_logger = logging.getLogger("test.log_event.suppress")
        with caplog.at_level(logging.WARNING, logger="test.log_event.suppress"):
            log_event(test_logger, logging.DEBUG, "hidden_event", x=1)
        assert not any("hidden_event" in r.message for r in caplog.records)

    def test_bool_field_unquoted(self, caplog: pytest.LogCaptureFixture) -> None:
        test_logger = logging.getLogger("test.log_event.bool")
        with caplog.at_level(logging.INFO, logger="test.log_event.bool"):
            log_event(test_logger, logging.INFO, "test", success=True)
        assert any("success=True" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# L3: Structured logging from solvers (ik_solve events)
# ---------------------------------------------------------------------------

class TestSolverStructuredLogging:
    """IK solvers emit event=ik_solve key=value log lines."""

    def _make_target(self, x: float, y: float) -> object:
        from roboarm.core.types import EndEffectorPose
        pos = np.array([x, y, 0.0])
        T = np.eye(4)
        T[:3, 3] = pos
        return EndEffectorPose(position=pos, rotation=np.eye(3), transform=T)

    @pytest.mark.parametrize("solver_name", [
        "damped_least_squares",
        "jacobian_pseudoinverse",
        "ccd",
        "fabrik",
    ])
    def test_success_emits_ik_solve_event(
        self,
        solver_name: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import roboarm.kinematics.solvers  # noqa: F401
        from roboarm.kinematics.solvers.registry import IKSolverRegistry
        from roboarm.robots.two_link_planar import create_two_link_planar

        robot = create_two_link_planar()
        solver = IKSolverRegistry.create(solver_name, robot)
        target = self._make_target(1.0, 0.5)

        with caplog.at_level(logging.DEBUG):
            solver.solve(target)  # type: ignore[arg-type]

        messages = " ".join(r.message for r in caplog.records)
        assert "event=ik_solve" in messages

    def test_failure_emits_warning_level(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import roboarm.kinematics.solvers  # noqa: F401
        from roboarm.kinematics.solvers.registry import IKSolverRegistry
        from roboarm.robots.two_link_planar import create_two_link_planar

        robot = create_two_link_planar()
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        target = self._make_target(99.0, 99.0)  # unreachable

        with caplog.at_level(logging.WARNING):
            solver.solve(target)  # type: ignore[arg-type]

        warning_messages = " ".join(
            r.message for r in caplog.records if r.levelno == logging.WARNING
        )
        assert "event=ik_solve" in warning_messages
        assert "success=False" in warning_messages


# ---------------------------------------------------------------------------
# F12: compare_solvers uses solver's own computation_time_ms
# ---------------------------------------------------------------------------

class TestCompareSolversTimingConsistency:
    """compare_solvers table uses result.computation_time_ms, not external perf_counter."""

    def test_compare_response_contains_time_column(self) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import roboarm.kinematics.solvers  # noqa: F401
        from roboarm.agents import RoboticsCoordinator
        from roboarm.robots.two_link_planar import create_two_link_planar

        robot = create_two_link_planar()
        coord = RoboticsCoordinator(robot)
        resp = coord.process("Compare all solvers for x=0.8, y=0.6")

        assert "Time (ms)" in resp
        # Every solver row should have a numeric time value (no dashes/blanks)
        lines = [ln for ln in resp.splitlines() if "analytical_2link" in ln]
        assert lines, "analytical_2link row missing from compare output"
        # The time column should contain a float, not a placeholder
        parts = lines[0].split()
        # Find the float that represents time — should parse cleanly
        floats_found = []
        for p in parts:
            try:
                floats_found.append(float(p))
            except ValueError:
                pass
        assert len(floats_found) >= 1, "No numeric time value in compare row"
