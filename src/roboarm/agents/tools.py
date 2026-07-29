"""Tool definition and registry for AI agents.

Provides :class:`ToolDefinition` for describing callable tools with
JSON-schema-like parameter metadata, and :class:`ToolRegistry` for
registering, discovering, and executing those tools.

The registry also maintains two observability surfaces:

* **Per-tool metrics** (:meth:`ToolRegistry.get_metrics`) — cumulative call
  counts, success/failure counts, and total wall-clock time per tool.
* **Audit call log** (:meth:`ToolRegistry.get_audit_log`) — a bounded FIFO
  of every invocation with timestamps, request IDs, and outcomes.  This is
  stored in memory only; callers that need persistence should serialise the
  list returned by :meth:`get_audit_log` themselves.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from roboarm.agents._request_context import current_request_id
from roboarm.utils.log_event import log_event

logger = logging.getLogger(__name__)

# Maximum number of audit log entries retained in memory.
_AUDIT_MAX = 1000


@dataclass
class ToolDefinition:
    """A single callable tool exposed to an agent.

    Attributes:
        name: Unique tool identifier (e.g. ``"compute_fk"``).
        description: Human-readable explanation of what the tool does.
        parameters: JSON-schema-like dictionary describing accepted
            keyword arguments, their types, and whether they are required.
        function: The callable that implements the tool logic.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    function: Callable[..., Any] = field(repr=False, default=lambda **kw: None)


@dataclass
class ToolMetrics:
    """Cumulative performance counters for one tool.

    Attributes:
        calls: Total number of invocations (success + failure).
        successes: Invocations that returned without raising.
        failures: Invocations that raised an exception.
        total_duration_ms: Accumulated wall-clock time across all calls.
    """

    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of calls that succeeded (0.0 – 1.0).  ``NaN`` if no calls yet."""
        if self.calls == 0:
            return float("nan")
        return self.successes / self.calls

    @property
    def avg_duration_ms(self) -> float:
        """Mean wall-clock time per call.  ``NaN`` if no calls yet."""
        if self.calls == 0:
            return float("nan")
        return self.total_duration_ms / self.calls


class ToolRegistry:
    """Registry of callable tools with built-in metrics and audit logging.

    Tools are stored by name and can be registered, looked up, listed,
    and executed through a uniform interface.

    Observability surfaces:
    - :meth:`get_metrics` — per-tool counters (calls, success rate, avg time)
    - :meth:`get_audit_log` — bounded FIFO of every invocation record

    Example::

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="greet",
            description="Say hello",
            parameters={"name": {"type": "string"}},
            function=lambda name="World": f"Hello, {name}!",
        ))
        result = registry.execute("greet", name="Agent")

        metrics = registry.get_metrics()
        print(metrics["greet"].calls)      # 1
        print(metrics["greet"].successes)  # 1

        log = registry.get_audit_log()
        print(log[0]["tool"])              # 'greet'
        print(log[0]["status"])            # 'ok'
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._metrics: dict[str, ToolMetrics] = {}
        self._audit: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool definition.

        If a tool with the same name already exists it is overwritten
        and a warning is logged.

        Args:
            tool: The tool to register.
        """
        if tool.name in self._tools:
            logger.warning(
                "Overwriting existing tool %r in registry", tool.name,
            )
        self._tools[tool.name] = tool
        # Initialise metrics entry (preserve existing counters on overwrite)
        if tool.name not in self._metrics:
            self._metrics[tool.name] = ToolMetrics()
        logger.debug("Registered tool %r", tool.name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolDefinition | None:
        """Look up a tool by name.

        Args:
            name: The tool identifier.

        Returns:
            The :class:`ToolDefinition` if found, otherwise ``None``.
        """
        return self._tools.get(name)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a registered tool by name.

        Records timing, updates per-tool metrics, and appends an audit
        log entry for every invocation.

        Args:
            name: The tool identifier.
            **kwargs: Keyword arguments forwarded to the tool function.

        Returns:
            Whatever the tool function returns.

        Raises:
            KeyError: If *name* is not registered.
            Exception: Re-raises any exception from the underlying function
                after logging it.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool {name!r} is not registered")

        logger.debug("Executing tool %r with args %s", name, kwargs)

        m = self._metrics.setdefault(name, ToolMetrics())
        request_id = current_request_id()

        t0 = time.perf_counter()
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        try:
            result = tool.function(**kwargs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            m.calls += 1
            m.successes += 1
            m.total_duration_ms += elapsed_ms

            log_event(logger, logging.INFO, "tool_call",
                      tool=name,
                      status="ok",
                      duration_ms=round(elapsed_ms, 2))

            self._append_audit(
                tool=name,
                status="ok",
                duration_ms=round(elapsed_ms, 2),
                timestamp=timestamp,
                request_id=request_id,
            )
            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            m.calls += 1
            m.failures += 1
            m.total_duration_ms += elapsed_ms

            log_event(logger, logging.ERROR, "tool_call",
                      tool=name,
                      status="error",
                      error=type(exc).__name__,
                      duration_ms=round(elapsed_ms, 2))

            self._append_audit(
                tool=name,
                status="error",
                error=type(exc).__name__,
                duration_ms=round(elapsed_ms, 2),
                timestamp=timestamp,
                request_id=request_id,
            )
            logger.exception("Tool %r raised an exception", name)
            raise

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, ToolMetrics]:
        """Return a snapshot of per-tool performance counters.

        Returns:
            Dictionary mapping tool name → :class:`ToolMetrics`.  The
            metrics objects are the live instances; copy them if you need
            a frozen snapshot.

        Example::

            m = registry.get_metrics()
            for name, stats in m.items():
                print(f"{name}: {stats.calls} calls, "
                      f"{stats.success_rate:.0%} success, "
                      f"{stats.avg_duration_ms:.1f} ms avg")
        """
        return dict(self._metrics)

    def get_audit_log(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """Return recent tool invocation records.

        Each record is a dictionary with at minimum these keys:

        * ``"tool"`` — tool name
        * ``"status"`` — ``"ok"`` or ``"error"``
        * ``"duration_ms"`` — wall-clock time in milliseconds
        * ``"timestamp"`` — ISO-8601 UTC timestamp of the call start
        * ``"request_id"`` — active request ID from :func:`current_request_id`,
          or ``None`` when called outside a :func:`request_context`

        Args:
            last_n: Return only the *last_n* most recent entries.  Defaults
                to returning all retained entries.

        Returns:
            List of invocation records in chronological order.

        Example::

            log = registry.get_audit_log(last_n=10)
            failures = [e for e in log if e["status"] == "error"]
        """
        entries = self._audit[-last_n:] if last_n is not None else list(self._audit)
        return [dict(e) for e in entries]

    def reset_metrics(self) -> None:
        """Reset all per-tool counters to zero.

        The tool definitions are not affected.  The audit log is also
        not cleared; use :meth:`clear_audit_log` for that.
        """
        for m in self._metrics.values():
            m.calls = m.successes = m.failures = 0
            m.total_duration_ms = 0.0
        logger.debug("ToolRegistry metrics reset")

    def clear_audit_log(self) -> None:
        """Discard all retained audit log entries."""
        self._audit.clear()
        logger.debug("ToolRegistry audit log cleared")

    # ------------------------------------------------------------------
    # Standard helpers
    # ------------------------------------------------------------------

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools.

        Returns:
            Sorted list of tool names.
        """
        return sorted(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function schemas for all tools.

        Each schema follows the structure expected by the OpenAI
        function-calling API (``type``, ``function.name``,
        ``function.description``, ``function.parameters``).

        Returns:
            List of schema dictionaries, one per registered tool.
        """
        schemas: list[dict[str, Any]] = []
        for name in sorted(self._tools):
            tool = self._tools[name]
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters,
                    },
                },
            })
        return schemas

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_tools()})"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _append_audit(self, **entry: Any) -> None:
        """Append *entry* to the audit log, evicting oldest if over limit."""
        self._audit.append(entry)
        if len(self._audit) > _AUDIT_MAX:
            self._audit = self._audit[-_AUDIT_MAX:]
