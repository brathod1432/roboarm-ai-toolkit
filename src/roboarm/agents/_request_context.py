"""Request-scoped context for correlating log lines across agent calls.

Usage::

    with request_context():
        coordinator.process("Solve IK for x=1.0, y=0.5")
    # All log lines emitted during process() share the same request_id.

The context variable is automatically read by the logging adapter returned
by :func:`get_logger`.  No propagation is needed — Python's ``contextvars``
module makes the value visible to every call-stack frame within the same
thread or asyncio task.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar

# Module-level ContextVar: value is None when no request is active.
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def current_request_id() -> str | None:
    """Return the active request ID, or ``None`` outside a request context."""
    return _request_id.get()


@contextmanager
def request_context(request_id: str | None = None) -> Generator[str, None, None]:
    """Context manager that sets a request ID for the duration of a block.

    Args:
        request_id: Explicit ID to use.  Defaults to a fresh UUID4.

    Yields:
        The active request ID string.

    Example::

        with request_context() as rid:
            coordinator.process("Describe the robot")
        # rid is a UUID like '3f2a1b...'
    """
    rid = request_id or str(uuid.uuid4())
    token = _request_id.set(rid)
    try:
        yield rid
    finally:
        _request_id.reset(token)


class _RequestAdapter(logging.LoggerAdapter):
    """LoggerAdapter that prepends ``[request_id]`` to every message."""

    def process(
        self, msg: object, kwargs: dict[str, object]
    ) -> tuple[object, dict[str, object]]:
        rid = current_request_id()
        prefix = f"[{rid[:8]}] " if rid else ""
        return f"{prefix}{msg}", kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return a :class:`logging.LoggerAdapter` that injects the request ID.

    Drop-in replacement for ``logging.getLogger(name)`` inside the agents
    package.  When used outside a :func:`request_context`, the prefix is
    simply omitted.

    Args:
        name: Logger name (typically ``__name__``).

    Returns:
        An adapter that prepends ``[<first-8-chars-of-request-id>]`` to
        every log message when a request context is active.
    """
    return _RequestAdapter(logging.getLogger(name), {})
