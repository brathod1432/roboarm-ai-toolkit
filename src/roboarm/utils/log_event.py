"""Structured key=value event logging helper.

Provides :func:`log_event` as a lightweight alternative to plain-string
logger calls.  Every call emits a single line in the form::

    event=ik_solve solver=damped_least_squares success=True iterations=8
    error=2.74e-07 duration_ms=0.73

This format is:

* Human-readable in a terminal without tooling
* Machine-parseable by log aggregators (Elasticsearch, Datadog, Splunk,
  CloudWatch Insights) using their native key=value parsers — no regex
  required.
* Fully backward compatible: existing plain-string log calls are
  unaffected; only callers that explicitly use :func:`log_event` emit
  structured lines.

Usage::

    from roboarm.utils.log_event import log_event
    import logging

    logger = logging.getLogger(__name__)

    log_event(logger, logging.INFO, "ik_solve",
              solver="damped_least_squares",
              success=True,
              iterations=8,
              error=2.74e-07,
              duration_ms=0.73)
"""

from __future__ import annotations

import logging
from typing import Any


def log_event(
    logger: logging.Logger | logging.LoggerAdapter,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Emit a structured ``event=X key=value …`` log line.

    Args:
        logger: The logger or adapter to emit through.
        level: Python logging level (e.g. ``logging.INFO``).
        event: Short snake_case event name (e.g. ``"ik_solve"``).
        **fields: Arbitrary key-value pairs appended after the event.
            Values are formatted with ``repr()`` so strings are quoted
            and numbers are rendered without quotes — producing clean
            machine-parseable output.

    Example::

        log_event(logger, logging.WARNING, "ik_failed",
                  solver="ccd", iterations=500, error=0.0342)
        # → WARNING  event=ik_failed solver='ccd' iterations=500 error=0.0342
    """
    if not logger.isEnabledFor(level):
        return  # avoid building the string when the level is suppressed
    parts = [f"event={event}"]
    for k, v in fields.items():
        if isinstance(v, float):
            # Compact scientific notation for small/large floats
            parts.append(f"{k}={v:.6g}")
        elif isinstance(v, bool):
            parts.append(f"{k}={v}")
        elif isinstance(v, int):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v!r}")
    logger.log(level, " ".join(parts))


def sanitize_for_log(text: str, max_length: int = 500) -> str:
    """Sanitize a user-supplied string before including it in a log message.

    Prevents log injection attacks by:
    * Replacing newlines and carriage returns with a visible marker.
    * Replacing null bytes and other control characters.
    * Truncating to *max_length* characters.

    Args:
        text: Raw user input string.
        max_length: Maximum length of the returned string.

    Returns:
        A safe single-line string suitable for log output.
    """
    if not isinstance(text, str):
        text = repr(text)
    # Replace line-ending control chars with visible markers
    sanitized = (
        text
        .replace("\r\n", "⏎")
        .replace("\n", "⏎")
        .replace("\r", "⏎")
        .replace("\t", "→")
        .replace("\x00", "")  # strip null bytes
    )
    # Strip other ASCII control characters (0-31 except visible ones)
    sanitized = "".join(
        ch if ch >= " " or ch in "\t" else "?" for ch in sanitized
    )
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "…"
    return sanitized
