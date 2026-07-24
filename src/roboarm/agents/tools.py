"""Tool definition and registry for AI agents.

Provides :class:`ToolDefinition` for describing callable tools with
JSON-schema-like parameter metadata, and :class:`ToolRegistry` for
registering, discovering, and executing those tools.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


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


class ToolRegistry:
    """Registry of callable tools.

    Tools are stored by name and can be registered, looked up, listed,
    and executed through a uniform interface.

    Example::

        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="greet",
            description="Say hello",
            parameters={"name": {"type": "string"}},
            function=lambda name="World": f"Hello, {name}!",
        ))
        result = registry.execute("greet", name="Agent")
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

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
        logger.debug("Registered tool %r", tool.name)

    def get(self, name: str) -> ToolDefinition | None:
        """Look up a tool by name.

        Args:
            name: The tool identifier.

        Returns:
            The :class:`ToolDefinition` if found, otherwise ``None``.
        """
        return self._tools.get(name)

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a registered tool by name.

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
        try:
            result = tool.function(**kwargs)
            logger.debug("Tool %r returned successfully", name)
            return result
        except Exception:
            logger.exception("Tool %r raised an exception", name)
            raise

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
