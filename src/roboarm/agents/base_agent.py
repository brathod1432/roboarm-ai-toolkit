"""Base agent abstractions for the robotics AI layer.

Provides :class:`AgentMessage` for structured conversation messages,
:class:`AgentMemory` for bounded conversation history, and
:class:`BaseAgent`, the abstract base class all specialist agents extend.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from roboarm.agents.tools import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """A single message in an agent conversation.

    Attributes:
        role: The message author role -- one of ``"user"``,
            ``"assistant"``, ``"system"``, or ``"tool_result"``.
        content: Textual content of the message.
        tool_name: Name of the tool that was called (for tool-result
            messages) or that the assistant wants to call.
        tool_args: Arguments passed to the tool, if applicable.
    """

    role: str
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = field(default=None)


class AgentMemory:
    """Simple bounded conversation memory.

    Stores up to *max_messages* :class:`AgentMessage` instances.  When
    the limit is exceeded the oldest messages are discarded (FIFO).

    Args:
        max_messages: Maximum number of messages to retain.

    Example::

        mem = AgentMemory(max_messages=10)
        mem.add(AgentMessage(role="user", content="Hello"))
        assert len(mem.get_history()) == 1
    """

    def __init__(self, max_messages: int = 50) -> None:
        self._messages: List[AgentMessage] = []
        self._max = max_messages

    def add(self, msg: AgentMessage) -> None:
        """Append a message, evicting the oldest if the limit is hit.

        Args:
            msg: The message to store.
        """
        self._messages.append(msg)
        if len(self._messages) > self._max:
            excess = len(self._messages) - self._max
            self._messages = self._messages[excess:]
            logger.debug(
                "AgentMemory trimmed %d oldest message(s)", excess,
            )

    def get_history(self) -> List[AgentMessage]:
        """Return a copy of the stored messages in chronological order.

        Returns:
            List of :class:`AgentMessage` instances.
        """
        return list(self._messages)

    def clear(self) -> None:
        """Remove all stored messages."""
        count = len(self._messages)
        self._messages.clear()
        logger.debug("AgentMemory cleared (%d messages removed)", count)

    @property
    def size(self) -> int:
        """Number of messages currently stored."""
        return len(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return (
            f"AgentMemory(size={len(self._messages)}, max={self._max})"
        )


class BaseAgent(ABC):
    """Abstract base class for all specialist agents.

    Every agent has a *name*, a :class:`ToolRegistry` giving it access
    to callable tools, and an :class:`AgentMemory` for conversation
    history.

    Subclasses must implement :meth:`process` to handle user queries.

    Args:
        name: Human-readable agent name.
        tools: Registry of tools the agent may invoke.
    """

    def __init__(self, name: str, tools: ToolRegistry) -> None:
        self._name = name
        self._tools = tools
        self._memory = AgentMemory()
        logger.debug("Initialised agent %r", name)

    @abstractmethod
    def process(self, user_input: str) -> str:
        """Process a user message and return the agent response.

        Implementations should:

        1. Parse the user input for intent and parameters.
        2. Call the appropriate tool(s) via ``self._tools``.
        3. Record messages in ``self._memory``.
        4. Return a human-readable response string.

        Args:
            user_input: Raw text from the user.

        Returns:
            Agent response as a formatted string.
        """

    @property
    def name(self) -> str:
        """Human-readable agent name."""
        return self._name

    @property
    def memory(self) -> AgentMemory:
        """Agent conversation memory."""
        return self._memory
