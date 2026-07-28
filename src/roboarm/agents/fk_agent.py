"""Forward-kinematics specialist agent.

:class:`FKAgent` parses natural-language queries related to forward
kinematics, extracts joint angles from the text, and invokes the
``compute_fk`` tool to return the end-effector position.
"""

from __future__ import annotations

import logging
import re

from roboarm.agents.base_agent import AgentMessage, BaseAgent

logger = logging.getLogger(__name__)

# Keywords that signal a forward-kinematics intent
_FK_KEYWORDS = {"fk", "forward", "angles", "position", "compute", "kinematics"}


class FKAgent(BaseAgent):
    """Specialist agent for forward kinematics queries.

    The agent uses keyword-based intent parsing (no external LLM) to
    detect FK requests and extract joint angles from the user input.

    Recognised patterns:

    * Bracket notation: ``[0.5, -0.3]``
    * Parenthesis notation: ``(0.5, -0.3)``
    * Inline list: ``angles 0.5, -0.3``
    * Keywords: *fk*, *forward*, *angles*, *position*, *compute*

    Args:
        name: Agent display name.
        tools: Registry containing at least the ``compute_fk`` tool.

    Example::

        agent = FKAgent("FK Agent", tools)
        response = agent.process("Compute FK for angles [0.5, -0.3]")
    """

    def process(self, user_input: str) -> str:
        """Parse *user_input* for FK intent and execute the tool.

        Returns:
            Human-readable result string, or an error / help message.
        """
        self._memory.add(AgentMessage(role="user", content=user_input))
        logger.info("FKAgent received: %s", user_input)

        lower = user_input.lower()

        # Check for FK intent
        if not self._has_fk_intent(lower):
            msg = (
                "I specialise in forward kinematics. Please include "
                "keywords like 'fk', 'forward kinematics', or 'compute "
                "position' along with joint angles in brackets, e.g. "
                "[0.5, -0.3]."
            )
            self._memory.add(AgentMessage(role="assistant", content=msg))
            return msg

        # Extract angles — first from the current query, then from memory
        angles = self._extract_angles(user_input)
        if angles is None:
            angles = self._recall_last_angles()
            if angles is not None:
                logger.debug("FKAgent: using angles from conversation memory")
        if angles is None:
            msg = (
                "I could not find joint angles in your query. Please "
                "provide them in brackets, e.g. [0.5, -0.3], or as a "
                "comma-separated list after 'angles'."
            )
            self._memory.add(AgentMessage(role="assistant", content=msg))
            return msg

        # Execute the tool
        try:
            result = self._tools.execute("compute_fk", angles=angles)
            response = str(result)
        except KeyError:
            response = (
                "Error: the compute_fk tool is not available in the "
                "tool registry."
            )
        except Exception as exc:
            logger.exception("FKAgent tool execution failed")
            response = f"Error computing forward kinematics: {exc}"

        self._memory.add(
            AgentMessage(
                role="assistant",
                content=response,
                tool_name="compute_fk",
                tool_args={"angles": angles},
            ),
        )
        return response

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _recall_last_angles(self) -> list[float] | None:
        """Search recent assistant messages for the last set of angles used.

        Enables queries like "repeat with the same angles" to work without
        the user repeating the values.
        """
        for msg in reversed(self._memory.get_history()):
            if msg.role == "assistant" and msg.tool_args:
                angles = msg.tool_args.get("angles")
                if isinstance(angles, list) and angles:
                    return angles
        return None

    @staticmethod
    def _has_fk_intent(text: str) -> bool:
        """Return ``True`` if *text* contains FK-related keywords."""
        tokens = set(re.findall(r"[a-z]+", text))
        return bool(tokens & _FK_KEYWORDS)

    @staticmethod
    def _extract_angles(text: str) -> list[float] | None:
        """Extract a list of numeric angles from *text*.

        Tries several patterns in order of specificity:
        1. Bracket list ``[0.5, -0.3, 1.2]``
        2. Parenthesis list ``(0.5, -0.3)``
        3. Bare comma / space separated numbers after the word "angles"
        4. Any sequence of two or more numbers in the string
        """
        # Pattern 1: numbers inside square brackets
        bracket_match = re.search(
            r"\[([^\]]+)\]", text,
        )
        if bracket_match:
            return _parse_number_list(bracket_match.group(1))

        # Pattern 2: numbers inside parentheses
        paren_match = re.search(
            r"\(([^)]+)\)", text,
        )
        if paren_match:
            nums = _parse_number_list(paren_match.group(1))
            if nums:
                return nums

        # Pattern 3: numbers after the word "angles"
        angles_match = re.search(
            r"angles?\s+([\d\s,.\-+eE]+)", text, re.IGNORECASE,
        )
        if angles_match:
            nums = _parse_number_list(angles_match.group(1))
            if nums:
                return nums

        # Pattern 4: fallback — collect all float-like tokens
        all_nums = re.findall(
            r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text,
        )
        if len(all_nums) >= 2:
            try:
                return [float(n) for n in all_nums]
            except ValueError:
                pass

        return None


def _parse_number_list(raw: str) -> list[float] | None:
    """Parse a string of comma / space separated numbers.

    Returns:
        List of floats, or ``None`` if parsing fails.
    """
    parts = re.split(r"[,\s]+", raw.strip())
    values: list[float] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values if values else None
