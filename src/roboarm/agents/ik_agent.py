"""Inverse-kinematics specialist agent.

:class:`IKAgent` parses natural-language queries related to inverse
kinematics, extracts target coordinates, and invokes the ``solve_ik``
or ``compare_solvers`` tool.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from roboarm.agents.base_agent import AgentMessage, BaseAgent
from roboarm.agents.tools import ToolRegistry

logger = logging.getLogger(__name__)

# Keywords that signal an IK intent
_IK_KEYWORDS = {
    "ik", "inverse", "solve", "reach", "target",
    "kinematics", "position", "point",
}

# Keywords that signal a comparison intent
_COMPARE_KEYWORDS = {"compare", "benchmark", "all", "versus", "vs"}


class IKAgent(BaseAgent):
    """Specialist agent for inverse kinematics queries.

    Uses keyword-based intent parsing (no external LLM) to detect IK
    requests, extract target coordinates, and invoke the appropriate
    tool.

    Recognised coordinate patterns:

    * Named: ``x=1.0, y=0.5`` or ``x: 1.0  y: 0.5``
    * Bracket list: ``[1.0, 0.5]`` (interpreted as ``[x, y]`` or
      ``[x, y, z]``)
    * Parenthesis: ``(1.0, 0.5)``
    * Keyword "target" followed by numbers

    Comparison mode is activated when the input also contains *compare*,
    *benchmark*, or *all*.

    Args:
        name: Agent display name.
        tools: Registry containing ``solve_ik`` and ``compare_solvers``.

    Example::

        agent = IKAgent("IK Agent", tools)
        agent.process("Solve IK for x=1.0, y=0.5")
    """

    def process(self, user_input: str) -> str:
        """Parse *user_input* for IK intent and execute the tool.

        Returns:
            Human-readable result string, or an error / help message.
        """
        self._memory.add(AgentMessage(role="user", content=user_input))
        logger.info("IKAgent received: %s", user_input)

        lower = user_input.lower()

        # Check for IK intent
        if not self._has_ik_intent(lower):
            msg = (
                "I specialise in inverse kinematics. Please include "
                "keywords like 'ik', 'inverse', 'solve', 'reach', or "
                "'target' along with target coordinates, e.g. "
                "x=1.0, y=0.5."
            )
            self._memory.add(AgentMessage(role="assistant", content=msg))
            return msg

        # Determine if this is a comparison request
        is_compare = self._has_compare_intent(lower)

        # Extract coordinates
        coords = self._extract_coordinates(user_input)
        if coords is None:
            msg = (
                "I could not find target coordinates in your query. "
                "Please provide them as x=1.0, y=0.5 or in brackets "
                "like [1.0, 0.5]."
            )
            self._memory.add(AgentMessage(role="assistant", content=msg))
            return msg

        target_x, target_y = coords[0], coords[1]
        target_z = coords[2] if len(coords) > 2 else None

        # Extract optional solver name
        solver_name = self._extract_solver_name(lower)

        # Execute the appropriate tool
        if is_compare:
            return self._run_compare(target_x, target_y, target_z)
        return self._run_solve(target_x, target_y, target_z, solver_name)

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _run_solve(
        self,
        target_x: float,
        target_y: float,
        target_z: Optional[float],
        solver_name: Optional[str],
    ) -> str:
        """Invoke the ``solve_ik`` tool and record the result."""
        kwargs: Dict[str, object] = {
            "target_x": target_x,
            "target_y": target_y,
        }
        if target_z is not None:
            kwargs["target_z"] = target_z
        if solver_name is not None:
            kwargs["solver_name"] = solver_name

        try:
            result = self._tools.execute("solve_ik", **kwargs)
            response = str(result)
        except KeyError:
            response = (
                "Error: the solve_ik tool is not available in the "
                "tool registry."
            )
        except Exception as exc:
            logger.exception("IKAgent solve_ik execution failed")
            response = f"Error solving inverse kinematics: {exc}"

        self._memory.add(
            AgentMessage(
                role="assistant",
                content=response,
                tool_name="solve_ik",
                tool_args=dict(kwargs),
            ),
        )
        return response

    def _run_compare(
        self,
        target_x: float,
        target_y: float,
        target_z: Optional[float],
    ) -> str:
        """Invoke the ``compare_solvers`` tool and record the result."""
        kwargs: Dict[str, object] = {
            "target_x": target_x,
            "target_y": target_y,
        }
        if target_z is not None:
            kwargs["target_z"] = target_z

        try:
            result = self._tools.execute("compare_solvers", **kwargs)
            response = str(result)
        except KeyError:
            response = (
                "Error: the compare_solvers tool is not available in "
                "the tool registry."
            )
        except Exception as exc:
            logger.exception("IKAgent compare_solvers execution failed")
            response = f"Error comparing solvers: {exc}"

        self._memory.add(
            AgentMessage(
                role="assistant",
                content=response,
                tool_name="compare_solvers",
                tool_args=dict(kwargs),
            ),
        )
        return response

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    @staticmethod
    def _has_ik_intent(text: str) -> bool:
        """Return ``True`` if *text* contains IK-related keywords."""
        tokens = set(re.findall(r"[a-z]+", text))
        # Also check for "x=" or "y=" patterns as strong IK signals
        if re.search(r"[xy]\s*=", text):
            return True
        return bool(tokens & _IK_KEYWORDS)

    @staticmethod
    def _has_compare_intent(text: str) -> bool:
        """Return ``True`` if *text* also requests a solver comparison."""
        tokens = set(re.findall(r"[a-z]+", text))
        return bool(tokens & _COMPARE_KEYWORDS)

    # ------------------------------------------------------------------
    # Coordinate extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_coordinates(text: str) -> Optional[List[float]]:
        """Extract ``[x, y]`` or ``[x, y, z]`` from *text*.

        Tries several patterns in order of specificity:
        1. Named coordinates: ``x=1.0``, ``y=0.5``, ``z=0.2``
        2. Bracket/parenthesis list: ``[1.0, 0.5]``
        3. Keyword "target"/"position" followed by numbers
        4. Fallback: any two or more numbers
        """
        # Pattern 1: named coordinates (x=..., y=..., z=...)
        named = _extract_named_coords(text)
        if named is not None:
            return named

        # Pattern 2: bracket list
        bracket = re.search(r"\[([^\]]+)\]", text)
        if bracket:
            nums = _parse_number_list(bracket.group(1))
            if nums and len(nums) >= 2:
                return nums[:3]

        # Pattern 3: parenthesis list
        paren = re.search(r"\(([^)]+)\)", text)
        if paren:
            nums = _parse_number_list(paren.group(1))
            if nums and len(nums) >= 2:
                return nums[:3]

        # Pattern 4: numbers after "target" or "position"
        kw_match = re.search(
            r"(?:target|position|point)\s+([\d\s,.\-+eE]+)",
            text,
            re.IGNORECASE,
        )
        if kw_match:
            nums = _parse_number_list(kw_match.group(1))
            if nums and len(nums) >= 2:
                return nums[:3]

        # Pattern 5: fallback — any two or more numbers
        all_nums = re.findall(
            r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text,
        )
        if len(all_nums) >= 2:
            try:
                return [float(n) for n in all_nums[:3]]
            except ValueError:
                pass

        return None

    @staticmethod
    def _extract_solver_name(text: str) -> Optional[str]:
        """Extract an explicit solver name from *text*.

        Looks for patterns like ``solver=damped_least_squares`` or
        ``using jacobian``.
        """
        # Pattern: solver=<name> or solver:<name>
        solver_match = re.search(
            r"solver\s*[=:]\s*([a-z_]+)", text,
        )
        if solver_match:
            return solver_match.group(1)

        # Pattern: "using <name>"
        using_match = re.search(
            r"using\s+([a-z_]+)", text,
        )
        if using_match:
            candidate = using_match.group(1)
            # Avoid matching generic words
            if candidate not in {"the", "a", "an", "this", "that"}:
                return candidate

        return None


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _extract_named_coords(text: str) -> Optional[List[float]]:
    """Extract named ``x=``, ``y=``, ``z=`` coordinates from *text*.

    Returns:
        ``[x, y]`` or ``[x, y, z]`` if at least ``x`` and ``y`` are
        found, otherwise ``None``.
    """
    x_match = re.search(
        r"x\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text,
        re.IGNORECASE,
    )
    y_match = re.search(
        r"y\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text,
        re.IGNORECASE,
    )
    if x_match and y_match:
        coords = [float(x_match.group(1)), float(y_match.group(1))]
        z_match = re.search(
            r"z\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text,
            re.IGNORECASE,
        )
        if z_match:
            coords.append(float(z_match.group(1)))
        return coords
    return None


def _parse_number_list(raw: str) -> Optional[List[float]]:
    """Parse a string of comma / space separated numbers.

    Returns:
        List of floats, or ``None`` if parsing fails.
    """
    parts = re.split(r"[,\s]+", raw.strip())
    values: List[float] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values if values else None
