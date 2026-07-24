"""Multi-agent coordinator for robotics queries.

:class:`RoboticsCoordinator` is the main entry-point for the AI agents
layer.  It inspects incoming user queries, routes them to the
appropriate specialist agent or tool, and returns a human-readable
response.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from roboarm.agents.base_agent import AgentMessage
from roboarm.agents.fk_agent import FKAgent
from roboarm.agents.ik_agent import IKAgent
from roboarm.agents.robotics_tools import build_robotics_tools
from roboarm.agents.tools import ToolRegistry
from roboarm.core.robot import RobotArm

logger = logging.getLogger(__name__)

# Keyword sets for routing (order matters -- see process())
_DESCRIBE_KEYWORDS = {"describe", "info", "about", "details"}
_COMPARE_KEYWORDS = {"compare", "benchmark"}
_JACOBIAN_KEYWORDS = {"jacobian", "manipulability", "singular", "singularity"}
_FK_KEYWORDS = {"fk", "forward", "angles"}
_IK_KEYWORDS = {"ik", "inverse", "solve", "reach", "target"}

# Help text returned when no intent is detected
_HELP_TEXT = (
    "I can help you with the following robotics tasks:\n"
    "\n"
    "  1. Describe the robot  -- 'describe', 'robot info'\n"
    "  2. Forward kinematics  -- 'compute FK for angles [0.5, -0.3]'\n"
    "  3. Inverse kinematics  -- 'solve IK for x=1.0, y=0.5'\n"
    "  4. Compare IK solvers  -- 'compare all solvers for x=0.8, y=0.6'\n"
    "  5. Jacobian analysis   -- 'compute jacobian for angles [0.5, -0.3]'\n"
    "\n"
    "Please rephrase your query using one of the patterns above."
)


class RoboticsCoordinator:
    """Multi-agent coordinator that routes queries to the right agent.

    The coordinator owns a shared :class:`ToolRegistry` and two
    specialist agents (:class:`FKAgent` and :class:`IKAgent`).  It
    classifies each incoming query using keyword matching and delegates
    to the most appropriate handler.

    Routing order is important: more specific intents (Jacobian,
    compare) are checked before broader ones (FK, IK) so that queries
    containing overlapping keywords (e.g. ``"jacobian for angles ..."``
    which has both "jacobian" *and* "angles") are routed correctly.

    Args:
        robot: The robot arm model all tools and agents operate on.

    Example::

        from roboarm.robots.two_link_planar import create_two_link_planar
        coordinator = RoboticsCoordinator(create_two_link_planar())
        print(coordinator.process("Describe the robot"))
    """

    def __init__(self, robot: RobotArm) -> None:
        self._robot = robot
        self._tools = build_robotics_tools(robot)
        self._fk_agent = FKAgent("FK Agent", self._tools)
        self._ik_agent = IKAgent("IK Agent", self._tools)
        logger.info(
            "RoboticsCoordinator initialised for %r with tools: %s",
            robot.name,
            self._tools.list_tools(),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, user_input: str) -> str:
        """Route *user_input* to the appropriate agent or tool.

        Routing rules (evaluated in priority order):

        1. **describe / info / about** -> ``describe_robot`` tool
        2. **compare / benchmark** -> :class:`IKAgent` (compare mode)
        3. **jacobian / manipulability / singular** -> ``compute_jacobian``
           (checked *before* FK because queries may contain "angles")
        4. **fk / forward / angles** -> :class:`FKAgent`
        5. **ik / inverse / solve / reach / target** -> :class:`IKAgent`
        6. **default** -> help message listing available commands

        Args:
            user_input: Raw text from the user.

        Returns:
            Human-readable response string.
        """
        logger.info("Coordinator received: %s", user_input)
        lower = user_input.lower()
        tokens = set(re.findall(r"[a-z]+", lower))

        # --- Route 1: Robot description --------------------------------
        if tokens & _DESCRIBE_KEYWORDS:
            return self._handle_describe()

        # --- Route 2: Solver comparison --------------------------------
        if tokens & _COMPARE_KEYWORDS:
            return self._ik_agent.process(user_input)

        # --- Route 3: Jacobian analysis (before FK!) -------------------
        if tokens & _JACOBIAN_KEYWORDS:
            return self._handle_jacobian(user_input)

        # --- Route 4: Forward kinematics -------------------------------
        if tokens & _FK_KEYWORDS:
            return self._fk_agent.process(user_input)

        # --- Route 5: Inverse kinematics ------------------------------
        if tokens & _IK_KEYWORDS or re.search(r"[xy]\s*=", lower):
            return self._ik_agent.process(user_input)

        # --- Default: help message -------------------------------------
        logger.info("No intent matched; returning help text")
        return _HELP_TEXT

    # ------------------------------------------------------------------
    # Direct tool handlers
    # ------------------------------------------------------------------

    def _handle_describe(self) -> str:
        """Execute the ``describe_robot`` tool directly."""
        try:
            result = self._tools.execute("describe_robot")
            return str(result)
        except Exception as exc:
            logger.exception("describe_robot tool failed")
            return f"Error describing robot: {exc}"

    def _handle_jacobian(self, user_input: str) -> str:
        """Extract angles and execute the ``compute_jacobian`` tool."""
        angles = self._extract_angles(user_input)
        if angles is None:
            return (
                "Please provide joint angles for the Jacobian "
                "computation, e.g. 'compute jacobian for angles "
                "[0.5, -0.3]'."
            )
        try:
            result = self._tools.execute(
                "compute_jacobian", angles=angles,
            )
            return str(result)
        except Exception as exc:
            logger.exception("compute_jacobian tool failed")
            return f"Error computing Jacobian: {exc}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_angles(text: str) -> Optional[List[float]]:
        """Extract a list of floats from *text*.

        Tries bracket notation, parenthesis notation, "angles" keyword,
        and a fallback of any two-or-more numbers in the string.
        """
        # Bracket notation
        bracket = re.search(r"\[([^\]]+)\]", text)
        if bracket:
            return _parse_number_list(bracket.group(1))

        # Parenthesis notation
        paren = re.search(r"\(([^)]+)\)", text)
        if paren:
            nums = _parse_number_list(paren.group(1))
            if nums:
                return nums

        # After the word "angles"
        angles_match = re.search(
            r"angles?\s+([\d\s,.\-+eE]+)", text, re.IGNORECASE,
        )
        if angles_match:
            nums = _parse_number_list(angles_match.group(1))
            if nums:
                return nums

        # Fallback: any two+ numbers
        all_nums = re.findall(
            r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text,
        )
        if len(all_nums) >= 2:
            try:
                return [float(n) for n in all_nums]
            except ValueError:
                pass
        return None

    @property
    def tools(self) -> ToolRegistry:
        """The shared tool registry."""
        return self._tools

    @property
    def fk_agent(self) -> FKAgent:
        """The forward-kinematics specialist agent."""
        return self._fk_agent

    @property
    def ik_agent(self) -> IKAgent:
        """The inverse-kinematics specialist agent."""
        return self._ik_agent


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _parse_number_list(raw: str) -> Optional[List[float]]:
    """Parse a string of comma / space separated numbers."""
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
