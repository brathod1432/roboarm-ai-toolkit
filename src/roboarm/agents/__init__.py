"""AI agents layer for natural-language robot arm interaction.

This package provides keyword-based (no external LLM) agents that parse
user queries and call robotics tools for forward/inverse kinematics,
Jacobian analysis, and solver comparison.

Quick start::

    from roboarm.robots.two_link_planar import create_two_link_planar
    from roboarm.agents import RoboticsCoordinator

    coordinator = RoboticsCoordinator(create_two_link_planar())
    print(coordinator.process("Describe the robot"))
"""

from __future__ import annotations

from roboarm.agents.base_agent import AgentMemory, AgentMessage, BaseAgent
from roboarm.agents.coordinator import RoboticsCoordinator
from roboarm.agents.fk_agent import FKAgent
from roboarm.agents.ik_agent import IKAgent
from roboarm.agents.robotics_tools import build_robotics_tools
from roboarm.agents.tools import ToolDefinition, ToolRegistry

__all__ = [
    "AgentMemory",
    "AgentMessage",
    "BaseAgent",
    "FKAgent",
    "IKAgent",
    "RoboticsCoordinator",
    "ToolDefinition",
    "ToolRegistry",
    "build_robotics_tools",
]
