# Using the AI Agent Layer

The AI agent layer lets you interact with the kinematics engine using natural-language queries instead of writing code. A coordinator routes your questions to specialised agents and tools, returning human-readable results. This tutorial covers creating a coordinator, issuing queries, understanding the tool system, and building your own tools.

## What the Agent Layer Does

The agent layer sits on top of the core kinematics library and provides:

- **Natural-language routing** -- type a question like *"compute FK for angles [0.5, -0.3]"* and the coordinator figures out which tool to call.
- **Tool abstraction** -- each capability (FK, IK, Jacobian, robot description, solver comparison) is wrapped in a `ToolDefinition` with a schema, so tools can be discovered and invoked uniformly.
- **Specialist agents** -- `FKAgent` handles forward-kinematics queries and `IKAgent` handles inverse-kinematics and solver-comparison queries, each parsing parameters from the user's text.

All agent code lives in [`src/roboarm/agents/`](../../src/roboarm/agents/).

## Creating a RoboticsCoordinator

The `RoboticsCoordinator` is the main entry point. It takes a `RobotArm` and sets up the tool registry and specialist agents automatically:

```python
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.agents.coordinator import RoboticsCoordinator

robot = create_two_link_planar()
coordinator = RoboticsCoordinator(robot)
```

## Example Queries

Once the coordinator is created, pass natural-language strings to `coordinator.process()`:

### Describe the robot

```python
print(coordinator.process("Describe this robot"))
```

Output:
```
Robot: 2-Link Planar (L1=1.0, L2=1.0)
  DOF: 2
  Joints: 2
  Joint names: ['J1', 'J2']
  J1: [-3.1416, 3.1416] rad
  J2: [-3.1416, 3.1416] rad
```

### Forward kinematics

```python
print(coordinator.process("Compute FK for angles [0.5, -0.3]"))
```

Output:
```
Forward Kinematics Result:
  Input angles (rad): [0.5, -0.3]
  End-effector position:
    x = 1.4553
    y = 0.2739
    z = 0.0000
```

### Inverse kinematics

```python
print(coordinator.process("Solve IK for x=1.2, y=0.8"))
```

The coordinator detects IK intent from keywords like *solve*, *inverse*, *target*, or the `x=` / `y=` pattern, and delegates to the `IKAgent`.

### Compare solvers

```python
print(coordinator.process("Compare all solvers for x=0.8, y=0.6"))
```

This runs every registered IK solver on the same target and returns a comparison table with success status, residual error, computation time, and iteration count.

### Jacobian analysis

```python
print(coordinator.process("Compute jacobian for angles [0.5, -0.3]"))
```

## How Query Routing Works

The coordinator uses keyword-based intent detection, evaluated in priority order to handle overlapping keywords correctly:

| Priority | Keywords | Route |
|----------|----------|-------|
| 1 | `describe`, `info`, `about`, `details` | `describe_robot` tool |
| 2 | `compare`, `benchmark` | `IKAgent` (comparison mode) |
| 3 | `jacobian`, `manipulability`, `singular` | `compute_jacobian` tool |
| 4 | `fk`, `forward`, `angles` | `FKAgent` |
| 5 | `ik`, `inverse`, `solve`, `reach`, `target` | `IKAgent` |
| 6 | (no match) | Help message |

Jacobian is checked *before* FK because a query like *"compute jacobian for angles [0.5, -0.3]"* contains both "jacobian" and "angles" keywords. The priority ordering ensures it routes correctly.

The routing logic is implemented in `RoboticsCoordinator.process()` in [`src/roboarm/agents/coordinator.py`](../../src/roboarm/agents/coordinator.py).

## The Tool System

Tools are the atomic units of functionality exposed to agents. The system has two key classes in [`src/roboarm/agents/tools.py`](../../src/roboarm/agents/tools.py):

### ToolDefinition

Each tool is described by a `ToolDefinition` dataclass:

```python
from roboarm.agents.tools import ToolDefinition

tool = ToolDefinition(
    name="my_tool",
    description="Does something useful",
    parameters={
        "value": {"type": "number", "description": "Input value"},
    },
    function=lambda value: f"Result: {value * 2}",
)
```

The `parameters` dictionary follows a JSON-schema-like format, making tools compatible with LLM function-calling APIs.

### ToolRegistry

The `ToolRegistry` stores and manages tools:

```python
from roboarm.agents.tools import ToolRegistry

registry = ToolRegistry()
registry.register(tool)

# Execute by name
result = registry.execute("my_tool", value=42)

# List all tools
print(registry.list_tools())

# Get OpenAI-compatible schemas
schemas = registry.get_schemas()
```

## Built-In Tools

The function `build_robotics_tools(robot)` in [`src/roboarm/agents/robotics_tools.py`](../../src/roboarm/agents/robotics_tools.py) creates a registry with five pre-built tools:

| Tool name | Description |
|-----------|-------------|
| `describe_robot` | Robot name, DOF, joints, and limits |
| `compute_fk` | Forward kinematics for given angles |
| `solve_ik` | Inverse kinematics with a chosen solver |
| `compute_jacobian` | Jacobian matrix and manipulability |
| `compare_solvers` | Run all IK solvers on a target |

## Building Custom Tools

You can extend the coordinator with your own tools:

```python
from roboarm.agents.tools import ToolDefinition

def workspace_volume(radius: float) -> str:
    import math
    vol = (4/3) * math.pi * radius**3
    return f"Approximate workspace volume: {vol:.4f} m^3"

custom_tool = ToolDefinition(
    name="workspace_volume",
    description="Estimate workspace volume from reach radius",
    parameters={
        "radius": {"type": "number", "description": "Max reach in metres"},
    },
    function=workspace_volume,
)

# Add to the coordinator's registry
coordinator.tools.register(custom_tool)

# Now it can be called directly
print(coordinator.tools.execute("workspace_volume", radius=2.0))
```

## Architecture Overview

```
User query (text)
      |
      v
RoboticsCoordinator.process()      # keyword-based routing
      |
      +--> describe_robot tool      # direct tool execution
      +--> compute_jacobian tool    # direct tool execution
      +--> FKAgent.process()        # parses angles, calls compute_fk
      +--> IKAgent.process()        # parses target, calls solve_ik or compare_solvers
      |
      v
Human-readable response (text)
```

The `FKAgent` and `IKAgent` inherit from `AgentBase` defined in [`src/roboarm/agents/base_agent.py`](../../src/roboarm/agents/base_agent.py). Each agent receives the shared `ToolRegistry` and extracts parameters from the user query before calling the appropriate tool.

---

## See Also

- [Getting Started](01_getting_started.md) -- installation and basic API usage
- [Inverse Kinematics Theory](../theory/inverse_kinematics.md) -- the solvers that the IK agent invokes
- [`src/roboarm/agents/coordinator.py`](../../src/roboarm/agents/coordinator.py) -- `RoboticsCoordinator` routing logic
- [`src/roboarm/agents/tools.py`](../../src/roboarm/agents/tools.py) -- `ToolDefinition` and `ToolRegistry`
- [`src/roboarm/agents/robotics_tools.py`](../../src/roboarm/agents/robotics_tools.py) -- built-in tool factory
- [`src/roboarm/agents/fk_agent.py`](../../src/roboarm/agents/fk_agent.py) -- FK specialist agent
- [`src/roboarm/agents/ik_agent.py`](../../src/roboarm/agents/ik_agent.py) -- IK specialist agent
