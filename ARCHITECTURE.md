# Architecture -- roboarm-ai-toolkit

---

## Design Philosophy

1. **Layered Architecture** -- Core math -> Kinematics -> Agents (each layer
   depends only on layers below it)
2. **Strategy Pattern for IK** -- All solvers implement `IKSolverBase`;
   swap freely without changing calling code
3. **Registry Pattern** -- Solvers self-register; discover and instantiate by name
4. **Tool-Calling for Agents** -- Kinematics operations wrapped as callable tools
   compatible with OpenAI function-calling schema
5. **Zero ROS Dependency** -- Entire toolkit runs standalone with `pip install`

---

## Layer Diagram

```
Layer 4:  AGENTS        ->  Natural language -> tool calls -> formatted responses
Layer 3:  KINEMATICS    ->  FK, IK solvers, Jacobian, trajectory
Layer 2:  ROBOT MODEL   ->  DH/MDH chain, joint limits, robot definitions
Layer 1:  CORE MATH     ->  SE(3) transforms, SO(3) rotations, types
```

---

## Module Responsibilities

| Module | Responsibility | Depends On |
|--------|---------------|------------|
| `core/types.py` | Dataclasses: JointConfig, Pose, DHParams, IKSolution | -- |
| `core/transform.py` | 4x4 homogeneous transforms (DH & MDH) | `types` |
| `core/rotations.py` | Euler, axis-angle, quaternion conversions | -- |
| `core/robot.py` | Robot model: joint chain, FK, frame queries | `transform`, `types` |
| `kinematics/jacobian.py` | Geometric & numerical Jacobian | `robot` |
| `kinematics/solvers/*` | IK solver implementations | `robot`, `jacobian` |
| `agents/tools.py` | Tool definitions for function-calling | `kinematics` |
| `agents/fk_agent.py` | FK-specialized agent | `tools` |
| `agents/ik_agent.py` | IK-specialized agent | `tools` |
| `agents/coordinator.py` | Multi-agent router | `fk_agent`, `ik_agent` |
| `visualization/` | Matplotlib plotting & animation | `robot` |

---

## Key Design Decisions

### Why MDH Support?

The GroupProject reference robot uses Modified DH (Craig convention).
Supporting both Classic DH and MDH ensures compatibility with textbook
examples AND the existing 6-DOF model.

### Why Multiple IK Solvers?

Different solvers excel in different scenarios. The registry pattern
lets users (and AI agents) select the best solver for the task at hand.

### Why AI Agents?

Demonstrates modern GenAI + robotics integration -- a differentiator
for senior roles. The tool-calling architecture is the same pattern
used by OpenAI, Anthropic, and LangChain.
