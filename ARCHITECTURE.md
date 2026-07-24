# Architecture -- roboarm-ai-toolkit

> See also: [README](README.md) | [Roadmap](ROADMAP.md) | [Contributing](CONTRIBUTING.md)

---

## Design Philosophy

1. **Layered Architecture** -- Core math -> Kinematics -> Agents (each layer
   depends only on layers below it)
2. **Strategy Pattern for IK** -- All solvers implement [`IKSolverBase`](src/roboarm/kinematics/inverse.py);
   swap freely without changing calling code
3. **Registry Pattern** -- Solvers self-register via [`IKSolverRegistry`](src/roboarm/kinematics/solvers/registry.py);
   discover and instantiate by name
4. **Tool-Calling for Agents** -- Kinematics operations wrapped as callable
   [`ToolDefinition`](src/roboarm/agents/tools.py) objects compatible with
   OpenAI function-calling schema
5. **Zero ROS Dependency** -- Entire toolkit runs standalone with `pip install`

---

## Layer Diagram

```
Layer 4:  AGENTS        ->  Natural language -> tool calls -> formatted responses
Layer 3:  KINEMATICS    ->  FK, IK solvers (5), Jacobian, trajectory
Layer 2:  ROBOT MODEL   ->  DH/MDH chain, joint limits, robot definitions
Layer 1:  CORE MATH     ->  SE(3) transforms, SO(3) rotations, types
```

---

## Module Responsibilities

| Module | Responsibility | Depends On | Docs |
|--------|---------------|------------|------|
| [`core/types.py`](src/roboarm/core/types.py) | Dataclasses: DHParams, Pose, IKSolution | -- | -- |
| [`core/exceptions.py`](src/roboarm/core/exceptions.py) | Error hierarchy | -- | -- |
| [`core/transform.py`](src/roboarm/core/transform.py) | 4x4 homogeneous transforms (DH & MDH) | `types` | [DH Theory](docs/theory/dh_parameters.md) |
| [`core/rotations.py`](src/roboarm/core/rotations.py) | Euler, axis-angle, quaternion conversions | -- | -- |
| [`core/robot.py`](src/roboarm/core/robot.py) | Robot model: joint chain, FK, frames | `transform`, `types` | [FK Theory](docs/theory/forward_kinematics.md) |
| [`kinematics/jacobian.py`](src/roboarm/kinematics/jacobian.py) | Geometric & numerical Jacobian | `robot` | [Jacobian Theory](docs/theory/jacobian.md) |
| [`kinematics/solvers/*`](src/roboarm/kinematics/solvers/) | 5 IK solver implementations | `robot`, `jacobian` | [IK Theory](docs/theory/inverse_kinematics.md) |
| [`robots/*`](src/roboarm/robots/) | Pre-defined robot models (2-link, 3-link, 6-DOF) | `core` | [Custom Robot Tutorial](docs/tutorials/02_custom_robot.md) |
| [`agents/*`](src/roboarm/agents/) | AI tool-calling agents | `kinematics` | [Agent Tutorial](docs/tutorials/03_ai_agents.md) |
| [`trajectory/*`](src/roboarm/trajectory/) | Interpolation, LSPB profiles | `core` | -- |
| [`workspace/*`](src/roboarm/workspace/) | Monte Carlo reachability | `robot` | -- |
| [`visualization/*`](src/roboarm/visualization/) | Matplotlib plotting | `robot` | -- |

---

## Key Design Decisions

### Why Both DH Conventions?

Many textbooks and tools use Standard DH, while the Craig convention (Modified
DH) is common in industrial practice and some academic curricula. Supporting
both ensures compatibility with a wide range of existing robot models and
educational materials. See [DH Parameters](docs/theory/dh_parameters.md).

### Why Multiple IK Solvers?

Different solvers excel in different scenarios:

| Solver | Strength | Weakness |
|--------|----------|----------|
| Analytical | Exact, fast, finds all solutions | Only works for specific geometries |
| Jacobian Pseudoinverse | General purpose | Fails at singularities |
| DLS | Robust near singularities | Slower convergence |
| CCD | Works for high-DOF | Can get stuck in local minima |
| FABRIK | Very fast, intuitive | Position-only |

The [Registry pattern](src/roboarm/kinematics/solvers/registry.py) lets users
(and AI agents) select the best solver at runtime.

### Why AI Agents?

The [tool-calling architecture](src/roboarm/agents/) demonstrates modern
GenAI + robotics integration. The same pattern (function schemas + intent
routing) is used by OpenAI, Anthropic, and LangChain. No external API
keys are required -- the agents use keyword-based intent parsing locally.

---

## Testing Strategy

| Category | Files | Purpose |
|----------|-------|---------|
| Unit tests | [`tests/unit/`](tests/unit/) | Fast, isolated correctness checks |
| Accuracy tests | [`test_accuracy.py`](tests/unit/test_accuracy.py) | Numerical precision across 200+ configurations |
| Negative tests | [`test_negative.py`](tests/unit/test_negative.py) | Error handling for every invalid input type |
| Security tests | [`test_security.py`](tests/unit/test_security.py) | Injection prevention, access control |
| Stress tests | [`test_stress.py`](tests/integration/test_stress.py) | 10K FK, 500 IK, performance benchmarks |
| Integration tests | [`tests/integration/`](tests/integration/) | FK<->IK roundtrip, 6-DOF, cross-module |

See [README - Test Results](README.md#test-results) for current numbers.
