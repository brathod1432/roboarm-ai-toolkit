# roboarm-ai-toolkit

> A modular, production-quality robot arm kinematics toolkit with AI-powered
> agents for forward kinematics, inverse kinematics, trajectory planning,
> and manipulability analysis.

**Python:** 3.10+ | **License:** MIT | **Free for everyone** -- see [Usage Terms](USAGE_TERMS.md)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Layout](#project-layout)
- [IK Solvers](#ik-solvers)
- [Test Results](#test-results)
- [Documentation](#documentation)
- [Examples](#examples)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

A complete **robot manipulator toolkit** covering the full pipeline of
serial-link robot arm engineering:

| Layer | What It Does | Key Files |
|-------|-------------|-----------|
| **Core Math** | SE(3) transforms, SO(3) rotations, DH/MDH parameters | [`src/roboarm/core/`](src/roboarm/core/) |
| **Robot Models** | Pre-built 2-link, 3-link, 6-DOF MDH arms | [`src/roboarm/robots/`](src/roboarm/robots/) |
| **Forward Kinematics** | Joint angles -> end-effector pose | [`src/roboarm/core/robot.py`](src/roboarm/core/robot.py) |
| **Inverse Kinematics** | Target pose -> joint angles (5 solvers) | [`src/roboarm/kinematics/solvers/`](src/roboarm/kinematics/solvers/) |
| **Jacobian Analysis** | Velocity kinematics, manipulability, singularity detection | [`src/roboarm/kinematics/jacobian.py`](src/roboarm/kinematics/jacobian.py) |
| **Trajectory Planning** | Joint-space interpolation, LSPB profiles | [`src/roboarm/trajectory/`](src/roboarm/trajectory/) |
| **AI Agents** | Natural language interface to all operations | [`src/roboarm/agents/`](src/roboarm/agents/) |
| **Visualization** | 2D arm rendering, workspace plots | [`src/roboarm/visualization/`](src/roboarm/visualization/) |

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/roboarm-ai-toolkit.git
cd roboarm-ai-toolkit
pip install -e ".[dev]"
```

For a detailed walkthrough, see the [Getting Started Tutorial](docs/tutorials/01_getting_started.md).

### Forward Kinematics

```python
from roboarm.robots import create_two_link_planar
import numpy as np

robot = create_two_link_planar(link1=1.0, link2=1.0)
pose = robot.forward_kinematics([np.pi/4, -np.pi/6])
print(f"End-effector: x={pose.x:.4f}, y={pose.y:.4f}")
```

### Inverse Kinematics

```python
from roboarm.kinematics.solvers.registry import IKSolverRegistry
import roboarm.kinematics.solvers  # register all solvers

solver = IKSolverRegistry.create("damped_least_squares", robot)
solution = solver.solve(target_pose)
print(f"Solved: {solution.success}, error: {solution.residual_error:.8f}")
```

### AI Agent (Natural Language)

```python
from roboarm.agents import RoboticsCoordinator

coordinator = RoboticsCoordinator(robot)
print(coordinator.process("Solve IK for x=1.0, y=0.5"))
print(coordinator.process("Compare all IK solvers for x=0.8, y=0.6"))
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **5 IK Solvers** | Analytical, Jacobian Pseudoinverse, Damped Least Squares, CCD, FABRIK |
| **AI Agent Layer** | FK Agent, IK Agent, Coordinator -- tool-calling architecture |
| **3 Pre-built Robots** | 2-link planar, 3-link planar (redundant), 6-DOF MDH |
| **Both DH Conventions** | Standard DH and Modified DH (Craig) fully supported |
| **234 Tests** | Unit, accuracy, negative, security, stress, integration tests |
| **Sub-mm Accuracy** | IK roundtrip: max error <0.001, mean error <0.00001 |
| **Lightweight** | Core runs on `numpy` + `matplotlib` only |
| **Production Patterns** | Registry, Strategy, Factory -- clean, extensible interfaces |

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full details.

```
Layer 4:  AGENTS        ->  Natural language -> tool calls -> formatted output
Layer 3:  KINEMATICS    ->  FK engine, IK solvers (5), Jacobian computer
Layer 2:  ROBOT MODEL   ->  Joint chain, DH/MDH parameters, frame queries
Layer 1:  CORE MATH     ->  SE(3) transforms, SO(3) rotations, types
```

Each layer depends only on layers below it. No circular imports.

---

## Project Layout

```
roboarm-ai-toolkit/
|-- README.md                          # This file
|-- LICENSE                            # MIT License
|-- USAGE_TERMS.md                     # Free-use terms for everyone
|-- CONTRIBUTING.md                    # How to contribute
|-- ARCHITECTURE.md                    # Design and layer diagram
|-- ROADMAP.md                         # Development phases
|-- CHANGELOG.md                       # Version history
|-- pyproject.toml                     # Build config (setuptools, pytest, ruff)
|-- Makefile                           # Dev commands (install, test, lint)
|
|-- docs/
|   |-- theory/
|   |   |-- dh_parameters.md           # DH convention explained
|   |   |-- forward_kinematics.md      # FK theory and examples
|   |   |-- inverse_kinematics.md      # IK solvers explained
|   |   +-- jacobian.md                # Jacobian and manipulability
|   +-- tutorials/
|       |-- 01_getting_started.md       # First steps
|       |-- 02_custom_robot.md          # Build your own robot model
|       +-- 03_ai_agents.md            # Using the AI agent layer
|
|-- src/roboarm/
|   |-- core/                          # Layer 1: Math foundations
|   |   |-- types.py                   #   Dataclasses (DHParams, Pose, IKSolution)
|   |   |-- exceptions.py             #   Error hierarchy
|   |   |-- transform.py              #   DH + MDH 4x4 transforms
|   |   |-- rotations.py              #   Euler, axis-angle, quaternion
|   |   +-- robot.py                  #   RobotArm model with FK
|   |-- kinematics/                    # Layer 3: Kinematics engine
|   |   |-- forward.py                #   FK convenience wrapper
|   |   |-- jacobian.py               #   Geometric + numerical Jacobian
|   |   |-- inverse.py                #   IKSolverBase ABC
|   |   +-- solvers/                  #   IK solver implementations
|   |       |-- registry.py           #     Solver registry (factory)
|   |       |-- analytical.py         #     2-link closed-form
|   |       |-- jacobian_ik.py        #     Pseudoinverse iterative
|   |       |-- damped_least_squares.py #   DLS (lambda damping)
|   |       |-- ccd.py                #     Cyclic Coordinate Descent
|   |       +-- fabrik.py             #     FABRIK algorithm
|   |-- robots/                        # Layer 2: Robot definitions
|   |   |-- two_link_planar.py        #   2-DOF RR planar
|   |   |-- three_link_planar.py      #   3-DOF RRR redundant
|   |   +-- six_dof_mdh.py           #   6-DOF MDH with TCP offset
|   |-- agents/                        # Layer 4: AI agents
|   |   |-- tools.py                  #   ToolDefinition + ToolRegistry
|   |   |-- base_agent.py            #   AgentMessage, Memory, BaseAgent
|   |   |-- robotics_tools.py        #   FK/IK/Jacobian tool builders
|   |   |-- fk_agent.py              #   FK specialist
|   |   |-- ik_agent.py              #   IK specialist
|   |   +-- coordinator.py           #   Multi-agent router
|   |-- trajectory/                    # Path planning
|   |   |-- interpolation.py         #   Linear, cubic, quintic
|   |   +-- lspb.py                  #   Trapezoidal velocity profiles
|   |-- workspace/                     # Reachability
|   |   +-- analysis.py              #   Monte Carlo workspace sampling
|   |-- visualization/                 # Plotting
|   |   |-- arm_plot.py              #   2D arm configuration plots
|   |   +-- workspace_plot.py        #   Workspace scatter plots
|   +-- utils/                         # Helpers
|       |-- angle_utils.py            #   wrap_angle, deg2rad, rad2deg
|       |-- validation.py            #   Input validators
|       +-- config.py                #   Default configurations
|
|-- tests/
|   |-- conftest.py                    # Shared fixtures
|   |-- unit/                          # Fast, isolated tests
|   |   |-- test_types.py             #   13 tests: dataclass validation
|   |   |-- test_transform.py         #   15 tests: DH/MDH/inverse/chain
|   |   |-- test_rotations.py         #   14 tests: Euler/quat/axis-angle
|   |   |-- test_forward_kinematics.py #  16 tests: FK correctness
|   |   |-- test_inverse_kinematics.py #  10 tests: IK roundtrip
|   |   |-- test_accuracy.py           #  28 tests: precision sweeps
|   |   |-- test_negative.py           #  40 tests: error handling
|   |   +-- test_security.py           #  24 tests: injection/access control
|   +-- integration/                   # Cross-module tests
|       |-- test_roundtrip.py          #  34 tests: FK<->IK accuracy
|       |-- test_six_dof.py            #  18 tests: 6-DOF MDH robot
|       +-- test_stress.py             #  12 tests: performance benchmarks
|
+-- examples/
    |-- 01_two_link_fk.py              # Basic FK demo
    |-- 02_three_link_fk.py            # Redundant arm FK
    |-- 03_ik_solver_comparison.py     # All solvers compared
    |-- 04_jacobian_analysis.py        # Manipulability & singularity
    +-- 05_ai_agent_demo.py            # Natural language queries
```

---

## IK Solvers

Five inverse kinematics solvers behind a common interface
([Strategy pattern](ARCHITECTURE.md)). See [IK Theory](docs/theory/inverse_kinematics.md).

| Solver | Registry Name | Method | Best For |
|--------|--------------|--------|----------|
| Analytical | `analytical_2link` | Closed-form (law of cosines) | 2-DOF planar arms |
| Jacobian Pseudoinverse | `jacobian_pseudoinverse` | J+ iterative | General, non-singular |
| Damped Least Squares | `damped_least_squares` | J^T(JJ^T + lambda^2 I)^-1 | Near singularities |
| CCD | `ccd` | Cyclic Coordinate Descent | High-DOF, animation |
| FABRIK | `fabrik` | Forward/Backward reaching | Fast, position-only |

```python
# Use any solver by name via the registry
from roboarm.kinematics.solvers.registry import IKSolverRegistry
solver = IKSolverRegistry.create("damped_least_squares", robot)
```

---

## Test Results

**234 tests, 100% passing, 5.64 seconds**

| Category | Tests | What It Validates |
|----------|------:|-------------------|
| Core types & transforms | 51 | Dataclasses, DH/MDH matrices, rotations |
| Forward kinematics | 16 | Known-answer FK at specific configurations |
| Inverse kinematics | 11 | FK<->IK roundtrip, solver registry |
| **Accuracy & precision** | **28** | **200-config sweeps, sub-mm error verification** |
| **Negative / error handling** | **40** | **NaN, Inf, None, wrong types, invalid configs** |
| **Security** | **24** | **Injection attacks, unsupported ops, access control** |
| **Stress / performance** | **12** | **10K FK, 500 IK solves, 1K Jacobians** |
| **Integration roundtrip** | **33** | **10 arm geometries, workspace quadrant coverage** |
| **6-DOF MDH** | **19** | **Model properties, FK validity, Jacobian shape** |

### Accuracy Benchmarks

| Metric | Result |
|--------|--------|
| FK->IK->FK max error | < 0.001 (sub-millimeter) |
| FK->IK->FK mean error | < 0.00001 (10 micrometers) |
| Repeatability (20 solves, same target) | std < 0.000001 |
| 10 arm geometries (L1/L2 = 0.3-3.0) | All roundtrip errors < 0.001 |
| DLS convergence rate (500 targets) | > 90% |

```bash
# Run all tests
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ -v --cov=roboarm --cov-report=term-missing

# Run specific categories
pytest tests/unit/test_accuracy.py -v       # Precision tests
pytest tests/unit/test_security.py -v       # Security tests
pytest tests/integration/test_stress.py -v  # Performance benchmarks
```

---

## Documentation

### Theory (How It Works)

| Document | Topic |
|----------|-------|
| [DH Parameters](docs/theory/dh_parameters.md) | Standard and Modified DH conventions explained |
| [Forward Kinematics](docs/theory/forward_kinematics.md) | FK chain multiplication with examples |
| [Inverse Kinematics](docs/theory/inverse_kinematics.md) | All 5 solver algorithms explained |
| [Jacobian Matrix](docs/theory/jacobian.md) | Velocity kinematics and manipulability |
| [Trajectory Planning](docs/theory/trajectory_planning.md) | Polynomial interpolation and LSPB profiles |
| [Workspace Analysis](docs/theory/workspace_analysis.md) | Monte Carlo reachability and bounding box |

### Tutorials (How to Use It)

| Tutorial | Topic |
|----------|-------|
| [Getting Started](docs/tutorials/01_getting_started.md) | Installation, first FK, first IK |
| [Custom Robot](docs/tutorials/02_custom_robot.md) | Build your own robot from DH parameters |
| [AI Agents](docs/tutorials/03_ai_agents.md) | Natural language kinematics queries |

### Design

| Document | Topic |
|----------|-------|
| [Architecture](ARCHITECTURE.md) | Layered design, patterns, module responsibilities |
| [Roadmap](ROADMAP.md) | Development phases and future plans |
| [Changelog](CHANGELOG.md) | Version history |
| [Contributing](CONTRIBUTING.md) | How to contribute |

---

## Examples

Run any example directly:

```bash
python examples/01_two_link_fk.py           # Basic FK at several configs
python examples/02_three_link_fk.py          # Redundant arm exploration
python examples/03_ik_solver_comparison.py   # All 5 solvers compared
python examples/04_jacobian_analysis.py      # Manipulability and singularity
python examples/05_ai_agent_demo.py          # Natural language demo
```

---

## Dependencies

### Pinned Requirements (Recommended)

| File | Purpose |
|------|---------|
| [`requirements_python312.txt`](requirements_python312.txt) | Core runtime only (numpy + matplotlib, pinned versions) |
| [`requirements_dev_python312.txt`](requirements_dev_python312.txt) | Core + dev tools (pytest, ruff, mypy, pre-commit) |

```bash
# Install core dependencies only
pip install -r requirements_python312.txt

# Install everything (core + dev tools)
pip install -r requirements_dev_python312.txt

# Or use pyproject.toml (flexible version ranges)
pip install -e ".[dev]"
```

### Package Summary

| Package | Pinned Version | Min Version | Purpose |
|---------|---------------|-------------|---------|
| numpy | 2.5.1 | >= 1.24 | Matrix operations, linear algebra |
| matplotlib | 3.11.1 | >= 3.7 | Visualization, arm plots |
| pytest | 9.1.1 | >= 7.0 | Test framework (dev) |
| pytest-cov | 7.1.0 | >= 4.0 | Coverage reporting (dev) |
| ruff | 0.16.0 | >= 0.1 | Linter and formatter (dev) |
| mypy | 2.3.0 | >= 1.0 | Static type checker (dev) |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License -- see [LICENSE](LICENSE).

This project is a **free contribution** to the robotics community.
Anyone can use it for any good cause. See [USAGE_TERMS.md](USAGE_TERMS.md).
