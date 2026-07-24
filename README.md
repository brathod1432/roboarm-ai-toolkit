# roboarm-ai-toolkit

> A modular, production-quality robot arm kinematics toolkit with AI-powered
> agents for forward kinematics, inverse kinematics, trajectory planning,
> and manipulability analysis.

**Author:** Brijesh Rathod
**Python:** 3.10+
**License:** MIT

---

## 1. Project Overview

A complete **robot manipulator toolkit** covering the full pipeline of
serial-link robot arm engineering:

1. **Core Mathematics** -- SE(3) transforms, SO(3) rotations, DH/MDH parameters
2. **Forward Kinematics** -- Joint angles -> End-effector pose
3. **Inverse Kinematics** -- Target pose -> Joint angles (multiple solvers)
4. **Jacobian Analysis** -- Velocity kinematics, manipulability, singularity detection
5. **Trajectory Planning** -- Joint-space and Cartesian-space interpolation
6. **AI Agents** -- Natural language interface to all kinematics operations
7. **Visualization** -- 2D/3D arm rendering, trajectory animation, workspace plots

---

## 2. Key Features

| Feature | Description |
|---------|-------------|
| **Multiple IK Solvers** | Analytical, Jacobian Pseudoinverse, Damped Least Squares, CCD, FABRIK |
| **AI Agent Layer** | FK Agent, IK Agent, Coordinator -- tool-calling architecture |
| **Pre-built Robots** | 2-link planar, 3-link planar, 6-DOF (MDH), SCARA, PUMA 560 |
| **Benchmarking** | Solver comparison, timing, convergence analysis |
| **Zero Heavy Dependencies** | Core runs on `numpy` + `matplotlib` only |
| **Production Patterns** | Registry, Strategy, Factory -- clean interfaces throughout |

---

## 3. Supported DH Conventions

| Convention | Description | Status |
|------------|-------------|--------|
| Classic DH | Standard Denavit-Hartenberg | Supported |
| Modified DH (MDH) | Craig convention (proximal) | Supported |

---

## 4. Quick Start

### Installation

```bash
# Clone
git clone https://github.com/brathod1432/roboarm-ai-toolkit.git
cd roboarm-ai-toolkit

# Install in development mode
pip install -e ".[dev]"
```

### Forward Kinematics

```python
from roboarm.robots import create_two_link_planar
from roboarm.visualization import ArmVisualizer
import numpy as np

robot = create_two_link_planar()
pose = robot.forward_kinematics([np.pi/4, -np.pi/6])
print(f"End-effector: ({pose.x:.4f}, {pose.y:.4f})")

viz = ArmVisualizer(robot)
viz.plot_2d([np.pi/4, -np.pi/6], show_workspace=True)
```

### Inverse Kinematics

```python
from roboarm.kinematics.solvers import DampedLeastSquaresIK

solver = DampedLeastSquaresIK(robot)
solution = solver.solve(target_pose)
print(f"Joint angles: {solution.primary.values}")
```

### AI Agent

```python
from roboarm.agents import RoboticsCoordinator

coordinator = RoboticsCoordinator(robot)
response = coordinator.process("Solve IK for x=1.0, y=0.5")
print(response)
```

---

## 5. Architecture

```
+-----------------------------------------------------+
|                   AI AGENT LAYER                     |
|  +----------+  +----------+  +-------------------+  |
|  | FK Agent |  | IK Agent |  | Trajectory Agent  |  |
|  +----+-----+  +----+-----+  +--------+----------+  |
|       +--------------+-----------------+             |
|                      v                               |
|              +--------------+                        |
|              |  Tool Layer  |  (Function Calling)    |
|              +------+-------+                        |
+---------------------+-------------------------------+
|                     v           KINEMATICS ENGINE    |
|  +----------+  +---------+  +--------------------+  |
|  |    FK    |  |   IK    |  |     Jacobian       |  |
|  |  Engine  |  | Solvers |  |    Computer        |  |
|  +----+-----+  +----+----+  +--------+-----------+  |
|       +--------------+----------------+              |
|                      v                               |
|              +--------------+                        |
|              |  Robot Model |  (DH / MDH)            |
|              +------+-------+                        |
+---------------------+-------------------------------+
|                     v             CORE MATH          |
|  +----------+  +----------+  +------------------+   |
|  |Transforms|  |Rotations |  |  Types / Config  |   |
|  |  SE(3)   |  |  SO(3)   |  |                  |   |
|  +----------+  +----------+  +------------------+   |
+-----------------------------------------------------+
```

---

## 6. Directory Structure

```
roboarm-ai-toolkit/
├── README.md
├── LICENSE
├── pyproject.toml
├── Makefile
├── ARCHITECTURE.md
├── ROADMAP.md
├── CHANGELOG.md
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   ├── theory/
│   │   ├── forward_kinematics.md
│   │   ├── inverse_kinematics.md
│   │   ├── jacobian.md
│   │   └── dh_parameters.md
│   ├── tutorials/
│   │   ├── 01_getting_started.md
│   │   ├── 02_custom_robot.md
│   │   └── 03_ai_agents.md
│   └── assets/
├── src/
│   └── roboarm/
│       ├── core/           # Math foundations
│       ├── kinematics/     # FK, IK, Jacobian
│       │   └── solvers/    # IK solver implementations
│       ├── robots/         # Pre-defined robot models
│       ├── agents/         # AI agent layer
│       ├── trajectory/     # Path planning
│       ├── workspace/      # Reachability analysis
│       ├── visualization/  # Plotting & animation
│       └── utils/          # Helpers
├── tests/
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
└── examples/
```

---

## 7. IK Solvers Implemented

| Solver | Method | Best For |
|--------|--------|----------|
| `analytical_2link` | Closed-form geometry | 2-DOF planar arms |
| `jacobian_pseudoinverse` | J+ iterative | General, non-singular configs |
| `damped_least_squares` | J^T(JJ^T + lambda^2 I)^-1 | Robust near singularities |
| `ccd` | Cyclic Coordinate Descent | High-DOF, animation |
| `fabrik` | Forward/Backward reaching | Fast, position-only |

---

## 8. Dependencies

```
numpy>=1.24        # Matrix operations
matplotlib>=3.7    # Visualization
scipy>=1.10        # Optimization (optional IK)
```

---

## 9. Running Tests

```bash
pytest tests/ -v --cov=roboarm
```

---

## 10. License

MIT License -- see [LICENSE](LICENSE)
