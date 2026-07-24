# roboarm-ai-toolkit — Development Agent Prompt

---

## ROLE

You are a senior Python robotics engineer tasked with implementing the
**roboarm-ai-toolkit** project — a modular robot arm kinematics library
with AI-powered agents. You write production-quality, well-tested,
well-documented Python code.

---

## PROJECT LOCATION

```
Project Root: <local_project_path>/roboarm-ai-toolkit/
```

This is a **local-only** repository. No Git push, no remote hosting,
no CI/CD execution. All work stays on disk.

---

## PYTHON VERSION

- **Primary:** Python 3.12.3
- **Minimum supported:** Python 3.10
- **Compatibility:** Code must run on Python 3.10, 3.11, 3.12, and 3.13+
- Use `from __future__ import annotations` in every module
- No version-specific syntax (no 3.12-only features like `type` statements)
- Type hints must use `typing` module forms compatible with 3.10
  (e.g., `Optional[X]`, `List[X]`, not `X | None`, `list[X]` in annotations
  unless guarded by `from __future__ import annotations`)

---

## CONFIDENTIALITY RULES (MANDATORY)

**Before writing ANY file, apply these rules:**

1. **No real names** — Do not include author names, student IDs, team
   member names, email addresses, or usernames anywhere in source code,
   docstrings, comments, configs, or documentation
2. **No institution names** — Do not reference any university, company,
   course code, or lab identifiers
3. **No file paths** — Do not hardcode or reference any user-specific
   file paths (e.g., `C:\Users\...`)
4. **No GitHub usernames** — Use placeholder `yourusername` if a URL
   is absolutely needed
5. **No API keys or credentials** — Never embed or reference any
6. **Author field** — Use `"Author: Contributor"` or leave blank
7. **License** — MIT License with `Copyright (c) 2025 Contributors`
8. **After generating ALL files** — Perform a full audit scanning every
   file for leaked personal information, paths, names, IDs, or
   institution references. Report findings.

---

## PROJECT OVERVIEW

### What This Project Is

A **modular, production-quality robot arm kinematics toolkit** that:

1. Implements core robotics mathematics from scratch (SE(3) transforms,
   SO(3) rotations, DH and Modified DH parameters)
2. Provides forward kinematics for arbitrary serial-link manipulators
3. Implements multiple inverse kinematics solvers behind a common
   interface (Strategy + Registry pattern)
4. Computes geometric and numerical Jacobians for velocity kinematics,
   manipulability analysis, and singularity detection
5. Wraps all kinematics operations as **callable tools** compatible
   with LLM function-calling schemas
6. Provides **AI agents** (FK Agent, IK Agent, Coordinator) that accept
   natural language queries, invoke the appropriate tools, and return
   formatted, explainable results
7. Includes pre-defined robot models (2-link planar, 3-link planar,
   6-DOF MDH) as ready-to-use test subjects
8. Produces 2D visualizations of arm configurations, trajectories,
   and workspace boundaries

### What This Project Demonstrates

- Robotics domain expertise (kinematics, Jacobian, manipulability)
- Software architecture (layered design, design patterns, clean APIs)
- AI/agent integration (tool-calling, natural language interface)
- Production engineering (testing, type hints, documentation, packaging)

---

## ARCHITECTURE (Layered)

```
Layer 4:  AGENTS        ->  NL input -> intent -> tool calls -> formatted output
Layer 3:  KINEMATICS    ->  FK engine, IK solvers, Jacobian computer
Layer 2:  ROBOT MODEL   ->  Joint chain, DH/MDH parameters, frame queries
Layer 1:  CORE MATH     ->  Transforms, rotations, types, exceptions
```

Each layer depends ONLY on layers below it. No circular imports.

---

## DIRECTORY STRUCTURE

Implement every `.py` file listed below. Do not skip any.

```
roboarm-ai-toolkit/
├── README.md
├── LICENSE
├── ARCHITECTURE.md
├── ROADMAP.md
├── CHANGELOG.md
├── pyproject.toml
├── Makefile
├── .gitignore
│
├── src/roboarm/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── exceptions.py
│   │   ├── transform.py
│   │   ├── rotations.py
│   │   └── robot.py
│   │
│   ├── kinematics/
│   │   ├── __init__.py
│   │   ├── forward.py
│   │   ├── inverse.py
│   │   ├── jacobian.py
│   │   └── solvers/
│   │       ├── __init__.py
│   │       ├── registry.py
│   │       ├── analytical.py
│   │       ├── jacobian_ik.py
│   │       ├── damped_least_squares.py
│   │       ├── ccd.py
│   │       └── fabrik.py
│   │
│   ├── robots/
│   │   ├── __init__.py
│   │   ├── two_link_planar.py
│   │   ├── three_link_planar.py
│   │   └── six_dof_mdh.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── tools.py
│   │   ├── robotics_tools.py
│   │   ├── fk_agent.py
│   │   ├── ik_agent.py
│   │   └── coordinator.py
│   │
│   ├── trajectory/
│   │   ├── __init__.py
│   │   ├── interpolation.py
│   │   └── lspb.py
│   │
│   ├── workspace/
│   │   ├── __init__.py
│   │   └── analysis.py
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── arm_plot.py
│   │   └── workspace_plot.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── angle_utils.py
│       ├── validation.py
│       └── config.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_types.py
│   │   ├── test_transform.py
│   │   ├── test_rotations.py
│   │   ├── test_forward_kinematics.py
│   │   └── test_inverse_kinematics.py
│   ├── integration/
│   │   └── __init__.py
│   └── benchmarks/
│       └── __init__.py
│
└── examples/
    ├── 01_two_link_fk.py
    ├── 02_three_link_fk.py
    ├── 03_ik_solver_comparison.py
    ├── 04_jacobian_analysis.py
    └── 05_ai_agent_demo.py
```

---

## IMPLEMENTATION INSTRUCTIONS

### Order of Implementation

Build bottom-up. Each step must pass its tests before moving on.

```
STEP 1:  core/types.py + core/exceptions.py
STEP 2:  core/transform.py (DH + MDH)
STEP 3:  core/rotations.py
STEP 4:  utils/angle_utils.py + utils/validation.py
STEP 5:  core/robot.py (RobotArm with FK)
STEP 6:  robots/two_link_planar.py + robots/three_link_planar.py
STEP 7:  tests/conftest.py + tests/unit/test_types.py
         + test_transform.py + test_rotations.py + test_forward_kinematics.py
STEP 8:  kinematics/jacobian.py
STEP 9:  kinematics/inverse.py + solvers/registry.py
STEP 10: solvers/analytical.py (2-link closed-form)
STEP 11: solvers/jacobian_ik.py
STEP 12: solvers/damped_least_squares.py
STEP 13: solvers/ccd.py + solvers/fabrik.py
STEP 14: tests/unit/test_inverse_kinematics.py
STEP 15: robots/six_dof_mdh.py (using MDH convention from reference)
STEP 16: visualization/arm_plot.py + workspace_plot.py
STEP 17: trajectory/interpolation.py + trajectory/lspb.py
STEP 18: workspace/analysis.py
STEP 19: agents/tools.py + agents/base_agent.py
STEP 20: agents/robotics_tools.py
STEP 21: agents/fk_agent.py + agents/ik_agent.py
STEP 22: agents/coordinator.py
STEP 23: examples/ (all 5 scripts)
STEP 24: __init__.py files (package exports)
STEP 25: FINAL AUDIT — scan every file for leaked info
```

---

### Coding Standards (Apply to EVERY file)

```python
"""Module docstring — one sentence summary.

Detailed description if needed.
"""

from __future__ import annotations

# Type hints: use Optional, List, Dict, Tuple from typing
# NOT: X | None, list[X] — for 3.10 compat with __future__

# Every class: docstring with purpose + example usage
# Every public method: docstring with Args/Returns/Raises
# Every module: module-level docstring

# Logging: use standard logging module, never print()
# Constants: UPPER_SNAKE_CASE
# Private methods: single underscore prefix
```

---

## REFERENCE: 6-DOF MDH ROBOT

When implementing `robots/six_dof_mdh.py`, use these parameters:

### MDH Parameter Table

| Link | alpha(i-1) [deg] | a(i-1) [cm] | theta(i) [deg] | d(i) [cm] |
|------|------------------:|------------:|---------------:|----------:|
| 0->1 | 0 | 0 | theta1 | 0 |
| 1->2 | 90 | 0 | theta2 | 0 |
| 2->3 | 0 | 15 | theta3 | 0 |
| 3->4 | 90 | 7.2 | theta4 | 0 |
| 4->5 | 90 | 0 | theta5 | 13.2 |
| 5->6 | 90 | 0 | theta6 | 3 |
| 6->TCP| 0 | 0 | 0 | 7.5 |

### Joint Limits

| Joint | Min [deg] | Max [deg] | Vmax [deg/s] | Amax [deg/s^2] |
|-------|----------:|----------:|-------------:|--------------:|
| J1 | -150 | 150 | 90 | 180 |
| J2 | 5 | 175 | 90 | 180 |
| J3 | -90 | 90 | 90 | 180 |
| J4 | -90 | 90 | 90 | 180 |
| J5 | 90 | 270 | 90 | 180 |
| J6 | -90 | 90 | 90 | 180 |

**Home pose:** `[0, 90, 0, 0, 180, 0]` degrees

**FK chain:** `T_total = T_01 x T_12 x T_23 x T_34 x T_45 x T_56 x T_6TCP`

### IK Reference Algorithm

- Position-only IK (3-DOF target -> 6 joints)
- Numerical Jacobian (3x6) via finite differences
- Update rule: `delta_theta = J^T (J J^T + lambda^2 I)^-1 x error`
- Convergence: `||error|| < 1e-4 cm`, max 300 iterations

---

## EXPECTED BEHAVIOR (End-to-End)

### Test 1: Forward Kinematics Correctness

```python
from roboarm.robots import create_two_link_planar
import numpy as np

robot = create_two_link_planar(link1=1.0, link2=1.0)

# Both joints at zero -> end-effector at (2.0, 0.0)
pose = robot.forward_kinematics([0.0, 0.0])
assert abs(pose.x - 2.0) < 1e-10
assert abs(pose.y - 0.0) < 1e-10

# Joint1=90deg, Joint2=0deg -> end-effector at (0.0, 2.0)
pose = robot.forward_kinematics([np.pi/2, 0.0])
assert abs(pose.x - 0.0) < 1e-10
assert abs(pose.y - 2.0) < 1e-10
```

### Test 2: IK <-> FK Roundtrip

```python
from roboarm.kinematics.solvers import DampedLeastSquaresIK

robot = create_two_link_planar(link1=1.0, link2=0.8)
solver = DampedLeastSquaresIK(robot)

# Pick a reachable target
target_q = [0.5, -0.3]
target_pose = robot.forward_kinematics(target_q)

# Solve IK
solution = solver.solve(target_pose)
assert solution.success is True

# Verify: FK of IK result matches original target
recovered = robot.forward_kinematics(solution.primary.values)
assert abs(recovered.x - target_pose.x) < 1e-3
assert abs(recovered.y - target_pose.y) < 1e-3
```

### Test 3: Solver Comparison

```python
from roboarm.kinematics.solvers.registry import IKSolverRegistry

# All registered solvers can solve the same problem
for name in IKSolverRegistry.available():
    solver = IKSolverRegistry.create(name, robot)
    result = solver.solve(target_pose)
    print(f"{name}: success={result.success}, "
          f"time={result.computation_time_ms:.2f}ms, "
          f"error={result.residual_error:.6f}")
```

### Test 4: Jacobian Correctness

```python
from roboarm.kinematics.jacobian import JacobianComputer

jac = JacobianComputer(robot)
q = [0.5, -0.3]

# Geometric vs numerical Jacobian must agree
J_geo = jac.compute(q)
J_num = jac.compute_numerical(q)
assert np.allclose(J_geo, J_num, atol=1e-4)
```

### Test 5: AI Agent End-to-End

```python
from roboarm.robots import create_two_link_planar
from roboarm.agents import RoboticsCoordinator

robot = create_two_link_planar()
coordinator = RoboticsCoordinator(robot)

# Agent should understand and respond to natural language
response = coordinator.process("Describe the robot")
assert "2-Link" in response or "DOF" in response

response = coordinator.process("Compute FK for angles [0.5, -0.3]")
assert "position" in response.lower() or "x" in response.lower()

response = coordinator.process("Solve IK for x=1.0, y=0.5")
assert "success" in response.lower() or "solution" in response.lower()

response = coordinator.process("Compare all IK solvers for x=0.8, y=0.6")
assert "solver" in response.lower() or "time" in response.lower()
```

### Test 6: 6-DOF MDH Robot

```python
from roboarm.robots import create_six_dof_mdh
import numpy as np

robot = create_six_dof_mdh()
assert robot.n_joints == 7  # 6 revolute + 1 fixed TCP
assert robot.n_dof == 6

# FK at home pose should return a valid position
home_deg = [0, 90, 0, 0, 180, 0]
home_rad = [np.radians(d) for d in home_deg]
pose = robot.forward_kinematics(home_rad)
assert pose.position is not None
assert len(pose.position) == 3
```

### Test 7: Visualization (No Crash)

```python
from roboarm.visualization.arm_plot import ArmVisualizer
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing

robot = create_two_link_planar()
viz = ArmVisualizer(robot)
ax = viz.plot_2d([0.5, -0.3], show_workspace=True)
assert ax is not None
```

---

## TESTING INSTRUCTIONS

After implementing all files, run:

```bash
cd <project_root>/roboarm-ai-toolkit

# Activate virtual environment
.venv\Scripts\activate        # Windows

# Install package
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ -v --cov=roboarm --cov-report=term-missing

# Run examples (should execute without errors)
python examples/01_two_link_fk.py
python examples/02_three_link_fk.py
python examples/03_ik_solver_comparison.py
python examples/04_jacobian_analysis.py
python examples/05_ai_agent_demo.py
```

**All tests must pass. All examples must run without errors.**

---

## FINAL AUDIT CHECKLIST

After all files are created, review EVERY file and confirm:

- [ ] No real person names in any file
- [ ] No student/employee IDs
- [ ] No university or company names
- [ ] No email addresses
- [ ] No hardcoded file paths (especially `C:\Users\...`)
- [ ] No GitHub usernames (except placeholder `yourusername`)
- [ ] No API keys, tokens, or credentials
- [ ] `LICENSE` uses `Contributors` not a real name
- [ ] `pyproject.toml` author field is generic
- [ ] No references to course codes, lab numbers, or
      academic assignments in source code
- [ ] README does not reference any external private project
- [ ] All comments and docstrings are professional and generic

**Report the audit results as a checklist with PASS/FAIL for each item.**

---

## DEPENDENCIES

Only these packages are allowed:

```
# Core (required)
numpy>=1.24
matplotlib>=3.7

# Optional
scipy>=1.10        # For optimization-based IK

# Dev only
pytest>=7.0
pytest-cov>=4.0
ruff>=0.1
mypy>=1.0
```

**Do NOT add** torch, tensorflow, openai, langchain, fastapi, flask,
or any other heavy dependency. The core toolkit must be lightweight.

---

## SUMMARY

1. Read and understand the full project structure above
2. Implement ALL files in the specified order
3. Ensure every module has `from __future__ import annotations`
4. Write unit tests that verify correctness with known answers
5. Create working examples that produce visible output
6. Run the full test suite — everything must pass
7. Perform the confidentiality audit — report results
8. Do NOT push to Git — this is local only
