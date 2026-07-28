# Project Reference — roboarm-ai-toolkit

Quick reference for build, test, lint, and verification commands.

## Environment

- Python: 3.10, 3.11, or 3.12
- Package layout: `src/` layout (`src/roboarm/`)
- Virtual environment: `.venv/` (create with `python -m venv .venv`)

## Install

```bash
# Editable install with all dev tools (recommended)
pip install -e ".[dev]"

# Or using the pinned requirements file (Python 3.12)
pip install -r requirements_dev_python312.txt

# Core runtime only (no dev tools)
pip install -r requirements_python312.txt
```

## Run Tests

```bash
# All 234 tests
pytest tests/ -v --tb=short

# With coverage report
pytest tests/ -v --cov=roboarm --cov-report=term-missing

# Unit tests only (fast, ~1.3 s)
pytest tests/unit/ -v --tb=short

# Integration tests only (~4.5 s)
pytest tests/integration/ -v --tb=short

# Specific test categories
pytest tests/unit/test_accuracy.py -v        # precision sweeps
pytest tests/unit/test_security.py -v        # injection / access control
pytest tests/integration/test_stress.py -v  # performance benchmarks
```

## Lint and Type Check

```bash
# Ruff linter (fast)
ruff check src/ tests/

# Auto-fix safe issues
ruff check --fix src/ tests/

# Auto-format
ruff format src/ tests/

# Mypy type checker
mypy src/roboarm/
```

## Run Examples

```bash
python examples/01_two_link_fk.py           # Basic FK at 6 configurations
python examples/02_three_link_fk.py          # Redundant arm + redundancy demo
python examples/03_ik_solver_comparison.py   # All 5 IK solvers compared
python examples/04_jacobian_analysis.py      # Manipulability & singularity
python examples/05_ai_agent_demo.py          # Natural language coordinator
```

All examples use `matplotlib.use("Agg")` so no display is required; PNG files
are saved to the working directory.

## Makefile Shortcuts

```bash
make install   # pip install -e ".[all]"
make test      # pytest with coverage
make lint      # ruff + mypy
make format    # ruff format
make clean     # remove build artifacts and __pycache__
```

## Project Layout

```
src/roboarm/
  core/           Layer 1: SE(3) math, DH/MDH transforms, types, exceptions
  robots/         Layer 2: Pre-built robot models (2-link, 3-link, 6-DOF MDH)
  kinematics/     Layer 3: FK wrapper, Jacobian, 5 IK solvers + registry
  agents/         Layer 4: Keyword-based AI coordinator, FK/IK agents, tools
  trajectory/     Polynomial interpolation + LSPB trapezoidal profiles
  workspace/      Monte Carlo workspace sampling and reachability
  visualization/  2D arm and workspace scatter plots (matplotlib)
  utils/          Angle utilities, input validation, default configs

tests/
  unit/           Fast isolated tests (170 tests, ~1.3 s)
  integration/    Cross-module + stress tests (64 tests, ~4.5 s)
  benchmarks/     Performance measurement stubs

docs/
  theory/         DH parameters, FK, IK, Jacobian, trajectory, workspace
  tutorials/      Getting started, custom robot, AI agents
```

## Key Architectural Rules

- Each layer depends **only** on layers below it — no circular imports.
- IK solvers self-register via `@IKSolverRegistry.register("name")`.
- Import `roboarm.kinematics.solvers` (the package) to register all 5 solvers at once.
- All library code uses `logging`, never `print()`.
- `from __future__ import annotations` in every module.
