# Contributing to roboarm-ai-toolkit

Thank you for your interest in contributing! This project is a **free and open
contribution** to the robotics and AI community. Everyone is welcome to use,
modify, and share it.

---

## Our Philosophy

This toolkit was built to help people learn and work with robot arm kinematics.
We believe knowledge should be freely shared for the benefit of all. If this
project helps you -- in education, research, hobby projects, or professional
work -- we are glad.

---

## How to Contribute

### Reporting Issues

- Open an issue describing the bug or feature request
- Include a minimal reproducible example if reporting a bug
- Mention your Python version and OS

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes following the coding standards below
4. Write tests for new functionality
5. Run the test suite: `pytest tests/ -v --tb=short`
6. Commit with a clear message
7. Open a pull request

### Coding Standards

- **Python 3.10+** compatibility required
- `from __future__ import annotations` in every module
- Type hints on all public functions (use `typing` module forms)
- Docstrings on every class and public method (Google style)
- Use `logging` module, never `print()` in library code
- Constants in `UPPER_SNAKE_CASE`
- Run `ruff check src/ tests/` before committing

### Test Requirements

- All existing tests must pass
- New features must include unit tests
- Target: maintain or improve current coverage (67%+)
- Test categories:
  - `tests/unit/` -- fast, isolated tests
  - `tests/integration/` -- cross-module tests, stress tests
  - `tests/benchmarks/` -- performance measurements

---

## Project Structure

```
src/roboarm/
  core/           Layer 1: Math foundations (transforms, rotations, types)
  kinematics/     Layer 3: FK, IK solvers, Jacobian
  robots/         Layer 2: Pre-defined robot models
  agents/         Layer 4: AI natural language interface
  trajectory/     Trajectory planning (LSPB, interpolation)
  workspace/      Reachability analysis
  visualization/  Matplotlib plotting
  utils/          Helpers (angles, validation, config)
```

Each layer depends only on layers below it. No circular imports.

---

## Code of Conduct

- Be respectful and constructive
- Focus on the work, not the person
- Welcome newcomers warmly
- Help others learn

---

## License

This project is released under the MIT License with an additional ethical
use statement. See [LICENSE](LICENSE) and [USAGE_TERMS.md](USAGE_TERMS.md).

**In short:** Use it freely for any good cause. Learn from it. Build on it.
Share it forward.
