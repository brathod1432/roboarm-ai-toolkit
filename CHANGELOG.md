# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> See also: [README](README.md) | [Roadmap](ROADMAP.md) | [Architecture](ARCHITECTURE.md)

---

## [0.1.0] -- 2025-07-24

### Added

**Core Mathematics (Layer 1)**
- `core/types.py` -- DHParams, JointConfig, JointLimits, EndEffectorPose, IKSolution dataclasses
- `core/exceptions.py` -- RobotArmError hierarchy (7 exception types)
- `core/transform.py` -- Standard DH and Modified DH 4x4 homogeneous transforms
- `core/rotations.py` -- SO(3) conversions: Euler, axis-angle, quaternion (with Rodrigues and Shepperd methods)
- `core/robot.py` -- RobotArm model with FK chain computation, joint positions, cumulative transforms

**Robot Models (Layer 2)**
- `robots/two_link_planar.py` -- 2-DOF RR planar arm with configurable link lengths
- `robots/three_link_planar.py` -- 3-DOF RRR redundant planar arm
- `robots/six_dof_mdh.py` -- 6-DOF serial manipulator using Modified DH convention (7 links, 6 DOF)

**Kinematics Engine (Layer 3)**
- `kinematics/jacobian.py` -- Geometric and numerical Jacobian, manipulability index, singularity detection
- `kinematics/inverse.py` -- IKSolverBase ABC with Strategy pattern
- 5 IK solvers: analytical (2-link), Jacobian pseudoinverse, damped least squares, CCD, FABRIK
- `kinematics/solvers/registry.py` -- Solver Registry pattern for runtime discovery

**AI Agent Layer (Layer 4)**
- `agents/tools.py` -- ToolDefinition and ToolRegistry (OpenAI-compatible schemas)
- `agents/base_agent.py` -- AgentMessage, AgentMemory, BaseAgent ABC
- `agents/robotics_tools.py` -- 5 tools: describe, FK, IK, Jacobian, solver comparison
- `agents/fk_agent.py` -- FK specialist with keyword-based intent parsing
- `agents/ik_agent.py` -- IK specialist with coordinate extraction
- `agents/coordinator.py` -- Multi-agent router

**Supporting Modules**
- `trajectory/interpolation.py` -- Linear, cubic, quintic joint-space interpolation
- `trajectory/lspb.py` -- LSPB trapezoidal velocity profiles
- `workspace/analysis.py` -- Monte Carlo workspace sampling and reachability
- `visualization/arm_plot.py` -- 2D arm configuration plotting
- `visualization/workspace_plot.py` -- Workspace scatter visualization
- `utils/angle_utils.py` -- Angle wrapping, deg/rad conversion
- `utils/validation.py` -- Input validators
- `utils/config.py` -- Default configuration dictionaries

**Testing (234 tests, all passing)**
- Unit tests: types, transforms, rotations, FK, IK (78 tests)
- Accuracy tests: 200-config sweeps, sub-mm precision verification (28 tests)
- Negative tests: NaN, Inf, None, wrong types, invalid configs (40 tests)
- Security tests: injection attacks, unsupported ops, access control (24 tests)
- Stress tests: 10K FK, 500 IK, 1K Jacobians, performance benchmarks (12 tests)
- Integration: FK<->IK roundtrip across 10 arm geometries, 6-DOF MDH (52 tests)

**Examples**
- 5 runnable example scripts demonstrating FK, IK, Jacobian, and AI agents

**Documentation**
- Theory docs: DH parameters, FK, IK, Jacobian
- Tutorials: Getting started, custom robot, AI agents
- Architecture, Roadmap, Contributing, Usage Terms
- MIT License with ethical use statement

**Project Infrastructure**
- `pyproject.toml` with setuptools src-layout
- GitHub Actions CI for Python 3.10-3.12
- Makefile with install, test, lint, format targets
