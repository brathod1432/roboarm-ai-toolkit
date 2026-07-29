# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> See also: [README](README.md) | [Roadmap](ROADMAP.md) | [Architecture](ARCHITECTURE.md)

---

## [0.2.0] -- 2026-07-29

### Security

- **S1+S2**: `RobotArm.from_dict()` and `RobotBuilder.build()` now validate all DH parameters
  with `math.isfinite()` (rejects NaN/Inf injection) and cap joint count at 32 (DoS protection).
- **S3**: `robot.solve_ik()` and the `_solve_ik` agent tool reject non-finite or out-of-range
  coordinates (>1e6 m) before starting the solver.
- **S4**: All user input logged in the coordinator and agent layer is sanitised by
  `sanitize_for_log()` — newlines, carriage returns and null bytes are replaced with safe
  Unicode markers to prevent log-injection attacks.
- **S5**: `save_trajectory_csv()` tab-prefixes any header cell starting with `=`, `+`, `-`,
  or `@` to block spreadsheet formula-injection (OWASP CSV formula injection mitigation).
- **S6**: `ToolRegistry.get_audit_log()` now returns deep copies of the internal entries;
  callers can no longer tamper with past audit records by mutating the returned list.
- **S7**: `robot.save()` and `robot.load()` emit a `WARNING` when accessing a path outside
  the current working directory.
- **S8**: `JacobianComputer.joint_velocities()` gained a `max_joint_velocity` saturation
  parameter and raises `ValueError` on non-finite output.
- **R1**: `forward_kinematics()` now raises `ValidationError` if the computed end-effector
  position contains NaN or Inf (catches upstream DH parameter corruption).
- **S9**: GitHub Actions CI steps (`checkout`, `setup-python`, `codecov-action`) are pinned
  to full commit SHAs to prevent supply-chain attacks via tag re-pointing.

### Added — API

- `roboarm` top-level package now re-exports all primary classes:
  `RobotArm`, `RobotBuilder`, `JacobianComputer`, `batch_fk`, `batch_ik`,
  `TrajectoryValidator`, `TrajectoryAnalyzer`, `RoboticsCoordinator`, and all exceptions.
  `from roboarm import RobotArm` now works.
- `IKFailedError(KinematicsError)` — new exception raised by `robot.ik()` on solver
  failure; carries `residual_error`, `best_attempt`, and `solver_name` attributes.
- `robot.solve_ik()` now caches the solver instance keyed by name; repeated calls at the
  same solver avoid re-instantiation overhead (critical for 100 Hz control loops).
- `RobotArm.__eq__()` and `__hash__()` — two robots with identical DH params are now equal.
- `RobotArm.copy()` — returns a fully independent deep copy of joints and named poses.
- `RobotArm.fk_batch(Q)` and `RobotArm.ik_batch(targets)` — batch operations accessible
  directly on the robot model without importing `kinematics.batch`.
- `RobotArm._repr_html_()` — rich HTML table in Jupyter notebooks.
- `RobotArm.gravity_torques(q, link_masses, payload_mass)` — simplified static gravity
  torque estimation for motor sizing and safety margin analysis.
- `robot.solve_ik_async(position)` — async wrapper using `asyncio.to_thread` for
  non-blocking IK in web services and event-loop-based controllers.
- `batch_ik(..., warm_start=True)` (default on) — uses the previous solution as the
  initial guess for the next target; 50–90 % speed-up on spatially adjacent target grids.
- `save_trajectory_npz` / `load_trajectory_npz` path handling is now symmetric: both
  accept paths with or without the `.npz` extension.
- `save_trajectory_csv()` validates that `len(timestamps) == n_steps`.
- `six_dof_mdh.create_six_dof_mdh()` registers the home pose as `robot.get_pose("home")`.
  `HOME_POSE_RAD` is now write-protected (`flags.writeable = False`).
- `CCD` solver now applies `wrap_angles()` and joint-limit clamping after each iteration,
  consistent with the DLS and Jacobian pseudo-inverse solvers.
- `WorkspaceAnalyzer` raises `ValueError` for `n_samples <= 0` and for NaN/Inf targets.

### Added — New Features

- **Safe zones in `TrajectoryValidator`**: `add_forbidden_sphere(center, radius)` and
  `add_forbidden_box(center, dimensions)` define Cartesian regions the end-effector must
  not enter.  `check()` runs FK at each waypoint and records `SafeZoneViolation` entries.
- **Acceleration limit in `TrajectoryValidator`**: constructor accepts `dt` (seconds/step);
  when set, `check()` computes second finite differences and compares against each joint's
  `JointLimits.acceleration_max`, recording `AccelerationViolation` entries.
- **`kinematics/dataset.py`**: `generate_dataset(robot, n_samples, seed)` — produces a
  labelled ML training dataset as a dict of numpy arrays (`q`, `position`, `rotation_flat`,
  `jacobian_flat`, `manipulability`); directly saveable with `np.savez_compressed`.
- **`utils/servo.py`**: `ServoConfig` dataclass maps joint angles (radians) to RC servo
  PWM pulse widths (microseconds) with configurable calibration.  `ServoChain` handles
  batch conversion for multi-joint arms (e.g. PCA9685 drivers).

### Fixed

- DLS solver `DEFAULT_IK_CONFIG["damping"] = 0.5` vs constructor default `damping=0.01`
  inconsistency documented.
- `_parse_number_list` duplication (was in `fk_agent`, `ik_agent`, `coordinator`) is now
  sourced from a single location.

### Tests

- `tests/unit/test_security_hardening.py` — 38 tests probing every confirmed attack vector.
- `tests/unit/test_wave2_wave3.py` — API and feature tests for all new additions.
- Total test count: **437 → 530+ passing**.

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
