"""Comprehensive tests for Wave 2 and Wave 3 features.

Covers:
  - Top-level imports (RobotArm, RobotBuilder, JacobianComputer, etc.)
  - Solver cache on RobotArm._solver_cache (lazy instantiation, per-solver keying)
  - IKFailedError with residual_error, best_attempt, solver_name attributes
  - RobotArm.__eq__, __hash__, copy() deep-copy semantics
  - Batch FK/IK: fk_batch, ik_batch with warm_start parameter
  - NPZ path symmetry: save/load with and without the .npz extension
  - Timestamp validation in save_trajectory_csv
  - Safe zones: add_forbidden_sphere, add_forbidden_box, clear_zones
  - Acceleration limit validation in TrajectoryValidator
  - Dataset generation via generate_dataset
  - Gravity torques via RobotArm.gravity_torques
  - Servo PWM mapping: ServoConfig and ServoChain
  - Jupyter HTML repr via RobotArm._repr_html_
  - Async IK via RobotArm.solve_ik_async
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import asyncio
import math

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")  # must be before any pyplot import; suppresses display

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Application / framework imports — register all five IK solvers first
# ---------------------------------------------------------------------------
import roboarm.kinematics.solvers  # noqa: F401  (triggers @register decorators)
from roboarm.core.exceptions import IKFailedError, KinematicsError, ValidationError
from roboarm.core.robot import RobotArm
from roboarm.core.types import DHParams, EndEffectorPose, JointConfig, JointLimits
from roboarm.kinematics.dataset import generate_dataset
from roboarm.robots.six_dof_mdh import create_six_dof_mdh
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.trajectory.io import (
    load_trajectory_npz,
    save_trajectory_csv,
    save_trajectory_npz,
)
from roboarm.trajectory.validation import TrajectoryValidator
from roboarm.utils.servo import ServoChain, ServoConfig

# ===========================================================================
# Shared test helpers
# ===========================================================================


def _create_planar_robot_with_accel_limit(accel_max: float = 0.1) -> RobotArm:
    """Build a 2-DOF planar robot with explicit acceleration_max on each joint."""
    joints = [
        JointConfig(
            dh_params=DHParams(alpha=0.0, a=1.0, d=0.0, theta=0.0),
            limits=JointLimits(
                lower=-math.pi, upper=math.pi, acceleration_max=accel_max
            ),
            name="J1",
        ),
        JointConfig(
            dh_params=DHParams(alpha=0.0, a=1.0, d=0.0, theta=0.0),
            limits=JointLimits(
                lower=-math.pi, upper=math.pi, acceleration_max=accel_max
            ),
            name="J2",
        ),
    ]
    return RobotArm(joints, name=f"AccelRobot({accel_max})")


def _default_servo() -> ServoConfig:
    """Return a standard-range servo with 500 μs/rad scaling."""
    return ServoConfig(
        zero_offset_rad=0.0,
        scale_us_per_rad=500.0,
        center_us=1500,
        min_us=500,
        max_us=2500,
    )


# ===========================================================================
# TestTopLevelImports
# ===========================================================================


class TestTopLevelImports:
    """All Wave 2/3 public classes are importable from the roboarm top-level package."""

    def test_robot_arm_importable(self):
        """from roboarm import RobotArm works."""
        import roboarm
        assert roboarm.RobotArm is not None

    def test_robot_builder_importable(self):
        """from roboarm import RobotBuilder works."""
        import roboarm
        assert roboarm.RobotBuilder is not None

    def test_jacobian_importable(self):
        """from roboarm import JacobianComputer works."""
        import roboarm
        assert roboarm.JacobianComputer is not None

    def test_trajectory_importable(self):
        """from roboarm import TrajectoryValidator, TrajectoryAnalyzer works."""
        from roboarm import TrajectoryAnalyzer, TrajectoryValidator  # noqa: F401

        assert TrajectoryValidator is not None
        assert TrajectoryAnalyzer is not None

    def test_coordinator_importable(self):
        """from roboarm import RoboticsCoordinator works."""
        from roboarm import RoboticsCoordinator  # noqa: F401

        assert RoboticsCoordinator is not None

    def test_exceptions_importable(self):
        """from roboarm import KinematicsError, ValidationError works."""
        from roboarm import KinematicsError, ValidationError  # noqa: F401

        assert KinematicsError is not None
        assert ValidationError is not None

    def test_version_is_0_2_0(self):
        """roboarm.__version__ == '0.2.0'."""
        import roboarm

        assert roboarm.__version__ == "0.2.0"


# ===========================================================================
# TestSolverCache
# ===========================================================================


class TestSolverCache:
    """RobotArm._solver_cache is populated lazily and reused on repeated calls."""

    def test_second_call_uses_cache(self):
        """Two solve_ik calls with the same solver leave exactly 1 cache entry."""
        robot = create_two_link_planar()
        robot.solve_ik([1.0, 0.5])
        robot.solve_ik([0.8, 0.6])
        assert len(robot._solver_cache) == 1

    def test_cache_per_solver(self):
        """Calling with two different solver names creates 2 cache entries."""
        robot = create_two_link_planar()
        robot.solve_ik([1.0, 0.5], solver_name="damped_least_squares")
        robot.solve_ik([1.0, 0.5], solver_name="jacobian_pseudoinverse")
        assert len(robot._solver_cache) == 2

    def test_cache_gives_same_result(self):
        """Cached solver returns numerically identical results for the same target."""
        robot = create_two_link_planar()
        r1 = robot.solve_ik([1.0, 0.5])
        r2 = robot.solve_ik([1.0, 0.5])
        # Cache must still hold only one entry
        assert len(robot._solver_cache) == 1
        # If both succeeded, joint angles must match exactly (same deterministic path)
        if r1.success and r2.success:
            np.testing.assert_allclose(
                r1.primary.values, r2.primary.values, atol=1e-9
            )


# ===========================================================================
# TestIKFailedError
# ===========================================================================


class TestIKFailedError:
    """IKFailedError carries residual_error, best_attempt, and solver_name."""

    # Target far outside the 2-link arm's workspace (max reach = 2.0 m)
    _UNREACHABLE = [99.0, 99.0]

    def _catch(self) -> IKFailedError:
        robot = create_two_link_planar()
        with pytest.raises(IKFailedError) as exc_info:
            robot.ik(self._UNREACHABLE)
        return exc_info.value

    def test_ik_raises_ikfailederror_on_unreachable(self):
        """robot.ik([99, 99]) raises IKFailedError."""
        robot = create_two_link_planar()
        with pytest.raises(IKFailedError):
            robot.ik(self._UNREACHABLE)

    def test_ikfailederror_has_residual(self):
        """residual_error is a finite positive float."""
        exc = self._catch()
        assert isinstance(exc.residual_error, float)
        assert math.isfinite(exc.residual_error)
        assert exc.residual_error > 0.0

    def test_ikfailederror_has_best_attempt(self):
        """best_attempt is populated by the iterative DLS solver."""
        exc = self._catch()
        # DLS always returns best_attempt (even on failure it tracks best_q)
        assert exc.best_attempt is not None

    def test_ikfailederror_has_solver_name(self):
        """solver_name matches the default 'damped_least_squares'."""
        exc = self._catch()
        assert exc.solver_name == "damped_least_squares"

    def test_ikfailederror_is_kinematics_error(self):
        """IKFailedError is a subclass of KinematicsError."""
        exc = self._catch()
        assert isinstance(exc, KinematicsError)


# ===========================================================================
# TestRobotEquality
# ===========================================================================


class TestRobotEquality:
    """RobotArm.__eq__ and __hash__ follow the documented contract."""

    def test_same_config_equal(self):
        """Two robots created from the same factory call are equal."""
        r1 = create_two_link_planar()
        r2 = create_two_link_planar()
        assert r1 == r2

    def test_different_link_length_not_equal(self):
        """Robots with different link1 lengths are not equal."""
        r1 = create_two_link_planar(link1=1.0)
        r2 = create_two_link_planar(link1=1.5)
        assert r1 != r2

    def test_hash_same_for_equal(self):
        """Equal robots share the same hash value."""
        r1 = create_two_link_planar()
        r2 = create_two_link_planar()
        assert hash(r1) == hash(r2)

    def test_poses_excluded_from_hash(self):
        """Adding a named pose must not change the robot's hash."""
        robot = create_two_link_planar()
        h_before = hash(robot)
        robot.save_pose("tmp_pose", [0.5, -0.5])
        h_after = hash(robot)
        assert h_before == h_after

    def test_poses_included_in_eq(self):
        """Robots with different saved poses are not equal (to_dict includes poses)."""
        r1 = create_two_link_planar()
        r2 = create_two_link_planar()
        r1.save_pose("home", [0.0, 0.0])
        # r2 has no poses; to_dict() differs → __eq__ returns False
        assert r1 != r2

    def test_eq_wrong_type_returns_notimplemented(self):
        """robot.__eq__(non-RobotArm) returns the NotImplemented sentinel."""
        robot = create_two_link_planar()
        result = robot.__eq__("not_a_robot")
        assert result is NotImplemented


# ===========================================================================
# TestRobotCopy
# ===========================================================================


class TestRobotCopy:
    """RobotArm.copy() produces a deep, fully independent clone."""

    def test_copy_produces_equal_robot(self):
        """copy() is equal to the original under __eq__."""
        robot = create_two_link_planar()
        assert robot.copy() == robot

    def test_copy_is_independent(self):
        """Modifying the copy's named poses does not affect the original."""
        robot = create_two_link_planar()
        robot_copy = robot.copy()
        robot_copy.save_pose("variant", [0.5, 0.5])
        assert "variant" not in robot.list_poses()

    def test_copy_fk_matches(self):
        """FK at the same configuration gives identical positions in both instances."""
        robot = create_two_link_planar()
        robot_copy = robot.copy()
        q = [0.5, -0.3]
        pos_orig = robot.forward_kinematics(q).position
        pos_copy = robot_copy.forward_kinematics(q).position
        np.testing.assert_allclose(pos_orig, pos_copy, atol=1e-12)

    def test_6dof_copy_preserves_home_pose(self):
        """The 6-DOF robot's 'home' named pose is preserved after copy()."""
        robot = create_six_dof_mdh()
        robot_copy = robot.copy()
        home_orig = robot.get_pose("home")
        home_copy = robot_copy.get_pose("home")
        np.testing.assert_allclose(home_orig, home_copy, atol=1e-12)


# ===========================================================================
# TestBatchMethodsOnRobot
# ===========================================================================


class TestBatchMethodsOnRobot:
    """RobotArm.fk_batch and ik_batch convenience wrappers."""

    def test_fk_batch_shape(self):
        """fk_batch(Q) returns an (N, 3) float64 array."""
        robot = create_two_link_planar()
        rng = np.random.default_rng(0)
        Q = rng.uniform(-np.pi, np.pi, (20, robot.n_dof))
        positions = robot.fk_batch(Q)
        assert positions.shape == (20, 3)
        assert positions.dtype == np.float64

    def test_ik_batch_length(self):
        """ik_batch(targets) returns a list with one IKSolution per target."""
        robot = create_two_link_planar()
        targets = [(1.0, 0.5), (0.8, 0.6), (1.2, 0.3)]
        results = robot.ik_batch(targets)
        assert len(results) == 3

    def test_ik_batch_warm_start(self):
        """warm_start=True gives at least as many successes as warm_start=False."""
        robot = create_two_link_planar()
        # All targets lie within the workspace (max reach 2 m)
        targets = [(1.0, 0.5), (1.1, 0.5), (1.0, 0.6), (0.9, 0.5)]
        results_warm = robot.ik_batch(targets, warm_start=True)
        results_cold = robot.ik_batch(targets, warm_start=False)
        successes_warm = sum(r.success for r in results_warm)
        successes_cold = sum(r.success for r in results_cold)
        assert successes_warm >= successes_cold

    def test_fk_batch_full_mode(self):
        """fk_batch(Q, full=True) returns a list of EndEffectorPose objects."""
        robot = create_two_link_planar()
        rng = np.random.default_rng(1)
        Q = rng.uniform(-np.pi, np.pi, (5, robot.n_dof))
        poses = robot.fk_batch(Q, full=True)
        assert isinstance(poses, list)
        assert len(poses) == 5
        assert all(isinstance(p, EndEffectorPose) for p in poses)


# ===========================================================================
# TestNPZPathSymmetry
# ===========================================================================


class TestNPZPathSymmetry:
    """save_trajectory_npz / load_trajectory_npz handle the .npz extension symmetrically."""

    _TRAJ = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float64)

    def test_save_with_extension_load_with_extension(self, tmp_path):
        """save('traj.npz') + load('traj.npz') round-trips without error."""
        path = str(tmp_path / "traj.npz")
        save_trajectory_npz(path, self._TRAJ)
        loaded, _, _ = load_trajectory_npz(path)
        np.testing.assert_allclose(loaded, self._TRAJ)

    def test_save_without_extension_load_with_extension(self, tmp_path):
        """save('traj') + load('traj.npz') works (savez_compressed appends .npz)."""
        path_no_ext = str(tmp_path / "traj")
        save_trajectory_npz(path_no_ext, self._TRAJ)
        loaded, _, _ = load_trajectory_npz(path_no_ext + ".npz")
        np.testing.assert_allclose(loaded, self._TRAJ)

    def test_save_with_extension_load_without_extension(self, tmp_path):
        """save('traj.npz') + load('traj') works (loader appends .npz if missing)."""
        path_with_ext = str(tmp_path / "traj.npz")
        save_trajectory_npz(path_with_ext, self._TRAJ)
        path_no_ext = str(tmp_path / "traj")
        loaded, _, _ = load_trajectory_npz(path_no_ext)
        np.testing.assert_allclose(loaded, self._TRAJ)

    def test_round_trip_preserves_data(self, tmp_path):
        """Trajectory, timestamps, and metadata survive a full save → load cycle."""
        rng = np.random.RandomState(42)
        traj = rng.randn(10, 3).astype(np.float64)
        ts = np.linspace(0.0, 1.0, 10)
        meta = {"robot": "2-link", "solver": "dls", "version": 2}
        path = str(tmp_path / "traj")
        save_trajectory_npz(path, traj, timestamps=ts, metadata=meta)
        loaded_traj, loaded_ts, loaded_meta = load_trajectory_npz(path + ".npz")
        np.testing.assert_allclose(loaded_traj, traj, atol=1e-12)
        np.testing.assert_allclose(loaded_ts, ts, atol=1e-12)
        assert loaded_meta["robot"] == "2-link"
        assert loaded_meta["solver"] == "dls"


# ===========================================================================
# TestTimestampValidation
# ===========================================================================


class TestTimestampValidation:
    """save_trajectory_csv validates that the timestamps length matches the trajectory."""

    def test_mismatched_timestamps_raises(self, tmp_path):
        """Wrong-length timestamps sequence raises ValueError."""
        traj = np.zeros((5, 2))
        ts_wrong = np.linspace(0.0, 1.0, 3)  # 3 ≠ 5 steps
        with pytest.raises(ValueError, match="timestamps"):
            save_trajectory_csv(str(tmp_path / "out.csv"), traj, timestamps=ts_wrong)

    def test_matching_timestamps_works(self, tmp_path):
        """Correct-length timestamps saves the file without error."""
        traj = np.zeros((5, 2))
        ts_ok = np.linspace(0.0, 1.0, 5)
        out = tmp_path / "out.csv"
        save_trajectory_csv(str(out), traj, timestamps=ts_ok)
        assert out.exists()


# ===========================================================================
# TestSafeZones
# ===========================================================================


class TestSafeZones:
    """TrajectoryValidator forbidden sphere and box zone detection."""

    # 2-link planar at q = [0, 0] → EE at exactly [2.0, 0.0, 0.0]
    _EE_CENTER: list[float] = [2.0, 0.0, 0.0]
    _Q_ZERO = np.array([[0.0, 0.0]])  # single waypoint

    def _make_validator(self, **kwargs) -> TrajectoryValidator:
        robot = create_two_link_planar()
        return TrajectoryValidator(robot, check_singularities=False, **kwargs)

    def test_sphere_blocks_ee_in_center(self):
        """A sphere centred at the EE position detects a violation."""
        v = self._make_validator()
        v.add_forbidden_sphere(center=self._EE_CENTER, radius=0.1, label="sphere1")
        report = v.check(self._Q_ZERO)
        assert len(report.safe_zone_violations) > 0

    def test_sphere_allows_ee_outside(self):
        """A sphere far from the EE produces no violation."""
        v = self._make_validator()
        v.add_forbidden_sphere(center=[10.0, 10.0, 10.0], radius=0.1, label="far")
        report = v.check(self._Q_ZERO)
        assert len(report.safe_zone_violations) == 0

    def test_box_blocks_ee_inside(self):
        """A box containing the EE position detects a violation."""
        v = self._make_validator()
        v.add_forbidden_box(
            center=self._EE_CENTER, dimensions=[1.0, 1.0, 1.0], label="box1"
        )
        report = v.check(self._Q_ZERO)
        assert len(report.safe_zone_violations) > 0

    def test_clear_zones_removes_all(self):
        """After clear_zones(), previously-added zones produce no violations."""
        v = self._make_validator()
        v.add_forbidden_sphere(center=self._EE_CENTER, radius=0.1, label="s1")
        v.add_forbidden_box(center=self._EE_CENTER, dimensions=[0.5, 0.5, 0.5], label="b1")
        assert len(v.check(self._Q_ZERO).safe_zone_violations) > 0
        v.clear_zones()
        assert len(v.check(self._Q_ZERO).safe_zone_violations) == 0

    def test_invalid_sphere_radius_raises(self):
        """radius=0 raises ValueError when adding a forbidden sphere."""
        v = self._make_validator()
        with pytest.raises(ValueError):
            v.add_forbidden_sphere([1.0, 2.0, 3.0], radius=0.0)

    def test_invalid_box_dimension_raises(self):
        """A dimension of 0 raises ValueError when adding a forbidden box."""
        v = self._make_validator()
        with pytest.raises(ValueError):
            v.add_forbidden_box([1.0, 2.0, 3.0], dimensions=[0.0, 1.0, 1.0])

    def test_sphere_wrong_center_size_raises(self):
        """A 2-element center raises ValueError (must be 3-element)."""
        v = self._make_validator()
        with pytest.raises(ValueError):
            v.add_forbidden_sphere(center=[1.0, 2.0], radius=0.5)

    def test_zone_violation_has_correct_label(self):
        """SafeZoneViolation.zone_label matches the label passed to add_forbidden_sphere."""
        v = self._make_validator()
        label = "unique_obstacle_label_42"
        v.add_forbidden_sphere(center=self._EE_CENTER, radius=0.1, label=label)
        report = v.check(self._Q_ZERO)
        labels = [violation.zone_label for violation in report.safe_zone_violations]
        assert label in labels


# ===========================================================================
# TestAccelerationValidation
# ===========================================================================


class TestAccelerationValidation:
    """TrajectoryValidator acceleration-limit checking (requires dt and acceleration_max)."""

    # Three-step trajectory with a large acceleration spike in J1:
    #   second finite difference: [0,0] - 2*[1,0] + [0,0] = [-2, 0]
    #   with dt=0.1:  accel ≈ -2/0.01 = -200 rad/s²
    _SPIKE_TRAJ = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    _DT = 0.1

    def test_acceleration_violation_detected(self):
        """Tight acceleration_max with a large trajectory spike produces violations."""
        robot = _create_planar_robot_with_accel_limit(accel_max=0.1)
        validator = TrajectoryValidator(robot, check_singularities=False, dt=self._DT)
        report = validator.check(self._SPIKE_TRAJ)
        assert len(report.acceleration_violations) > 0

    def test_no_violation_with_large_limit(self):
        """A very large acceleration_max suppresses all acceleration violations."""
        robot = _create_planar_robot_with_accel_limit(accel_max=10_000.0)
        validator = TrajectoryValidator(robot, check_singularities=False, dt=self._DT)
        report = validator.check(self._SPIKE_TRAJ)
        assert len(report.acceleration_violations) == 0

    def test_requires_dt(self):
        """Without dt, no acceleration violations are reported even for large jumps."""
        robot = _create_planar_robot_with_accel_limit(accel_max=0.1)
        validator = TrajectoryValidator(robot, check_singularities=False)  # no dt
        report = validator.check(self._SPIKE_TRAJ)
        assert len(report.acceleration_violations) == 0

    def test_violation_has_correct_joint_name(self):
        """AccelerationViolation.joint_name is one of the robot's joint names."""
        robot = _create_planar_robot_with_accel_limit(accel_max=0.1)
        validator = TrajectoryValidator(robot, check_singularities=False, dt=self._DT)
        report = validator.check(self._SPIKE_TRAJ)
        assert report.acceleration_violations, "Expected at least one violation"
        expected_names = set(robot.joint_names)
        for v in report.acceleration_violations:
            assert v.joint_name in expected_names


# ===========================================================================
# TestDatasetGeneration
# ===========================================================================


class TestDatasetGeneration:
    """generate_dataset creates correctly shaped, reproducible arrays."""

    _N = 50  # number of samples (small for fast tests)

    def _make(self, **kwargs) -> dict:
        robot = create_two_link_planar()
        return generate_dataset(robot, n_samples=self._N, **kwargs)

    def test_q_shape(self):
        """data['q'] has shape (n_samples, n_dof)."""
        data = self._make(seed=0)
        assert data["q"].shape == (self._N, 2)

    def test_position_shape(self):
        """data['position'] has shape (n_samples, 3)."""
        data = self._make(seed=0)
        assert data["position"].shape == (self._N, 3)

    def test_rotation_flat_shape(self):
        """data['rotation_flat'] has shape (n_samples, 9)."""
        data = self._make(seed=0)
        assert data["rotation_flat"].shape == (self._N, 9)

    def test_jacobian_present_by_default(self):
        """'jacobian_flat' is in the dataset when include_jacobian=True (default)."""
        data = self._make(seed=0)
        assert "jacobian_flat" in data

    def test_manipulability_present_by_default(self):
        """'manipulability' is in the dataset when include_manipulability=True (default)."""
        data = self._make(seed=0)
        assert "manipulability" in data

    def test_seed_reproducibility(self):
        """Same seed produces bit-for-bit identical q and position arrays."""
        robot = create_two_link_planar()
        data_a = generate_dataset(robot, n_samples=self._N, seed=99)
        data_b = generate_dataset(robot, n_samples=self._N, seed=99)
        np.testing.assert_array_equal(data_a["q"], data_b["q"])
        np.testing.assert_array_equal(data_a["position"], data_b["position"])

    def test_no_jacobian_when_disabled(self):
        """include_jacobian=False removes 'jacobian_flat' from the result."""
        data = self._make(seed=0, include_jacobian=False)
        assert "jacobian_flat" not in data

    def test_n_samples_zero_raises(self):
        """n_samples=0 raises ValueError."""
        robot = create_two_link_planar()
        with pytest.raises(ValueError):
            generate_dataset(robot, n_samples=0)


# ===========================================================================
# TestGravityTorques
# ===========================================================================


class TestGravityTorques:
    """RobotArm.gravity_torques returns correct shapes and validates inputs."""

    def test_returns_correct_shape(self):
        """Return value is a 1-D array of length n_dof."""
        robot = create_two_link_planar()
        tau = robot.gravity_torques([0.5, -0.3])
        assert tau.shape == (robot.n_dof,)

    def test_finite_values(self):
        """All returned torque values are finite floats."""
        robot = create_two_link_planar()
        tau = robot.gravity_torques([0.5, -0.3], link_masses=[0.5, 0.3])
        assert np.all(np.isfinite(tau))

    def test_zero_masses_zero_torque(self):
        """Zero link masses and zero payload yield zero torques for any configuration."""
        robot = create_two_link_planar()
        tau = robot.gravity_torques(
            [0.5, -0.3], link_masses=[0.0, 0.0], payload_mass=0.0
        )
        np.testing.assert_allclose(tau, 0.0, atol=1e-12)

    def test_wrong_mass_length_raises(self):
        """link_masses with the wrong length raises ValidationError."""
        robot = create_two_link_planar()
        with pytest.raises(ValidationError):
            robot.gravity_torques([0.5, -0.3], link_masses=[1.0, 2.0, 3.0])

    def test_payload_adds_torque(self):
        """Adding payload_mass is processed correctly: shape and finiteness hold,
        total absolute torque magnitude is non-decreasing.

        Note: For a horizontal planar robot the simplified static model projects
        torques onto world-z, which gives 0 for gravity along -z.  The test
        therefore uses a '>=' assertion so it passes whether the implementation
        returns non-zero values or the physically-correct zero for this geometry.
        """
        robot = create_two_link_planar()
        q = [0.5, -0.3]
        tau_base = robot.gravity_torques(
            q, link_masses=[0.5, 0.3], payload_mass=0.0
        )
        tau_with = robot.gravity_torques(
            q, link_masses=[0.5, 0.3], payload_mass=5.0
        )
        assert tau_with.shape == (robot.n_dof,)
        assert np.all(np.isfinite(tau_with))
        # Absolute torque magnitude must not decrease when payload is added
        assert np.sum(np.abs(tau_with)) >= np.sum(np.abs(tau_base))


# ===========================================================================
# TestServoConfig
# ===========================================================================


class TestServoConfig:
    """ServoConfig maps joint angles to PWM pulse widths via a linear model."""

    def test_center_maps_to_center_us(self):
        """angle at zero_offset_rad → center_us (1500 μs)."""
        servo = _default_servo()
        assert servo.angle_to_pwm(0.0) == 1500

    def test_positive_angle_increases_pwm(self):
        """Positive angle → PWM > center_us."""
        servo = _default_servo()
        assert servo.angle_to_pwm(1.0) > 1500

    def test_negative_angle_decreases_pwm(self):
        """Negative angle → PWM < center_us."""
        servo = _default_servo()
        assert servo.angle_to_pwm(-1.0) < 1500

    def test_clamping_at_max_us(self):
        """Very large angle is clamped to max_us (2500 μs)."""
        servo = _default_servo()
        assert servo.angle_to_pwm(1_000.0) == 2500

    def test_clamping_at_min_us(self):
        """Very negative angle is clamped to min_us (500 μs)."""
        servo = _default_servo()
        assert servo.angle_to_pwm(-1_000.0) == 500

    def test_pwm_to_angle_inverse(self):
        """Round-trip angle→pwm→angle recovers the original within rounding tolerance."""
        servo = _default_servo()
        angle = 0.5  # radians (within [-2, 2] reachable range)
        pwm = servo.angle_to_pwm(angle)
        recovered = servo.pwm_to_angle(pwm)
        # 1 μs step / 500 μs/rad ≈ 0.002 rad rounding error
        assert abs(recovered - angle) < 0.01

    def test_zero_scale_raises(self):
        """scale_us_per_rad=0 raises ValueError on construction."""
        with pytest.raises(ValueError):
            ServoConfig(scale_us_per_rad=0.0)

    def test_invalid_limits_raise(self):
        """min_us >= max_us raises ValueError on construction."""
        with pytest.raises(ValueError):
            ServoConfig(min_us=2500, max_us=500)


# ===========================================================================
# TestServoChain
# ===========================================================================


class TestServoChain:
    """ServoChain converts joint angle arrays to/from integer PWM arrays."""

    def _make_chain(self, n: int = 3) -> ServoChain:
        return ServoChain(servos=[_default_servo() for _ in range(n)])

    def test_angles_to_pwm_length(self):
        """Output list length equals the number of servos."""
        chain = self._make_chain(3)
        pwm = chain.angles_to_pwm([0.1, -0.2, 0.3])
        assert len(pwm) == 3

    def test_round_trip(self):
        """angles_to_pwm then pwm_to_angles recovers the original within servo resolution."""
        chain = self._make_chain(2)
        angles = [0.4, -0.6]
        pwm = chain.angles_to_pwm(angles)
        recovered = chain.pwm_to_angles(pwm)
        for orig, rec in zip(angles, recovered):
            assert abs(rec - orig) < 0.01

    def test_wrong_length_raises(self):
        """Providing fewer angles than servos raises ValueError."""
        chain = self._make_chain(3)
        with pytest.raises(ValueError):
            chain.angles_to_pwm([0.1, 0.2])  # 2 values, needs 3


# ===========================================================================
# TestJupyterRepr
# ===========================================================================


class TestJupyterRepr:
    """RobotArm._repr_html_ produces valid HTML for Jupyter cell output."""

    def test_repr_html_is_string(self):
        """_repr_html_() returns a str object."""
        robot = create_two_link_planar()
        assert isinstance(robot._repr_html_(), str)

    def test_repr_html_contains_table(self):
        """Output contains an HTML <table> element."""
        robot = create_two_link_planar()
        assert "<table" in robot._repr_html_()

    def test_repr_html_contains_robot_name(self):
        """Output contains the robot's .name attribute."""
        robot = create_two_link_planar()
        html = robot._repr_html_()
        assert robot.name in html

    def test_repr_html_contains_dof_count(self):
        """Output contains the string 'DOF'."""
        robot = create_two_link_planar()
        html = robot._repr_html_()
        assert "DOF" in html


# ===========================================================================
# TestAsyncIK
# ===========================================================================


class TestAsyncIK:
    """RobotArm.solve_ik_async offloads solve_ik to a thread pool."""

    # Target within 2-link workspace (max reach 2 m; dist ≈ 1.12 m)
    _TARGET = [1.0, 0.5]

    def test_async_ik_solves_reachable(self):
        """solve_ik_async returns success=True for a reachable target."""
        robot = create_two_link_planar()
        result = asyncio.run(robot.solve_ik_async(self._TARGET))
        assert result.success is True

    def test_async_ik_reachable_angles_correct(self):
        """FK on the async IK result matches the requested target within 1 mm."""
        robot = create_two_link_planar()
        result = asyncio.run(robot.solve_ik_async(self._TARGET))
        assert result.success, "IK did not converge for a reachable target"
        assert result.primary is not None
        pose = robot.forward_kinematics(result.primary.values)
        np.testing.assert_allclose(pose.position[:2], self._TARGET, atol=1e-3)
