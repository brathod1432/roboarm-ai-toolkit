"""Comprehensive unit tests for the 13 new use-cases in roboarm-ai-toolkit.

Covers:
  1.  TestSolveIKShortcut    — RobotArm.solve_ik / .ik
  2.  TestNamedPoses         — save_pose, get_pose, fk_at, list_poses, delete_pose
  3.  TestRobotSerialization — to_dict, from_dict, save, load
  4.  TestRobotBuilder       — RobotBuilder fluent DSL
  5.  TestTrajectoryIO       — CSV and NPZ save/load round-trips
  6.  TestViaPointTrajectory — via_point_trajectory multi-segment planner
  7.  TestTrajectoryValidator — TrajectoryValidator.check
  8.  TestTrajectoryAnalysis  — TrajectoryAnalyzer.analyze
  9.  TestJacobianVelocityControl — joint_velocities, manipulability_gradient
  10. TestBatchKinematics    — batch_fk, batch_ik
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. Standard library
# ---------------------------------------------------------------------------
import math
import os
import tempfile

# ---------------------------------------------------------------------------
# 2. Third-party — matplotlib backend MUST be set before any pyplot import
# ---------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 3. Trigger IK-solver self-registration (all 5 solvers)
# ---------------------------------------------------------------------------
import roboarm.kinematics.solvers  # noqa: F401

# ---------------------------------------------------------------------------
# 4. Application imports — roboarm modules
# ---------------------------------------------------------------------------
from roboarm.core.builder import RobotBuilder
from roboarm.core.exceptions import KinematicsError, ValidationError
from roboarm.core.robot import RobotArm
from roboarm.core.types import EndEffectorPose, IKSolution
from roboarm.kinematics.batch import batch_fk, batch_ik
from roboarm.kinematics.jacobian import JacobianComputer
from roboarm.robots.six_dof_mdh import create_six_dof_mdh
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.trajectory.analysis import TrajectoryAnalyzer, TrajectoryMetrics
from roboarm.trajectory.io import (
    load_trajectory_csv,
    load_trajectory_npz,
    save_trajectory_csv,
    save_trajectory_npz,
)
from roboarm.trajectory.multipoint import via_point_trajectory
from roboarm.trajectory.validation import TrajectoryReport, TrajectoryValidator

# ===========================================================================
# 1. TestSolveIKShortcut
# ===========================================================================


class TestSolveIKShortcut:
    """Tests for RobotArm.solve_ik and RobotArm.ik convenience methods.

    The 2-link planar robot has L1 = L2 = 1.0, so its workspace radius
    spans [0, 2.0] m.  The reachable target (1.0, 0.5) has distance ≈ 1.12 m;
    the unreachable target (5.0, 5.0) has distance ≈ 7.07 m.
    """

    def test_solve_ik_returns_iksolution_with_success_true(self):
        """solve_ik for a reachable target must return IKSolution(success=True)."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        result = robot.solve_ik([1.0, 0.5])
        assert isinstance(result, IKSolution)
        assert result.success is True

    def test_ik_returns_ndarray(self):
        """ik() must return a numpy ndarray, not an IKSolution."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        q = robot.ik([1.0, 0.5])
        assert isinstance(q, np.ndarray)
        assert q.shape == (robot.n_dof,)

    def test_ik_keyword_form_x_y(self):
        """ik(x=…, y=…) keyword form returns correct joint-angle array."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        q = robot.ik(x=1.0, y=0.5)
        assert isinstance(q, np.ndarray)
        assert q.shape == (robot.n_dof,)
        # Verify the FK round-trip error is small
        pose = robot.forward_kinematics(q)
        err = np.linalg.norm(pose.position[:2] - np.array([1.0, 0.5]))
        assert err < 1e-2, f"FK round-trip error too large: {err:.4e}"

    def test_ik_positional_form(self):
        """ik(position=[x, y]) positional form returns a valid joint-angle array."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        q = robot.ik(position=[1.0, 0.5])
        assert isinstance(q, np.ndarray)
        pose = robot.forward_kinematics(q)
        err = np.linalg.norm(pose.position[:2] - np.array([1.0, 0.5]))
        assert err < 1e-2, f"FK round-trip error too large: {err:.4e}"

    def test_ik_raises_kinematics_error_for_unreachable_target(self):
        """ik() must raise KinematicsError when the target is outside the workspace."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        with pytest.raises(KinematicsError):
            robot.ik(x=5.0, y=5.0)

    def test_solve_ik_non_default_solver(self):
        """solve_ik with solver_name='analytical_2link' must return success=True."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        result = robot.solve_ik([1.0, 0.5], solver_name="analytical_2link")
        assert isinstance(result, IKSolution)
        assert result.success is True


# ===========================================================================
# 2. TestNamedPoses
# ===========================================================================


class TestNamedPoses:
    """Tests for the named pose store: save_pose, get_pose, list_poses,
    fk_at, and delete_pose."""

    def test_save_and_get_pose_round_trip(self):
        """save_pose followed by get_pose must return the same angles."""
        robot = create_two_link_planar()
        robot.save_pose("home", [0.0, 0.0])
        q = robot.get_pose("home")
        np.testing.assert_allclose(q, [0.0, 0.0])

    def test_list_poses_returns_sorted_names(self):
        """list_poses must return pose names in lexicographic sort order."""
        robot = create_two_link_planar()
        robot.save_pose("place", [1.0, 0.5])
        robot.save_pose("home", [0.0, 0.0])
        robot.save_pose("pick", [0.5, -0.3])
        names = robot.list_poses()
        assert names == sorted(names)
        assert set(names) == {"home", "pick", "place"}

    def test_fk_at_matches_forward_kinematics(self):
        """fk_at must return the same EndEffectorPose as forward_kinematics."""
        robot = create_two_link_planar()
        q = [0.5, -0.3]
        robot.save_pose("test_config", q)
        fk_direct = robot.forward_kinematics(q)
        fk_named = robot.fk_at("test_config")
        np.testing.assert_allclose(fk_named.position, fk_direct.position, atol=1e-12)

    def test_get_pose_raises_key_error_for_missing_name(self):
        """get_pose raises KeyError for an unknown pose identifier."""
        robot = create_two_link_planar()
        with pytest.raises(KeyError):
            robot.get_pose("nonexistent_pose")

    def test_delete_pose_removes_entry(self):
        """delete_pose must remove the pose; subsequent get_pose raises KeyError."""
        robot = create_two_link_planar()
        robot.save_pose("temp", [0.1, 0.2])
        assert "temp" in robot.list_poses()
        robot.delete_pose("temp")
        assert "temp" not in robot.list_poses()
        with pytest.raises(KeyError):
            robot.get_pose("temp")

    def test_save_pose_raises_validation_error_for_wrong_dof(self):
        """save_pose raises ValidationError when angle count does not match n_dof."""
        robot = create_two_link_planar()  # n_dof == 2
        with pytest.raises(ValidationError):
            robot.save_pose("bad_pose", [0.1, 0.2, 0.3])  # 3 angles for 2-DOF robot


# ===========================================================================
# 3. TestRobotSerialization
# ===========================================================================


class TestRobotSerialization:
    """Tests for RobotArm.to_dict, from_dict, save, and load."""

    def test_to_dict_includes_name_and_joints(self):
        """to_dict must include 'name' and 'joints' keys with correct values."""
        robot = create_two_link_planar()
        d = robot.to_dict()
        assert "name" in d
        assert "joints" in d
        assert d["name"] == robot.name
        assert len(d["joints"]) == robot.n_joints

    def test_from_dict_recreates_robot_with_same_n_dof(self):
        """from_dict must produce a robot with the same n_dof as the original."""
        robot = create_two_link_planar()
        d = robot.to_dict()
        robot2 = RobotArm.from_dict(d)
        assert robot2.n_dof == robot.n_dof

    def test_poses_survive_to_dict_from_dict_round_trip(self):
        """Named poses stored before to_dict must be accessible after from_dict."""
        robot = create_two_link_planar()
        robot.save_pose("home", [0.0, 0.0])
        robot.save_pose("pick", [0.5, -0.3])
        d = robot.to_dict()
        robot2 = RobotArm.from_dict(d)
        np.testing.assert_allclose(robot2.get_pose("home"), [0.0, 0.0])
        np.testing.assert_allclose(robot2.get_pose("pick"), [0.5, -0.3])

    def test_save_load_round_trip_produces_identical_fk(self):
        """save/load round-trip must reproduce identical forward-kinematics results."""
        robot = create_two_link_planar()
        q = [0.5, -0.3]
        pose_orig = robot.forward_kinematics(q)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            tmp_path = fh.name
        try:
            robot.save(tmp_path)
            robot2 = RobotArm.load(tmp_path)
            pose_loaded = robot2.forward_kinematics(q)
            np.testing.assert_allclose(
                pose_orig.position, pose_loaded.position, atol=1e-12
            )
        finally:
            os.unlink(tmp_path)

    def test_load_is_classmethod_works_on_6dof_robot(self):
        """RobotArm.load (classmethod) correctly reconstructs a 6-DOF robot."""
        robot6 = create_six_dof_mdh()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            tmp_path = fh.name
        try:
            robot6.save(tmp_path)
            robot6_loaded = RobotArm.load(tmp_path)
            assert robot6_loaded.n_dof == 6
        finally:
            os.unlink(tmp_path)


# ===========================================================================
# 4. TestRobotBuilder
# ===========================================================================


class TestRobotBuilder:
    """Tests for the RobotBuilder fluent DSL."""

    def test_two_revolute_joints_gives_n_dof_2(self):
        """Building two revolute joints must give n_dof == 2."""
        robot = (
            RobotBuilder("Test2DOF")
            .add_revolute(a=1.0, name="J1")
            .add_revolute(a=0.8, name="J2")
            .build()
        )
        assert robot.n_dof == 2

    def test_add_revolute_limits_in_degrees_converts_to_radians(self):
        """add_revolute with limits=(-180, 180) must store them as (-π, π) rad."""
        robot = (
            RobotBuilder("LimitTest")
            .add_revolute(a=1.0, limits=(-180, 180), limits_in_degrees=True)
            .build()
        )
        lim = robot.joint_limits[0]
        assert lim is not None
        assert lim.lower == pytest.approx(-math.pi, abs=1e-9)
        assert lim.upper == pytest.approx(math.pi, abs=1e-9)

    def test_add_fixed_creates_non_variable_joint(self):
        """add_fixed must produce a joint with is_variable=False; n_dof stays 1."""
        robot = (
            RobotBuilder("FixedTest")
            .add_revolute(a=1.0, name="J1")
            .add_fixed(d=0.1, name="TCP")
            .build()
        )
        assert robot.n_dof == 1       # only the revolute counts
        assert robot.n_joints == 2    # total joints including fixed
        assert robot.joints[-1].is_variable is False

    def test_build_before_any_joints_raises_value_error(self):
        """build() with no joints added must raise ValueError."""
        with pytest.raises(ValueError):
            RobotBuilder("Empty").build()

    def test_3dof_arm_fk_at_all_zeros_is_correct(self):
        """3-DOF planar arm at all-zeros must place the end-effector at L1+L2+L3."""
        L1, L2, L3 = 1.0, 0.8, 0.6
        robot = (
            RobotBuilder("3DOF")
            .add_revolute(a=L1, name="J1")
            .add_revolute(a=L2, name="J2")
            .add_revolute(a=L3, name="J3")
            .build()
        )
        pose = robot.forward_kinematics([0.0, 0.0, 0.0])
        assert pose.x == pytest.approx(L1 + L2 + L3, abs=1e-9)
        assert pose.y == pytest.approx(0.0, abs=1e-9)


# ===========================================================================
# 5. TestTrajectoryIO
# ===========================================================================


class TestTrajectoryIO:
    """Tests for trajectory CSV and NPZ file I/O."""

    @staticmethod
    def _make_traj(n: int = 10, n_dof: int = 2) -> np.ndarray:
        """Return a simple (n, n_dof) linearly spaced test trajectory."""
        return np.column_stack(
            [np.linspace(0.0, 0.5 * (i + 1), n) for i in range(n_dof)]
        )

    def test_csv_round_trip_preserves_trajectory_values(self):
        """save_trajectory_csv / load_trajectory_csv must preserve joint angles."""
        traj = self._make_traj()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
            tmp_path = fh.name
        try:
            save_trajectory_csv(tmp_path, traj)
            traj_loaded, _, _ = load_trajectory_csv(tmp_path)
            np.testing.assert_allclose(traj_loaded, traj, atol=1e-10)
        finally:
            os.unlink(tmp_path)

    def test_csv_with_timestamps_preserves_time_array(self):
        """save/load CSV with explicit timestamps must preserve the time vector."""
        traj = self._make_traj()
        t = np.linspace(0.0, 1.0, 10)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
            tmp_path = fh.name
        try:
            save_trajectory_csv(tmp_path, traj, timestamps=t)
            _, t_loaded, _ = load_trajectory_csv(tmp_path)
            np.testing.assert_allclose(t_loaded, t, atol=1e-10)
        finally:
            os.unlink(tmp_path)

    def test_csv_metadata_contains_joint_names(self):
        """load_trajectory_csv metadata must contain 'joint_names' matching what was saved."""
        traj = self._make_traj()
        joint_names = ["Shoulder_rad", "Elbow_rad"]
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as fh:
            tmp_path = fh.name
        try:
            save_trajectory_csv(tmp_path, traj, joint_names=joint_names)
            _, _, meta = load_trajectory_csv(tmp_path)
            assert "joint_names" in meta
            assert meta["joint_names"] == joint_names
        finally:
            os.unlink(tmp_path)

    def test_npz_round_trip_preserves_trajectory_values(self):
        """save_trajectory_npz / load_trajectory_npz must preserve joint angles."""
        traj = self._make_traj()
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = os.path.join(tmp_dir, "trajectory")
            save_trajectory_npz(base_path, traj)
            # save_trajectory_npz appends ".npz" to the given path
            traj_loaded, _, _ = load_trajectory_npz(base_path + ".npz")
            np.testing.assert_allclose(traj_loaded, traj, atol=1e-12)

    def test_npz_with_metadata_dict_preserves_custom_fields(self):
        """Metadata dict passed to save_trajectory_npz must be retrievable."""
        traj = self._make_traj()
        meta_in = {"robot": "2-link-planar", "solver": "dls", "version": 1}
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_path = os.path.join(tmp_dir, "trajectory")
            save_trajectory_npz(base_path, traj, metadata=meta_in)
            _, _, meta_out = load_trajectory_npz(base_path + ".npz")
        assert meta_out["robot"] == "2-link-planar"
        assert meta_out["solver"] == "dls"


# ===========================================================================
# 6. TestViaPointTrajectory
# ===========================================================================


class TestViaPointTrajectory:
    """Tests for via_point_trajectory multi-segment planner."""

    # Three waypoints with sign-consistent joint slopes between segments
    _WAYPOINTS_3 = [
        [0.0, 0.0],
        [0.5, -0.5],
        [1.0, 0.0],
    ]

    def test_shape_3_waypoints_20_steps_per_segment(self):
        """3 waypoints × 20 steps/segment → total (41, 2) shape."""
        traj = via_point_trajectory(self._WAYPOINTS_3, n_steps_per_segment=20)
        # total_steps = (N-1) * n_steps_per_segment + 1 = 2*20+1 = 41
        assert traj.shape == (41, 2)

    def test_first_row_equals_first_waypoint(self):
        """First row of the trajectory must exactly equal the first waypoint."""
        traj = via_point_trajectory(self._WAYPOINTS_3, n_steps_per_segment=20)
        np.testing.assert_allclose(traj[0], self._WAYPOINTS_3[0], atol=1e-12)

    def test_last_row_equals_last_waypoint(self):
        """Last row of the trajectory must exactly equal the last waypoint."""
        traj = via_point_trajectory(self._WAYPOINTS_3, n_steps_per_segment=20)
        np.testing.assert_allclose(traj[-1], self._WAYPOINTS_3[-1], atol=1e-12)

    def test_linear_method_produces_monotone_trajectory(self):
        """method='linear' with monotone waypoints must give a monotone trajectory."""
        mono_waypoints = [[0.0, 0.0], [0.5, -0.5], [1.0, -1.0]]
        traj = via_point_trajectory(
            mono_waypoints, n_steps_per_segment=20, method="linear"
        )
        # Joint 0 values must be non-decreasing
        assert np.all(np.diff(traj[:, 0]) >= -1e-12)
        # Joint 1 values must be non-increasing
        assert np.all(np.diff(traj[:, 1]) <= 1e-12)

    def test_c1_continuity_velocity_jump_at_junction_less_than_threshold(self):
        """Blended cubic: velocity finite-difference jump at the junction < 0.05."""
        traj = via_point_trajectory(
            self._WAYPOINTS_3,
            n_steps_per_segment=20,
            method="cubic",
            blend_velocities=True,
        )
        # Junction point is at index 20 (first point of segment 2 = waypoint 1)
        junction = 20
        left_vel = traj[junction] - traj[junction - 1]
        right_vel = traj[junction + 1] - traj[junction]
        jump = float(np.max(np.abs(right_vel - left_vel)))
        assert jump < 0.05, f"Velocity jump at junction too large: {jump:.4e}"

    def test_raises_value_error_for_fewer_than_2_waypoints(self):
        """via_point_trajectory must raise ValueError when given only 1 waypoint."""
        with pytest.raises(ValueError):
            via_point_trajectory([[0.0, 0.0]])

    def test_raises_value_error_for_inconsistent_dof(self):
        """via_point_trajectory must raise ValueError for waypoints with different DOF."""
        with pytest.raises(ValueError):
            via_point_trajectory([[0.0, 0.0], [1.0, 2.0, 3.0]])


# ===========================================================================
# 7. TestTrajectoryValidator
# ===========================================================================


class TestTrajectoryValidator:
    """Tests for TrajectoryValidator.check()."""

    def test_clean_trajectory_within_limits_is_safe(self):
        """Trajectory within joint limits and away from singularities → is_safe=True."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        validator = TrajectoryValidator(robot)
        # [0.5, 0.5] is inside [-π, π] and has manipulability ≈ 0.48 >> 1e-4
        traj = np.tile([0.5, 0.5], (10, 1))
        report = validator.check(traj)
        assert isinstance(report, TrajectoryReport)
        assert report.is_safe is True
        assert len(report.limit_violations) == 0

    def test_out_of_limit_joint_generates_limit_violation(self):
        """A joint angle exceeding its limit must produce at least one LimitViolation."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        # Disable singularity check to isolate the limit-violation logic
        validator = TrajectoryValidator(robot, check_singularities=False)
        traj = np.tile([0.5, 0.5], (5, 1))
        traj[2, 0] = 4.0  # J1 limit is [-π, π]; 4.0 > π ≈ 3.14 → violation
        report = validator.check(traj)
        assert len(report.limit_violations) > 0

    def test_violation_records_correct_step_joint_and_value(self):
        """LimitViolation must carry the exact step index and angle value."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        validator = TrajectoryValidator(robot, check_singularities=False)
        traj = np.tile([0.5, 0.5], (5, 1))
        traj[3, 1] = -4.0  # J2 below lower limit −π at step 3
        report = validator.check(traj)
        assert len(report.limit_violations) >= 1
        violation = next(v for v in report.limit_violations if v.step == 3)
        assert violation.value == pytest.approx(-4.0)

    def test_singularity_at_zero_config_is_flagged(self):
        """A fully-extended [0, 0] configuration is singular; must appear in report."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        validator = TrajectoryValidator(robot, check_singularities=True)
        # At q=[0,0]: J = [[0,0],[2,1]], det(JJT)=0, manipulability=0 < 1e-4
        traj = np.tile([0.0, 0.0], (3, 1))
        report = validator.check(traj)
        assert len(report.singularities) > 0

    def test_report_summary_returns_non_empty_string(self):
        """TrajectoryReport.summary() must return a non-empty descriptive string."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        validator = TrajectoryValidator(robot)
        traj = np.tile([0.5, 0.5], (5, 1))
        report = validator.check(traj)
        summary = report.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_wrong_trajectory_dof_raises_value_error(self):
        """Trajectory whose DOF does not match the robot must raise ValueError."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)  # n_dof == 2
        validator = TrajectoryValidator(robot)
        traj_wrong = np.zeros((5, 3))  # 3 columns for a 2-DOF robot
        with pytest.raises(ValueError):
            validator.check(traj_wrong)


# ===========================================================================
# 8. TestTrajectoryAnalysis
# ===========================================================================


class TestTrajectoryAnalysis:
    """Tests for TrajectoryAnalyzer.analyze()."""

    @staticmethod
    def _robot():
        return create_two_link_planar(link1=1.0, link2=1.0)

    @staticmethod
    def _moving_traj(n: int = 10) -> np.ndarray:
        """Simple 2-DOF trajectory that moves the arm from one config to another."""
        return np.column_stack(
            [np.linspace(0.3, 0.8, n), np.linspace(0.3, 0.8, n)]
        )

    def test_path_length_positive_for_non_stationary_trajectory(self):
        """Cartesian path length must be > 0 when the arm actually moves."""
        robot = self._robot()
        analyzer = TrajectoryAnalyzer(robot)
        metrics = analyzer.analyze(self._moving_traj())
        assert isinstance(metrics, TrajectoryMetrics)
        assert metrics.cartesian_path_length_m > 0.0

    def test_manipulability_profile_has_correct_shape(self):
        """manipulability_profile must have shape (n_steps,)."""
        robot = self._robot()
        analyzer = TrajectoryAnalyzer(robot)
        n = 15
        metrics = analyzer.analyze(self._moving_traj(n=n))
        assert metrics.manipulability_profile.shape == (n,)

    def test_min_manipulability_equals_minimum_of_profile(self):
        """min_manipulability must equal np.min(manipulability_profile)."""
        robot = self._robot()
        analyzer = TrajectoryAnalyzer(robot)
        metrics = analyzer.analyze(self._moving_traj())
        expected = float(np.min(metrics.manipulability_profile))
        assert metrics.min_manipulability == pytest.approx(expected, rel=1e-6)

    def test_smoothness_is_zero_for_linear_trajectory(self):
        """A perfectly linear trajectory has zero second differences → smoothness == 0."""
        robot = self._robot()
        analyzer = TrajectoryAnalyzer(robot)
        # Exact linspace → all first differences equal → all second differences zero
        traj = np.column_stack(
            [np.linspace(0.5, 0.8, 10), np.linspace(0.5, 0.8, 10)]
        )
        metrics = analyzer.analyze(traj)
        assert metrics.smoothness == pytest.approx(0.0, abs=1e-12)

    def test_speed_profile_has_correct_shape_with_positive_dt(self):
        """With dt > 0, joint_speed_profile must have shape (n_steps-1, n_dof)."""
        robot = self._robot()
        analyzer = TrajectoryAnalyzer(robot, dt=0.05)
        n = 12
        metrics = analyzer.analyze(self._moving_traj(n=n))
        assert metrics.joint_speed_profile.shape == (n - 1, robot.n_dof)

    def test_speed_profile_is_empty_with_zero_dt(self):
        """With the default dt=0, joint_speed_profile must be an empty array."""
        robot = self._robot()
        analyzer = TrajectoryAnalyzer(robot, dt=0.0)
        metrics = analyzer.analyze(self._moving_traj())
        assert metrics.joint_speed_profile.size == 0


# ===========================================================================
# 9. TestJacobianVelocityControl
# ===========================================================================


class TestJacobianVelocityControl:
    """Tests for JacobianComputer.joint_velocities and manipulability_gradient."""

    @staticmethod
    def _robot():
        return create_two_link_planar(link1=1.0, link2=1.0)

    def test_joint_velocities_returns_correct_shape(self):
        """joint_velocities must return a (n_dof,) float64 array."""
        robot = self._robot()
        jc = JacobianComputer(robot)
        dq = jc.joint_velocities([0.5, -0.3], [0.1, 0.0])
        assert dq.shape == (robot.n_dof,)

    def test_zero_ee_velocity_yields_zero_joint_velocities(self):
        """Providing a zero end-effector velocity must return a zero joint-velocity vector."""
        robot = self._robot()
        jc = JacobianComputer(robot)
        dq = jc.joint_velocities([0.5, -0.3], [0.0, 0.0])
        np.testing.assert_allclose(dq, np.zeros(robot.n_dof), atol=1e-12)

    def test_damping_changes_result_near_singularity(self):
        """At a singular config, damping > 0 must give a different result than damping = 0."""
        robot = self._robot()
        jc = JacobianComputer(robot)
        # q=[0,0] is a singularity: J = [[0,0],[2,1]], det(JJT)=0
        q_sing = [0.0, 0.0]
        ee_vel = [0.1, 0.1]
        dq_undamped = jc.joint_velocities(q_sing, ee_vel, damping=0.0)
        dq_damped = jc.joint_velocities(q_sing, ee_vel, damping=0.1)
        assert not np.allclose(dq_undamped, dq_damped, atol=1e-6), (
            "Expected different results near singularity with/without damping"
        )

    def test_manipulability_gradient_returns_correct_shape(self):
        """manipulability_gradient must return a (n_dof,) array."""
        robot = self._robot()
        jc = JacobianComputer(robot)
        grad = jc.manipulability_gradient([0.5, -0.3])
        assert grad.shape == (robot.n_dof,)

    def test_manipulability_gradient_is_non_zero_at_non_singular_config(self):
        """Gradient at a non-singular configuration must have non-trivial magnitude."""
        robot = self._robot()
        jc = JacobianComputer(robot)
        # [0.5, -0.3] is non-singular: manipulability ≈ 0.29, well above 1e-4
        grad = jc.manipulability_gradient([0.5, -0.3])
        assert np.linalg.norm(grad) > 1e-10

    def test_wrong_ee_velocity_length_raises_value_error(self):
        """Providing wrong-length ee_velocity must raise ValueError."""
        robot = self._robot()  # planar → Jacobian is (2, n_dof) → expects 2-element input
        jc = JacobianComputer(robot)
        with pytest.raises(ValueError):
            jc.joint_velocities([0.5, -0.3], [0.1, 0.0, 0.0])  # 3 elements is wrong


# ===========================================================================
# 10. TestBatchKinematics
# ===========================================================================


class TestBatchKinematics:
    """Tests for batch_fk and batch_ik."""

    @staticmethod
    def _robot():
        return create_two_link_planar(link1=1.0, link2=1.0)

    def test_batch_fk_returns_n_by_3_array(self):
        """batch_fk with Q.shape=(20, 2) must return an ndarray of shape (20, 3)."""
        robot = self._robot()
        rng = np.random.default_rng(42)
        Q = rng.uniform(-1.0, 1.0, (20, 2))
        positions = batch_fk(robot, Q)
        assert isinstance(positions, np.ndarray)
        assert positions.shape == (20, 3)

    def test_batch_fk_single_row_1d_input(self):
        """batch_fk with a 1-D single-config input must reshape and return (1, 3)."""
        robot = self._robot()
        q = np.array([0.5, -0.3])  # shape (2,) — 1-D input
        positions = batch_fk(robot, q)
        assert isinstance(positions, np.ndarray)
        assert positions.shape == (1, 3)

    def test_batch_fk_full_true_returns_list_of_end_effector_pose(self):
        """batch_fk with full=True must return a list of EndEffectorPose instances."""
        robot = self._robot()
        Q = np.zeros((5, robot.n_dof))
        poses = batch_fk(robot, Q, full=True)
        assert isinstance(poses, list)
        assert len(poses) == 5
        assert all(isinstance(p, EndEffectorPose) for p in poses)

    def test_batch_ik_returns_correct_number_of_results(self):
        """batch_ik with 3 targets must return a list of 3 IKSolution objects."""
        robot = self._robot()
        targets = [[1.0, 0.5], [0.8, 0.6], [1.2, 0.3]]
        results = batch_ik(robot, targets)
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(isinstance(r, IKSolution) for r in results)

    def test_batch_ik_all_reachable_targets_succeed(self):
        """All three reachable targets (within L1+L2=2.0 radius) must succeed."""
        robot = self._robot()
        # All targets have ||(x,y)|| < 2.0 — well inside the workspace
        targets = [[1.0, 0.5], [0.8, 0.6], [1.2, 0.3]]
        results = batch_ik(robot, targets)
        assert all(r.success for r in results), (
            f"Unexpected failures: {[r.residual_error for r in results if not r.success]}"
        )

    def test_batch_fk_wrong_dof_raises_value_error(self):
        """batch_fk with Q whose column count != n_dof must raise ValueError."""
        robot = self._robot()  # n_dof == 2
        Q_wrong = np.zeros((5, 3))  # 3 columns for a 2-DOF robot
        with pytest.raises(ValueError):
            batch_fk(robot, Q_wrong)
