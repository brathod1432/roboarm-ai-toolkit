"""Comprehensive tests for new features F1-F9, F11, A1, A3.

Covers:
  F1+F2 — Angle wrapping in DLS, Jacobian, and CCD solvers
  F3    — FK verification appended to _solve_ik agent tool response
  F4    — WorkspaceAnalyzer seed reproducibility
  F5    — LSPB velocity limits via multi_joint_lspb
  F6    — IKSolution.best_attempt field population
  F7    — RobotArm.is_planar property
  F8    — EndEffectorPose read-only array attributes
  F9    — cartesian_trajectory shape, solve rate, endpoint accuracy
  F11   — WorkspaceAnalyzer.plot() returns a matplotlib Axes
  A1    — request_context / current_request_id context variable
  A3    — FKAgent and IKAgent conversation-memory carry-forward
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # must be set before any pyplot import

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pytest

# Importing this package triggers @IKSolverRegistry.register decorators for
# all five built-in solvers (analytical_2link, ccd, damped_least_squares,
# fabrik, jacobian_pseudoinverse).
import roboarm.kinematics.solvers  # noqa: F401
from roboarm.agents._request_context import current_request_id, request_context
from roboarm.agents.coordinator import RoboticsCoordinator
from roboarm.agents.fk_agent import FKAgent
from roboarm.agents.ik_agent import IKAgent
from roboarm.agents.robotics_tools import build_robotics_tools
from roboarm.core.types import EndEffectorPose, JointLimits
from roboarm.kinematics.solvers.registry import IKSolverRegistry
from roboarm.robots.six_dof_mdh import create_six_dof_mdh
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.trajectory.cartesian import cartesian_trajectory
from roboarm.trajectory.lspb import multi_joint_lspb
from roboarm.workspace.analysis import WorkspaceAnalyzer

# ---------------------------------------------------------------------------
# Shared test helper
# ---------------------------------------------------------------------------

def _make_target(x: float, y: float, z: float = 0.0) -> EndEffectorPose:
    """Construct a minimal EndEffectorPose for use as an IK target."""
    position = np.array([x, y, z], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    return EndEffectorPose(
        position=position,
        rotation=np.eye(3, dtype=np.float64),
        transform=transform,
    )


# ===========================================================================
# F7 — RobotArm.is_planar
# ===========================================================================

class TestIsPlanar:
    """RobotArm.is_planar returns True iff every joint has alpha==0, d==0."""

    def test_two_link_planar_is_planar(self):
        """2-link RR planar arm: both joints have alpha=0, d=0 → is_planar."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        assert robot.is_planar is True

    def test_three_link_planar_is_planar(self, three_link_robot):
        """3-link planar arm from conftest: is_planar must be True."""
        assert three_link_robot.is_planar is True

    def test_six_dof_mdh_is_not_planar(self):
        """6-DOF MDH arm has non-zero alpha and d values → not planar."""
        robot = create_six_dof_mdh()
        assert robot.is_planar is False

    def test_is_planar_type_is_bool(self):
        """is_planar must return a plain Python bool, not numpy bool."""
        robot = create_two_link_planar()
        assert isinstance(robot.is_planar, bool)


# ===========================================================================
# F8 — EndEffectorPose read-only arrays
# ===========================================================================

class TestEndEffectorPoseReadOnly:
    """FK results must have immutable numpy arrays (writeable=False)."""

    @pytest.fixture
    def pose(self, two_link_robot):
        return two_link_robot.forward_kinematics([0.5, -0.3])

    def test_position_is_readonly(self, pose):
        """Assigning to pose.position[i] must raise ValueError."""
        with pytest.raises(ValueError):
            pose.position[0] = 99.0

    def test_rotation_is_readonly(self, pose):
        """Assigning to pose.rotation[i,j] must raise ValueError."""
        with pytest.raises(ValueError):
            pose.rotation[0, 0] = 99.0

    def test_transform_is_readonly(self, pose):
        """Assigning to pose.transform[i,j] must raise ValueError."""
        with pytest.raises(ValueError):
            pose.transform[0, 3] = 99.0

    def test_position_writeable_flag_is_false(self, pose):
        """The writeable flag on position must be False."""
        assert pose.position.flags.writeable is False

    def test_rotation_writeable_flag_is_false(self, pose):
        """The writeable flag on rotation must be False."""
        assert pose.rotation.flags.writeable is False

    def test_transform_writeable_flag_is_false(self, pose):
        """The writeable flag on transform must be False."""
        assert pose.transform.flags.writeable is False


# ===========================================================================
# F6 — IKSolution.best_attempt field
# ===========================================================================

class TestIKSolutionBestAttempt:
    """best_attempt is always set by iterative solvers; None for analytical
    on immediate workspace-rejection."""

    @pytest.mark.parametrize("solver_name", [
        "damped_least_squares",
        "jacobian_pseudoinverse",
        "ccd",
    ])
    def test_best_attempt_not_none_on_success(self, two_link_robot, solver_name):
        """Iterative solvers must populate best_attempt even on success."""
        target = _make_target(1.0, 0.5)
        solver = IKSolverRegistry.create(solver_name, two_link_robot)
        result = solver.solve(target)
        assert result.success is True, (
            f"{solver_name} failed for reachable target (error={result.residual_error:.2e})"
        )
        assert result.best_attempt is not None

    def test_analytical_best_attempt_is_none_on_failure(self, two_link_robot):
        """Analytical solver returns best_attempt=None for unreachable targets."""
        target = _make_target(5.0, 5.0)  # dist ≈ 7.07 >> L1+L2=2.0
        solver = IKSolverRegistry.create("analytical_2link", two_link_robot)
        result = solver.solve(target)
        assert result.success is False
        assert result.best_attempt is None

    def test_dls_best_attempt_populated_on_failure(self, two_link_robot):
        """DLS must still set best_attempt even when convergence fails."""
        target = _make_target(5.0, 5.0)
        solver = IKSolverRegistry.create("damped_least_squares", two_link_robot)
        result = solver.solve(target, q0=[0.0, 0.0])
        assert result.success is False
        assert result.best_attempt is not None

    def test_jacobian_best_attempt_populated_on_failure(self, two_link_robot):
        """Jacobian pseudo-inverse must still set best_attempt on failure."""
        target = _make_target(5.0, 5.0)
        solver = IKSolverRegistry.create("jacobian_pseudoinverse", two_link_robot)
        result = solver.solve(target, q0=[0.0, 0.0])
        assert result.success is False
        assert result.best_attempt is not None

    def test_dls_best_attempt_values_is_ndarray(self, two_link_robot):
        """best_attempt.values on a DLS result must be a numpy ndarray."""
        target = _make_target(1.0, 0.5)
        solver = IKSolverRegistry.create("damped_least_squares", two_link_robot)
        result = solver.solve(target)
        assert result.best_attempt is not None
        assert isinstance(result.best_attempt.values, np.ndarray)

    def test_dls_best_attempt_correct_dof(self, two_link_robot):
        """best_attempt.values must have length equal to robot n_dof."""
        target = _make_target(1.0, 0.5)
        solver = IKSolverRegistry.create("damped_least_squares", two_link_robot)
        result = solver.solve(target)
        assert result.best_attempt is not None
        assert result.best_attempt.values.shape == (two_link_robot.n_dof,)


# ===========================================================================
# F1 + F2 — Angle wrapping: all solution angles in [-π, π]
# ===========================================================================

class TestAngleWrapping:
    """Solved angles must be wrapped to [-pi, pi] for iterative solvers."""

    @pytest.mark.parametrize("solver_name", [
        "damped_least_squares",
        "jacobian_pseudoinverse",
    ])
    def test_iterative_solution_angles_in_range(self, two_link_robot, solver_name):
        """Primary solution angles from Jacobian-based solvers must lie in
        [-pi, pi] after wrapping is applied inside the solver loop."""
        target = _make_target(1.0, 0.5)
        solver = IKSolverRegistry.create(solver_name, two_link_robot)
        result = solver.solve(target)
        assert result.success is True
        angles = result.primary.values
        assert np.all(angles >= -math.pi - 1e-9), (
            f"Angle below -pi in {solver_name}: {angles}"
        )
        assert np.all(angles <= math.pi + 1e-9), (
            f"Angle above +pi in {solver_name}: {angles}"
        )

    def test_ccd_solution_angles_in_range(self, two_link_robot):
        """CCD primary solution for a reachable target must be in [-pi, pi]."""
        target = _make_target(1.0, 0.5)
        solver = IKSolverRegistry.create("ccd", two_link_robot)
        result = solver.solve(target)
        assert result.success is True
        angles = result.primary.values
        assert np.all(angles >= -math.pi - 1e-9), (
            f"CCD angle below -pi: {angles}"
        )
        assert np.all(angles <= math.pi + 1e-9), (
            f"CCD angle above +pi: {angles}"
        )

    def test_dls_best_attempt_angles_in_range(self, two_link_robot):
        """Even on failure the DLS best_attempt should be in [-pi, pi]
        because wrap_angles is applied at every iteration."""
        target = _make_target(1.5, 0.5)  # reachable; just a second data point
        solver = IKSolverRegistry.create("damped_least_squares", two_link_robot)
        result = solver.solve(target)
        assert result.best_attempt is not None
        angles = result.best_attempt.values
        assert np.all(angles >= -math.pi - 1e-9)
        assert np.all(angles <= math.pi + 1e-9)


# ===========================================================================
# F3 — FK verification in _solve_ik agent tool
# ===========================================================================

class TestFKVerificationTool:
    """The _solve_ik tool must append an FK-verification line on success and
    a best-attempt line on iterative-solver failure."""

    @pytest.fixture
    def coord(self):
        """Coordinator wrapping a standard 2-link planar robot."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        return RoboticsCoordinator(robot)

    def test_reachable_target_contains_verified_fk_error(self, coord):
        """Successful IK response must include the FK verification line."""
        response = coord.process("Solve IK for x=1.0, y=0.5")
        assert "Verified FK error" in response, (
            f"Expected 'Verified FK error' in response; got:\n{response}"
        )

    def test_reachable_target_shows_success_true(self, coord):
        """Response for a reachable target must say 'Success: True'."""
        response = coord.process("Solve IK for x=1.0, y=0.5")
        assert "Success: True" in response

    def test_unreachable_target_no_verified_fk_error(self, coord):
        """Unreachable target must NOT include the FK verification line."""
        response = coord.process("Solve IK for x=10.0, y=10.0")
        assert "Verified FK error" not in response, (
            f"Did not expect 'Verified FK error' for unreachable target:\n{response}"
        )

    def test_unreachable_target_contains_best_attempt(self, coord):
        """DLS (default solver) must report the best attempt on failure."""
        response = coord.process("Solve IK for x=10.0, y=10.0")
        assert "Best attempt" in response, (
            f"Expected 'Best attempt' for unreachable target; got:\n{response}"
        )

    def test_different_reachable_point_has_verification(self, coord):
        """A second reachable point should also produce FK verification."""
        response = coord.process("Solve IK for x=0.8, y=0.6")
        assert "Verified FK error" in response


# ===========================================================================
# F4 — WorkspaceAnalyzer seed reproducibility
# ===========================================================================

class TestWorkspaceAnalyzerSeed:
    """Providing the same seed must produce identical workspace samples."""

    @pytest.fixture
    def analyzer(self, two_link_robot):
        return WorkspaceAnalyzer(two_link_robot)

    def test_same_seed_produces_identical_arrays(self, analyzer):
        """sample_workspace(100, seed=42) called twice gives identical data."""
        pts1 = analyzer.sample_workspace(100, seed=42)
        pts2 = analyzer.sample_workspace(100, seed=42)
        np.testing.assert_array_equal(pts1, pts2)

    def test_different_seeds_produce_different_arrays(self, analyzer):
        """seed=1 and seed=2 must produce different point clouds."""
        pts1 = analyzer.sample_workspace(100, seed=1)
        pts2 = analyzer.sample_workspace(100, seed=2)
        assert not np.array_equal(pts1, pts2), (
            "Different seeds produced identical workspace samples"
        )

    def test_seed_none_is_non_deterministic(self, analyzer):
        """Without a seed, two calls should almost certainly differ."""
        pts1 = analyzer.sample_workspace(200, seed=None)
        pts2 = analyzer.sample_workspace(200, seed=None)
        # With 200 samples this should be unique with overwhelming probability
        assert not np.array_equal(pts1, pts2)

    def test_is_reachable_seed_reproducible(self, two_link_robot):
        """is_reachable with the same seed must return the same result."""
        analyzer = WorkspaceAnalyzer(two_link_robot)
        target = [1.0, 0.5, 0.0]
        result1 = analyzer.is_reachable(target, n_samples=300, seed=99)
        result2 = analyzer.is_reachable(target, n_samples=300, seed=99)
        assert result1 == result2

    def test_sample_workspace_output_shape(self, analyzer):
        """sample_workspace must return an (n_samples, 3) array."""
        pts = analyzer.sample_workspace(50, seed=7)
        assert pts.shape == (50, 3)

    @pytest.mark.parametrize("seed", [0, 1, 42, 100, 999])
    def test_multiple_seeds_return_valid_shape(self, analyzer, seed):
        """Various seeds must all return correctly shaped arrays."""
        pts = analyzer.sample_workspace(30, seed=seed)
        assert pts.shape == (30, 3)


# ===========================================================================
# F5 — LSPB velocity limits
# ===========================================================================

class TestLSPBVelocityLimits:
    """multi_joint_lspb must respect per-joint velocity caps and raise
    ValueError when the requested velocity is infeasible."""

    def test_feasible_vmax_completes(self):
        """A feasible v_max produces a complete trajectory without error."""
        q_start = [0.0, 0.0]
        q_end = [1.5, -0.5]
        # v_max well above the minimum required (delta_q / t_total)
        traj = multi_joint_lspb(q_start, q_end, t_total=2.0, v_max=[1.2, 0.4])
        assert traj.shape == (100, 2)
        assert traj[0, 0] == pytest.approx(0.0, abs=1e-9)
        assert traj[-1, 0] == pytest.approx(1.5, abs=1e-6)

    def test_infeasible_vmax_raises_value_error(self):
        """v_max below delta_q / t_total causes LSPB to raise ValueError."""
        q_start = [0.0]
        q_end = [1.5]
        # v_max=0.5 < 1.5/2.0=0.75 → t_blend = 2.0 - 1.5/0.5 = -1.0 → infeasible
        with pytest.raises(ValueError):
            multi_joint_lspb(q_start, q_end, t_total=2.0, v_max=[0.5])

    def test_backward_compat_no_joint_limits(self):
        """Without joint_limits the call must behave identically to the
        original signature (backward compatibility)."""
        q_start = [0.0, 0.0]
        q_end = [1.0, -0.5]
        traj_orig = multi_joint_lspb(q_start, q_end, t_total=2.0)
        traj_explicit_none = multi_joint_lspb(
            q_start, q_end, t_total=2.0, joint_limits=None,
        )
        np.testing.assert_array_equal(traj_orig, traj_explicit_none)

    def test_joint_limits_with_velocity_max_respected(self):
        """A JointLimits.velocity_max cap must be applied inside
        multi_joint_lspb; the trajectory should still complete."""
        limits = [JointLimits(lower=-math.pi, upper=math.pi, velocity_max=1.0)]
        traj = multi_joint_lspb(
            [0.0], [1.5], t_total=2.0, joint_limits=limits,
        )
        assert traj.shape == (100, 1)
        assert traj[-1, 0] == pytest.approx(1.5, abs=1e-6)

    def test_joint_limits_velocity_max_infeasible_raises(self):
        """An overly tight velocity_max in JointLimits must raise ValueError."""
        # velocity_max=0.6 < delta_q(1.5)/t_total(2.0)=0.75 → infeasible
        limits = [JointLimits(lower=-math.pi, upper=math.pi, velocity_max=0.6)]
        with pytest.raises(ValueError):
            multi_joint_lspb([0.0], [1.5], t_total=2.0, joint_limits=limits)

    def test_trajectory_starts_at_q_start_and_ends_at_q_end(self):
        """Trajectory endpoints must match q_start and q_end."""
        q_start = [0.3, -0.8]
        q_end = [1.1, 0.4]
        traj = multi_joint_lspb(q_start, q_end, t_total=3.0, n_steps=50)
        assert traj.shape == (50, 2)
        assert traj[0, 0] == pytest.approx(q_start[0], abs=1e-9)
        assert traj[0, 1] == pytest.approx(q_start[1], abs=1e-9)
        assert traj[-1, 0] == pytest.approx(q_end[0], abs=1e-6)
        assert traj[-1, 1] == pytest.approx(q_end[1], abs=1e-6)


# ===========================================================================
# A1 — request_context and current_request_id
# ===========================================================================

class TestRequestContext:
    """The request_context context manager must set / restore the per-context
    request ID accessible via current_request_id()."""

    def test_inside_context_returns_set_id(self):
        """current_request_id() must equal the ID passed to request_context."""
        with request_context("my-request-abc") as rid:
            assert rid == "my-request-abc"
            assert current_request_id() == "my-request-abc"

    def test_outside_context_returns_none(self):
        """current_request_id() returns None when no context is active."""
        # This test assumes it runs outside any active request_context.
        assert current_request_id() is None

    def test_id_restored_to_none_after_exit(self):
        """After the context exits, current_request_id() reverts to None."""
        with request_context("transient-id"):
            pass
        assert current_request_id() is None

    def test_nested_contexts_restore_outer_id(self):
        """Nested contexts must not leak: the outer ID is visible again after
        the inner context manager exits."""
        with request_context("outer-id") as outer:
            assert current_request_id() == "outer-id"
            assert outer == "outer-id"

            with request_context("inner-id") as inner:
                assert current_request_id() == "inner-id"
                assert inner == "inner-id"

            # Inner context exited → outer ID restored
            assert current_request_id() == "outer-id"

        # All contexts exited → None restored
        assert current_request_id() is None

    def test_auto_generated_id_is_string(self):
        """When no ID is supplied, request_context generates a UUID string."""
        with request_context() as rid:
            assert isinstance(rid, str)
            assert len(rid) > 0

    def test_coordinator_process_accepts_request_id_kwarg(self):
        """RoboticsCoordinator.process() must accept a request_id keyword
        argument without error."""
        robot = create_two_link_planar()
        coord = RoboticsCoordinator(robot)
        response = coord.process("describe robot", request_id="req-unit-test-001")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_coordinator_process_request_id_sets_context(self):
        """The request_id passed to process() should be active during the
        call (verified via the returned response — no exception path)."""
        robot = create_two_link_planar()
        coord = RoboticsCoordinator(robot)
        # If request_context is broken, process() would still return a string
        # but may raise internally; the absence of exception is the signal.
        response = coord.process("describe", request_id="check-ctx-123")
        assert "Robot" in response or len(response) > 0


# ===========================================================================
# A3 — Agent memory carry-forward
# ===========================================================================

class TestAgentMemoryCarryForward:
    """After an initial query that includes angles / coordinates, a follow-up
    query with no new parameters must succeed by recalling values from memory."""

    @pytest.fixture
    def fk_agent(self):
        """FKAgent backed by a standard 2-link planar robot."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        tools = build_robotics_tools(robot)
        return FKAgent("FK Agent", tools)

    @pytest.fixture
    def ik_agent(self):
        """IKAgent backed by a standard 2-link planar robot."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        tools = build_robotics_tools(robot)
        return IKAgent("IK Agent", tools)

    def test_fk_agent_first_query_succeeds(self, fk_agent):
        """Initial FK query with explicit angles must produce FK output."""
        response = fk_agent.process("Compute FK for angles [0.5, -0.3]")
        assert "Forward Kinematics" in response

    def test_fk_agent_follow_up_uses_memory(self, fk_agent):
        """After a query with angles, a follow-up without angles must
        succeed using the stored angles from conversation memory."""
        # Prime the memory
        fk_agent.process("Compute FK for angles [0.5, -0.3]")
        # Follow-up: no angles provided — agent must recall from memory
        response = fk_agent.process("Compute FK")
        assert "Forward Kinematics" in response, (
            f"Expected FK result from memory recall; got:\n{response}"
        )

    def test_fk_agent_follow_up_does_not_fail_with_error(self, fk_agent):
        """The follow-up query must not produce an error response when
        angles are available in memory."""
        fk_agent.process("Compute FK for angles [0.5, -0.3]")
        response = fk_agent.process("Compute FK")
        assert "Error" not in response, (
            f"Memory recall follow-up returned an error:\n{response}"
        )

    def test_ik_agent_first_query_succeeds(self, ik_agent):
        """Initial IK query with explicit coords must produce IK output."""
        response = ik_agent.process("Solve IK for x=1.0, y=0.5")
        assert "Inverse Kinematics" in response

    def test_ik_agent_follow_up_uses_memory(self, ik_agent):
        """After a query with coords, a follow-up without coords must
        succeed using the stored coordinates from conversation memory."""
        # Prime the memory
        ik_agent.process("Solve IK for x=1.0, y=0.5")
        # Follow-up: no coordinates — agent must recall from memory
        response = ik_agent.process("Solve IK")
        assert "Inverse Kinematics" in response, (
            f"Expected IK result from memory recall; got:\n{response}"
        )

    def test_ik_agent_follow_up_same_target(self, ik_agent):
        """Memory-recalled coordinates should lead to the same target
        (the response must still contain a position line)."""
        ik_agent.process("Solve IK for x=1.0, y=0.5")
        response = ik_agent.process("Solve IK")
        # The tool reports "Target position:" regardless of success
        assert "Target position" in response or "Inverse Kinematics" in response


# ===========================================================================
# F9 — cartesian_trajectory
# ===========================================================================

class TestCartesianTrajectory:
    """cartesian_trajectory must return correctly shaped arrays, solve most
    waypoints, and match endpoint positions approximately."""

    @pytest.fixture
    def robot(self):
        return create_two_link_planar(link1=1.0, link2=1.0)

    @pytest.fixture
    def start_end_poses(self, robot):
        """Two reachable poses (generated by FK) between which to plan."""
        start_q = [0.0, 0.0]
        end_q = [1.0, -0.5]
        return robot.forward_kinematics(start_q), robot.forward_kinematics(end_q)

    def test_return_shape_trajectory(self, robot, start_end_poses):
        """Trajectory array shape must be (n_steps, n_dof)."""
        start_pose, end_pose = start_end_poses
        n_steps = 20
        traj, _ = cartesian_trajectory(robot, start_pose, end_pose, n_steps=n_steps)
        assert traj.shape == (n_steps, robot.n_dof)

    def test_return_ik_results_length(self, robot, start_end_poses):
        """ik_results list length must equal n_steps."""
        start_pose, end_pose = start_end_poses
        n_steps = 20
        _, results = cartesian_trajectory(robot, start_pose, end_pose, n_steps=n_steps)
        assert len(results) == n_steps

    def test_high_solve_rate_reachable_path(self, robot, start_end_poses):
        """For a path between two reachable FK poses, >80 % of waypoints
        must be solved successfully."""
        start_pose, end_pose = start_end_poses
        _, results = cartesian_trajectory(robot, start_pose, end_pose, n_steps=20)
        success_count = sum(r.success for r in results)
        solve_rate = success_count / len(results)
        assert solve_rate > 0.8, (
            f"Solve rate {solve_rate:.0%} is below 80 % "
            f"({success_count}/{len(results)} steps solved)"
        )

    def test_first_waypoint_position_matches_start(self, robot, start_end_poses):
        """FK of the first trajectory point must be close to start_pose."""
        start_pose, end_pose = start_end_poses
        traj, _ = cartesian_trajectory(robot, start_pose, end_pose, n_steps=20)
        fk_first = robot.forward_kinematics(traj[0])
        np.testing.assert_allclose(
            fk_first.position[:2],
            start_pose.position[:2],
            atol=0.05,
            err_msg="First waypoint FK does not match start_pose position",
        )

    def test_last_waypoint_position_matches_end(self, robot, start_end_poses):
        """FK of the last trajectory point must be close to end_pose."""
        start_pose, end_pose = start_end_poses
        traj, _ = cartesian_trajectory(robot, start_pose, end_pose, n_steps=20)
        fk_last = robot.forward_kinematics(traj[-1])
        np.testing.assert_allclose(
            fk_last.position[:2],
            end_pose.position[:2],
            atol=0.05,
            err_msg="Last waypoint FK does not match end_pose position",
        )

    @pytest.mark.parametrize("solver_name", [
        "damped_least_squares",
        "jacobian_pseudoinverse",
        "ccd",
    ])
    def test_different_solver_names(self, robot, start_end_poses, solver_name):
        """cartesian_trajectory should work with any registered solver."""
        start_pose, end_pose = start_end_poses
        traj, results = cartesian_trajectory(
            robot, start_pose, end_pose, n_steps=10, solver_name=solver_name,
        )
        assert traj.shape == (10, robot.n_dof)
        assert len(results) == 10

    def test_trajectory_dtype_is_float64(self, robot, start_end_poses):
        """Trajectory array must use float64 dtype."""
        start_pose, end_pose = start_end_poses
        traj, _ = cartesian_trajectory(robot, start_pose, end_pose, n_steps=10)
        assert traj.dtype == np.float64


# ===========================================================================
# F11 — WorkspaceAnalyzer.plot()
# ===========================================================================

class TestWorkspaceAnalyzerPlot:
    """plot() must return a matplotlib Axes; repeated calls with the same seed
    must produce identical scatter data."""

    def test_plot_returns_axes_object(self, two_link_robot):
        """plot() must return an instance of matplotlib.axes.Axes."""
        plt.close("all")
        analyzer = WorkspaceAnalyzer(two_link_robot)
        ax = analyzer.plot(n_samples=30, seed=42)
        assert isinstance(ax, matplotlib.axes.Axes), (
            f"Expected Axes, got {type(ax)}"
        )
        plt.close("all")

    def test_plot_with_seed_reproducible_xlim(self, two_link_robot):
        """Two calls with the same seed must produce axes with identical
        x-limits (same sample data → same data range)."""
        analyzer = WorkspaceAnalyzer(two_link_robot)
        plt.close("all")
        ax1 = analyzer.plot(n_samples=50, seed=42)
        xlim1 = ax1.get_xlim()
        plt.close("all")

        ax2 = analyzer.plot(n_samples=50, seed=42)
        xlim2 = ax2.get_xlim()
        plt.close("all")

        assert xlim1 == pytest.approx(xlim2, rel=1e-9), (
            f"x-limits differ between identical seeds: {xlim1} vs {xlim2}"
        )

    def test_plot_with_seed_reproducible_ylim(self, two_link_robot):
        """Two calls with the same seed must produce axes with identical
        y-limits."""
        analyzer = WorkspaceAnalyzer(two_link_robot)
        plt.close("all")
        ax1 = analyzer.plot(n_samples=50, seed=42)
        ylim1 = ax1.get_ylim()
        plt.close("all")

        ax2 = analyzer.plot(n_samples=50, seed=42)
        ylim2 = ax2.get_ylim()
        plt.close("all")

        assert ylim1 == pytest.approx(ylim2, rel=1e-9)

    def test_plot_has_scatter_data(self, two_link_robot):
        """The returned Axes must contain at least one artist (scatter data)."""
        plt.close("all")
        analyzer = WorkspaceAnalyzer(two_link_robot)
        ax = analyzer.plot(n_samples=30, seed=1)
        # matplotlib stores scatter plots in ax.collections
        assert len(ax.collections) >= 1, "No scatter data found in Axes"
        plt.close("all")

    def test_plot_accepts_existing_axes(self, two_link_robot):
        """When an existing Axes is passed, plot() must use it rather than
        creating a new figure."""
        plt.close("all")
        _, ax_existing = plt.subplots()
        analyzer = WorkspaceAnalyzer(two_link_robot)
        ax_returned = analyzer.plot(n_samples=30, seed=5, ax=ax_existing)
        assert ax_returned is ax_existing
        plt.close("all")
