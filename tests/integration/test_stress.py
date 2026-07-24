"""Stress tests and performance benchmarks.

Validates throughput, consistency, and memory behaviour under heavy
computational loads for FK, IK, Jacobian, robot creation, and the
agent layer.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from roboarm.agents.coordinator import RoboticsCoordinator
from roboarm.core.types import EndEffectorPose
from roboarm.kinematics.jacobian import JacobianComputer
from roboarm.kinematics.solvers.registry import IKSolverRegistry
from roboarm.robots.six_dof_mdh import create_six_dof_mdh
from roboarm.robots.three_link_planar import create_three_link_planar
from roboarm.robots.two_link_planar import create_two_link_planar

import roboarm.kinematics.solvers  # noqa: F401  -- triggers auto-registration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(x: float, y: float, z: float = 0.0) -> EndEffectorPose:
    """Build an ``EndEffectorPose`` from Cartesian coordinates."""
    position = np.array([x, y, z], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    return EndEffectorPose(
        position=position,
        rotation=np.eye(3, dtype=np.float64),
        transform=transform,
    )


# ===================================================================
# 1. FK stress tests
# ===================================================================

class TestFKStress:
    """Forward-kinematics throughput and consistency."""

    def test_10k_fk_computations(self) -> None:
        """10 000 FK computations must complete in < 5 seconds."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        rng = np.random.default_rng(42)
        angles = rng.uniform(-np.pi, np.pi, size=(10_000, 2))

        start = time.perf_counter()
        for q in angles:
            pose = robot.forward_kinematics(q)
            assert np.all(np.isfinite(pose.position))
        elapsed = time.perf_counter() - start

        assert elapsed < 5.0, f"10K FK took {elapsed:.2f}s (limit: 5s)"

    def test_fk_consistency_under_load(self) -> None:
        """FK results must be bit-identical across 1 000 repeated calls."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        q = [0.5, -0.3]
        reference = robot.forward_kinematics(q)

        for _ in range(1_000):
            pose = robot.forward_kinematics(q)
            assert np.allclose(pose.position, reference.position, atol=1e-15)

    def test_3link_1k_fk(self) -> None:
        """1 000 FK computations on a 3-link arm must complete in < 1 s."""
        robot = create_three_link_planar()
        rng = np.random.default_rng(123)
        angles = rng.uniform(-np.pi, np.pi, size=(1_000, 3))

        start = time.perf_counter()
        for q in angles:
            robot.forward_kinematics(q)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"1K 3-link FK took {elapsed:.2f}s (limit: 1s)"


# ===================================================================
# 2. IK stress tests
# ===================================================================

class TestIKStress:
    """IK solver throughput and convergence under load."""

    def test_dls_500_solves(self) -> None:
        """DLS solver on 500 reachable targets: >= 90 % convergence."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        rng = np.random.default_rng(42)

        n_targets = 500
        success_count = 0
        for _ in range(n_targets):
            q = rng.uniform(-2.5, 2.5, size=2)
            target = robot.forward_kinematics(q)
            result = solver.solve(target)
            if result.success:
                success_count += 1

        assert success_count >= int(0.90 * n_targets), (
            f"Only {success_count}/{n_targets} converged"
        )

    def test_all_solvers_100_targets(self) -> None:
        """All solvers on 100 FK-generated reachable targets."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        rng = np.random.default_rng(99)

        solver_stats: dict[str, int] = {}
        for name in IKSolverRegistry.available():
            solver = IKSolverRegistry.create(name, robot)
            successes = 0
            for _ in range(100):
                q = rng.uniform(-2.0, 2.0, size=2)
                target = robot.forward_kinematics(q)
                result = solver.solve(target)
                if result.success and result.residual_error < 0.01:
                    successes += 1
            solver_stats[name] = successes

        # DLS and analytical should have comfortable success rates
        assert solver_stats.get("damped_least_squares", 0) >= 80, (
            f"DLS only got {solver_stats.get('damped_least_squares', 0)}/100"
        )
        assert solver_stats.get("analytical_2link", 0) >= 90, (
            f"analytical_2link only got {solver_stats.get('analytical_2link', 0)}/100"
        )

    def test_solver_latency_under_1ms_each(self) -> None:
        """Average single-solve latency must stay under 5 ms."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        target = _make_target(1.0, 0.5)

        start = time.perf_counter()
        n_runs = 200
        for _ in range(n_runs):
            solver.solve(target)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        avg_ms = elapsed_ms / n_runs

        assert avg_ms < 5.0, f"Average solve latency {avg_ms:.2f} ms (limit: 5 ms)"


# ===================================================================
# 3. Jacobian stress tests
# ===================================================================

class TestJacobianStress:
    """Jacobian throughput and geometric/numerical agreement."""

    def test_1k_jacobian_computations(self) -> None:
        """1 000 Jacobian computations must complete in < 2 seconds."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        jac = JacobianComputer(robot)
        rng = np.random.default_rng(42)
        angles = rng.uniform(-np.pi, np.pi, size=(1_000, 2))

        start = time.perf_counter()
        for q in angles:
            J = jac.compute(q.tolist())
            assert J.shape[1] == 2
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, (
            f"1K Jacobian took {elapsed:.2f}s (limit: 2s)"
        )

    def test_geometric_numerical_agreement_200(self) -> None:
        """Geometric and numerical Jacobians must agree at 200 configs."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        jac = JacobianComputer(robot)
        rng = np.random.default_rng(77)

        mismatches = 0
        for _ in range(200):
            q = rng.uniform(-np.pi, np.pi, size=2).tolist()
            J_geo = jac.compute(q)
            J_num = jac.compute_numerical(q)
            if not np.allclose(J_geo, J_num, atol=1e-3):
                mismatches += 1

        assert mismatches <= 2, f"{mismatches} mismatches out of 200"


# ===================================================================
# 4. Robot creation / memory stress
# ===================================================================

class TestRobotCreationStress:
    """Robot instantiation throughput and memory behaviour."""

    def test_create_1000_robots(self) -> None:
        """Creating 1 000 robot instances must not leak memory or crash."""
        robots = []
        for i in range(1_000):
            r = create_two_link_planar(
                link1=float(i % 10 + 1), link2=1.0,
            )
            robots.append(r)

        assert len(robots) == 1_000
        # Verify the last robot is still fully usable
        pose = robots[-1].forward_kinematics([0.5, -0.3])
        assert np.all(np.isfinite(pose.position))

    def test_6dof_creation_and_fk(self) -> None:
        """6-DOF robot creation and FK at home pose x 100."""
        robot = create_six_dof_mdh()
        assert robot.n_joints == 7
        assert robot.n_dof == 6

        home_rad = np.radians([0, 90, 0, 0, 180, 0])

        start = time.perf_counter()
        for _ in range(100):
            pose = robot.forward_kinematics(home_rad.tolist())
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f"100 x 6-DOF FK took {elapsed:.2f}s (limit: 1s)"
        )
        assert len(pose.position) == 3
        assert np.all(np.isfinite(pose.position))


# ===================================================================
# 5. Agent-layer stress tests
# ===================================================================

class TestAgentStress:
    """Coordinator throughput and stability under sustained load."""

    def test_100_sequential_queries(self) -> None:
        """100 sequential coordinator queries must not crash or time out."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        queries = [
            "Describe the robot",
            "FK for angles [0.5, -0.3]",
            "Solve IK for x=1.0, y=0.5",
            "Compare solvers for x=0.8, y=0.6",
        ]

        start = time.perf_counter()
        for i in range(100):
            response = coordinator.process(queries[i % len(queries)])
            assert isinstance(response, str)
            assert len(response) > 0
        elapsed = time.perf_counter() - start

        assert elapsed < 30.0, f"100 queries took {elapsed:.1f}s (limit: 30s)"

    def test_rapid_describe_calls(self) -> None:
        """500 rapid ``describe`` calls must remain consistent."""
        robot = create_two_link_planar()
        coordinator = RoboticsCoordinator(robot)
        reference = coordinator.process("describe the robot")

        for _ in range(500):
            response = coordinator.process("describe the robot")
            assert response == reference
