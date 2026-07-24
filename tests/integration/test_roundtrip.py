"""End-to-end FK <-> IK roundtrip accuracy tests.

Validates that forward kinematics followed by inverse kinematics and then
forward kinematics again recovers the original end-effector position to
within tight tolerances.  Accuracy is the single most critical metric in
robotic arm control, so every test in this module enforces sub-millimetre
position error bounds.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

import roboarm.kinematics.solvers  # noqa: F401 – triggers auto-registration
from roboarm.core.transform import is_valid_transform
from roboarm.core.types import EndEffectorPose
from roboarm.kinematics.solvers.registry import IKSolverRegistry
from roboarm.robots.three_link_planar import create_three_link_planar
from roboarm.robots.two_link_planar import create_two_link_planar


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


# ======================================================================
# 2-Link planar roundtrip grid tests
# ======================================================================


class TestTwoLinkRoundtripGrid:
    """Systematic FK -> IK -> FK roundtrip on a grid of 2-link configs."""

    _Q_VALS: list[float] = [-2.0, -1.0, 0.0, 1.0, 2.0]

    def _run_grid_roundtrip(
        self,
        link1: float,
        link2: float,
        *,
        tol: float = 1e-3,
    ) -> None:
        """Generate 25 FK targets on a q-grid, solve IK, verify accuracy."""
        robot = create_two_link_planar(link1=link1, link2=link2)
        solver = IKSolverRegistry.create("damped_least_squares", robot)

        successes = 0
        for q1, q2 in itertools.product(self._Q_VALS, repeat=2):
            target = robot.forward_kinematics([q1, q2])
            result = solver.solve(target)
            if not result.success:
                continue
            successes += 1
            recovered = robot.forward_kinematics(result.primary.values)
            error = np.linalg.norm(
                recovered.position[:2] - target.position[:2],
            )
            assert error < tol, (
                f"Roundtrip error {error:.8f} at q=[{q1}, {q2}] "
                f"for L1={link1}, L2={link2}"
            )

        # At least 80 % of targets must converge
        assert successes >= 20, (
            f"Only {successes}/25 converged for L1={link1}, L2={link2}"
        )

    def test_roundtrip_grid_equal_links(self) -> None:
        """Grid of 25 targets for equal-link arm (L1=L2=1.0)."""
        self._run_grid_roundtrip(1.0, 1.0)

    def test_roundtrip_grid_unequal_links(self) -> None:
        """Grid of 25 targets for unequal-link arm (L1=1.0, L2=0.5)."""
        self._run_grid_roundtrip(1.0, 0.5)

    def test_roundtrip_grid_long_short(self) -> None:
        """L1=2.0, L2=0.3 -- high ratio stresses accuracy at extremes."""
        self._run_grid_roundtrip(2.0, 0.3)

    def test_roundtrip_at_near_boundary_targets(self) -> None:
        """10 targets near workspace boundary (90-99 % of max reach)."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        max_reach = 2.0  # L1 + L2

        rng = np.random.default_rng(seed=7)
        for _ in range(10):
            fraction = rng.uniform(0.90, 0.99)
            angle = rng.uniform(-math.pi, math.pi)
            r = max_reach * fraction
            target = _make_target(r * math.cos(angle), r * math.sin(angle))
            result = solver.solve(target)
            assert result.success, (
                f"IK failed near boundary at fraction={fraction:.2f}"
            )
            recovered = robot.forward_kinematics(result.primary.values)
            error = np.linalg.norm(
                recovered.position[:2] - target.position[:2],
            )
            assert error < 1e-3, (
                f"Boundary error {error:.8f} at fraction={fraction:.2f}"
            )

    def test_roundtrip_at_inner_workspace(self) -> None:
        """10 targets at 10-50 % of max reach (inner workspace)."""
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        max_reach = 2.0

        rng = np.random.default_rng(seed=11)
        for _ in range(10):
            fraction = rng.uniform(0.10, 0.50)
            angle = rng.uniform(-math.pi, math.pi)
            r = max_reach * fraction
            target = _make_target(r * math.cos(angle), r * math.sin(angle))
            result = solver.solve(target)
            assert result.success, (
                f"IK failed at inner workspace fraction={fraction:.2f}"
            )
            recovered = robot.forward_kinematics(result.primary.values)
            error = np.linalg.norm(
                recovered.position[:2] - target.position[:2],
            )
            assert error < 1e-3, (
                f"Inner workspace error {error:.8f} at "
                f"fraction={fraction:.2f}"
            )


# ======================================================================
# 3-Link planar roundtrip tests
# ======================================================================


class TestThreeLinkRoundtripGrid:
    """FK -> IK -> FK roundtrip for redundant 3-link arm."""

    def test_roundtrip_grid_3link(self) -> None:
        """20 targets for 3-link arm; IK may find a different q but the
        recovered position must match the original target."""
        robot = create_three_link_planar(link1=1.0, link2=1.0, link3=0.5)
        solver = IKSolverRegistry.create("damped_least_squares", robot)

        rng = np.random.default_rng(seed=99)
        successes = 0
        for _ in range(20):
            q = rng.uniform(-2.0, 2.0, size=3)
            target = robot.forward_kinematics(q.tolist())
            result = solver.solve(target)
            if not result.success:
                continue
            successes += 1
            recovered = robot.forward_kinematics(result.primary.values)
            error = np.linalg.norm(
                recovered.position[:2] - target.position[:2],
            )
            assert error < 1e-3, f"3-link roundtrip error {error:.8f}"

        assert successes >= 15, f"Only {successes}/20 converged"

    def test_redundancy_multiple_solutions(self) -> None:
        """Same target reached by different joint configs in 3-link arm.

        Solving from several initial guesses should yield solutions that
        may differ in joint space but converge to the same Cartesian
        position.
        """
        robot = create_three_link_planar()
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        target = _make_target(1.2, 0.8)

        solutions = []
        rng = np.random.default_rng(seed=55)
        for _ in range(8):
            q0 = rng.uniform(-2.0, 2.0, size=3).tolist()
            result = solver.solve(target, q0=q0)
            if result.success:
                pos = robot.forward_kinematics(
                    result.primary.values,
                ).position[:2]
                solutions.append(
                    (result.primary.values.copy(), pos.copy()),
                )

        assert len(solutions) >= 3, "Need at least 3 solutions"

        # All recovered positions must be close to each other
        positions = np.array([s[1] for s in solutions])
        mean_pos = positions.mean(axis=0)
        for pos in positions:
            assert np.linalg.norm(pos - mean_pos) < 1e-3, (
                "Redundant solutions disagree on position"
            )


# ======================================================================
# Parametrised multi-geometry roundtrip
# ======================================================================


class TestMultipleArmConfigurations:
    """Test accuracy across many different arm geometries."""

    @pytest.mark.parametrize(
        "l1,l2",
        [
            (0.5, 0.5),
            (1.0, 1.0),
            (2.0, 2.0),
            (1.0, 0.5),
            (0.5, 1.0),
            (1.0, 0.3),
            (0.3, 1.0),
            (1.5, 0.8),
            (0.8, 1.5),
            (3.0, 1.0),
        ],
    )
    def test_fk_ik_roundtrip_varied_geometry(
        self, l1: float, l2: float,
    ) -> None:
        """FK -> IK -> FK with DLS for 10 different arm geometries."""
        robot = create_two_link_planar(link1=l1, link2=l2)
        solver = IKSolverRegistry.create("damped_least_squares", robot)

        q_test = [0.7, -0.4]
        target = robot.forward_kinematics(q_test)
        result = solver.solve(target)
        assert result.success, f"IK failed for L1={l1}, L2={l2}"

        recovered = robot.forward_kinematics(result.primary.values)
        error = np.linalg.norm(
            recovered.position[:2] - target.position[:2],
        )
        assert error < 1e-3, (
            f"Roundtrip error {error:.8f} for L1={l1}, L2={l2}"
        )

    @pytest.mark.parametrize(
        "l1,l2",
        [
            (0.5, 0.5),
            (1.0, 1.0),
            (2.0, 2.0),
            (1.0, 0.5),
            (0.5, 1.0),
            (1.0, 0.3),
            (0.3, 1.0),
            (1.5, 0.8),
            (0.8, 1.5),
            (3.0, 1.0),
        ],
    )
    def test_fk_transform_validity_varied_geometry(
        self, l1: float, l2: float,
    ) -> None:
        """FK transforms must be valid SE(3) for all arm geometries."""
        robot = create_two_link_planar(link1=l1, link2=l2)
        rng = np.random.default_rng(seed=42)
        for _ in range(5):
            q = rng.uniform(-2.5, 2.5, size=2).tolist()
            pose = robot.forward_kinematics(q)
            assert is_valid_transform(pose.transform), (
                f"Invalid SE(3) at q={q} for L1={l1}, L2={l2}"
            )


# ======================================================================
# Analytical vs numerical solver comparison
# ======================================================================


class TestAnalyticalVsNumerical:
    """Compare analytical IK solutions against numerical solvers."""

    def test_analytical_matches_dls_20_targets(self) -> None:
        """Both analytical and DLS must find equivalent positions."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        analytical = IKSolverRegistry.create("analytical_2link", robot)
        dls = IKSolverRegistry.create("damped_least_squares", robot)

        rng = np.random.default_rng(seed=42)
        both_ok = 0
        for _ in range(20):
            q = rng.uniform(-2.0, 2.0, size=2)
            target = robot.forward_kinematics(q.tolist())

            a_result = analytical.solve(target)
            d_result = dls.solve(target)

            if a_result.success and d_result.success:
                both_ok += 1
                a_pos = robot.forward_kinematics(
                    a_result.primary.values,
                ).position[:2]
                d_pos = robot.forward_kinematics(
                    d_result.primary.values,
                ).position[:2]
                assert np.allclose(a_pos, target.position[:2], atol=1e-3), (
                    f"Analytical position error: "
                    f"{np.linalg.norm(a_pos - target.position[:2]):.8f}"
                )
                assert np.allclose(d_pos, target.position[:2], atol=1e-3), (
                    f"DLS position error: "
                    f"{np.linalg.norm(d_pos - target.position[:2]):.8f}"
                )

        assert both_ok >= 15, (
            f"Only {both_ok}/20 targets solved by both solvers"
        )

    def test_analytical_elbow_alternatives(self) -> None:
        """Analytical solver should produce elbow-up and elbow-down
        alternatives that both reach the target."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        analytical = IKSolverRegistry.create("analytical_2link", robot)

        target = _make_target(1.0, 0.5)
        result = analytical.solve(target)
        assert result.success
        assert len(result.alternatives) >= 1, "Expected at least one alternative"

        # Both primary and alternative must reach the target
        for sol in [result.primary] + result.alternatives:
            pos = robot.forward_kinematics(sol.values).position[:2]
            error = np.linalg.norm(pos - target.position[:2])
            assert error < 1e-6, (
                f"Elbow solution error {error:.8f}"
            )


# ======================================================================
# Position accuracy benchmark
# ======================================================================


class TestPositionAccuracyBenchmark:
    """Position accuracy benchmark -- the most critical robotics metric."""

    def test_sub_millimeter_accuracy(self) -> None:
        """For a 1-metre arm, IK must achieve sub-millimetre accuracy."""
        robot = create_two_link_planar(link1=0.5, link2=0.5)
        solver = IKSolverRegistry.create("damped_least_squares", robot)

        errors: list[float] = []
        rng = np.random.default_rng(seed=123)
        for _ in range(50):
            q = rng.uniform(-2.5, 2.5, size=2)
            target = robot.forward_kinematics(q.tolist())
            result = solver.solve(target)
            if result.success:
                recovered = robot.forward_kinematics(result.primary.values)
                error = float(np.linalg.norm(
                    recovered.position[:2] - target.position[:2],
                ))
                errors.append(error)

        assert len(errors) >= 40, (
            f"Too few convergences: {len(errors)}/50"
        )
        max_error = max(errors)
        mean_error = sum(errors) / len(errors)
        assert max_error < 1e-3, (
            f"Max error {max_error:.8f} exceeds 1 mm"
        )
        assert mean_error < 1e-5, (
            f"Mean error {mean_error:.8f} exceeds 0.01 mm"
        )

    def test_repeatability(self) -> None:
        """Same target solved 20 times must give consistent results."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        target = _make_target(1.0, 0.5)

        positions: list[np.ndarray] = []
        for _ in range(20):
            result = solver.solve(target)
            if result.success:
                pos = robot.forward_kinematics(
                    result.primary.values,
                ).position[:2]
                positions.append(pos)

        assert len(positions) >= 18, (
            f"Too few solutions: {len(positions)}/20"
        )
        pos_array = np.array(positions)
        std = np.std(pos_array, axis=0)
        assert np.all(std < 1e-6), f"Repeatability std: {std}"

    def test_accuracy_across_workspace_quadrants(self) -> None:
        """Verify consistent accuracy in all four XY quadrants.

        Each target is generated via FK from known joint angles pointing
        into the corresponding quadrant, guaranteeing reachability and
        providing a good initial guess for the iterative solver.
        """
        robot = create_two_link_planar(link1=1.0, link2=1.0)
        solver = IKSolverRegistry.create("damped_least_squares", robot)

        # Joint-angle seeds that place the end-effector in each quadrant
        quadrant_seeds = [
            [0.3, -0.5],          # Q1 (x > 0, y > 0)
            [math.pi - 0.3, 0.5], # Q2 (x < 0, y > 0)
            [-math.pi + 0.3, -0.5], # Q3 (x < 0, y < 0)
            [-0.3, 0.5],          # Q4 (x > 0, y < 0)
        ]
        for q_seed in quadrant_seeds:
            target = robot.forward_kinematics(q_seed)
            result = solver.solve(target, q0=q_seed)
            assert result.success, (
                f"Failed in quadrant for q_seed={q_seed}"
            )
            recovered = robot.forward_kinematics(result.primary.values)
            error = np.linalg.norm(
                recovered.position[:2] - target.position[:2],
            )
            assert error < 1e-3, (
                f"Quadrant error {error:.8f} for q_seed={q_seed}"
            )

    def test_accuracy_statistical_summary(self) -> None:
        """100-sample statistical summary: median and 95th percentile."""
        robot = create_two_link_planar(link1=1.0, link2=0.7)
        solver = IKSolverRegistry.create("damped_least_squares", robot)

        errors: list[float] = []
        rng = np.random.default_rng(seed=200)
        for _ in range(100):
            q = rng.uniform(-2.0, 2.0, size=2)
            target = robot.forward_kinematics(q.tolist())
            result = solver.solve(target)
            if result.success:
                recovered = robot.forward_kinematics(result.primary.values)
                err = float(np.linalg.norm(
                    recovered.position[:2] - target.position[:2],
                ))
                errors.append(err)

        assert len(errors) >= 80, (
            f"Too few convergences: {len(errors)}/100"
        )
        err_arr = np.array(errors)
        median_err = float(np.median(err_arr))
        p95_err = float(np.percentile(err_arr, 95))

        assert median_err < 1e-6, (
            f"Median error {median_err:.8e} too large"
        )
        assert p95_err < 1e-4, (
            f"95th-percentile error {p95_err:.8e} too large"
        )
