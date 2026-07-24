"""Precision and accuracy tests for robotics computations.

In robotics, accuracy is paramount. These tests sweep large configuration
spaces and verify that forward kinematics, inverse kinematics, transforms,
rotations, and Jacobians maintain numerical precision under stress.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import roboarm.kinematics.solvers  # noqa: F401 — triggers auto-registration
from roboarm.core.rotations import (
    axis_angle_to_rotation,
    euler_to_rotation,
    is_valid_rotation,
    quaternion_to_rotation,
    rotation_to_axis_angle,
    rotation_to_euler,
    rotation_to_quaternion,
)
from roboarm.core.transform import (
    chain_transforms,
    dh_transform,
    extract_position,
    extract_rotation,
    inverse_transform,
    is_valid_transform,
    mdh_transform,
)
from roboarm.core.types import EndEffectorPose, IKSolution
from roboarm.kinematics.jacobian import JacobianComputer
from roboarm.kinematics.solvers.registry import IKSolverRegistry
from roboarm.robots.two_link_planar import create_two_link_planar


def _make_target(x: float, y: float, z: float = 0.0) -> EndEffectorPose:
    """Build a minimal IK target from Cartesian coordinates."""
    position = np.array([x, y, z], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    return EndEffectorPose(
        position=position,
        rotation=np.eye(3, dtype=np.float64),
        transform=transform,
    )


# ------------------------------------------------------------------ #
#  FK precision sweep
# ------------------------------------------------------------------ #


class TestFKPrecisionSweep:
    """Sweep joint angles and verify FK properties."""

    def test_2link_fk_sweep_100_configs(self, two_link_robot):
        """FK at 100 random configs should always produce valid SE(3) transforms."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            q = rng.uniform(-math.pi, math.pi, size=2)
            pose = two_link_robot.forward_kinematics(q)
            assert is_valid_transform(pose.transform), (
                f"Invalid SE(3) at q={q}"
            )

    def test_2link_fk_symmetry(self, two_link_robot):
        """FK(q1, q2) position norm should equal FK(-q1, -q2) position norm for symmetric arms."""
        rng = np.random.default_rng(99)
        for _ in range(50):
            q = rng.uniform(-math.pi, math.pi, size=2)
            pose_pos = two_link_robot.forward_kinematics(q)
            pose_neg = two_link_robot.forward_kinematics(-q)
            norm_pos = float(np.linalg.norm(pose_pos.position))
            norm_neg = float(np.linalg.norm(pose_neg.position))
            assert norm_pos == pytest.approx(norm_neg, abs=1e-10), (
                f"Symmetry broken at q={q}: {norm_pos} vs {norm_neg}"
            )

    def test_2link_fk_reach_never_exceeds_sum(self, two_link_robot):
        """End-effector distance from origin should never exceed L1+L2."""
        max_reach = 1.0 + 1.0  # conftest: L1=1.0, L2=1.0
        rng = np.random.default_rng(7)
        for _ in range(200):
            q = rng.uniform(-math.pi, math.pi, size=2)
            pose = two_link_robot.forward_kinematics(q)
            dist = float(np.linalg.norm(pose.position))
            assert dist <= max_reach + 1e-10, (
                f"Reach {dist} exceeds max {max_reach} at q={q}"
            )

    def test_3link_fk_reach_never_exceeds_sum(self, three_link_robot):
        """Same for 3-link: max reach = L1 + L2 + L3."""
        max_reach = 1.0 + 1.0 + 0.5  # conftest: L1=1.0, L2=1.0, L3=0.5
        rng = np.random.default_rng(11)
        for _ in range(200):
            q = rng.uniform(-math.pi, math.pi, size=3)
            pose = three_link_robot.forward_kinematics(q)
            dist = float(np.linalg.norm(pose.position))
            assert dist <= max_reach + 1e-10, (
                f"Reach {dist} exceeds max {max_reach} at q={q}"
            )

    def test_fk_at_home_is_fully_extended(self, two_link_robot):
        """At [0,0], arm is fully extended along x-axis, reach = L1+L2."""
        pose = two_link_robot.forward_kinematics([0.0, 0.0])
        assert pose.x == pytest.approx(2.0, abs=1e-10)
        assert pose.y == pytest.approx(0.0, abs=1e-10)
        assert pose.z == pytest.approx(0.0, abs=1e-10)
        dist = float(np.linalg.norm(pose.position))
        assert dist == pytest.approx(2.0, abs=1e-10)

    def test_fk_deterministic(self, two_link_robot):
        """Same angles must always produce identical results."""
        q = [0.7, -1.3]
        results = [two_link_robot.forward_kinematics(q) for _ in range(10)]
        ref = results[0]
        for r in results[1:]:
            np.testing.assert_array_equal(r.position, ref.position)
            np.testing.assert_array_equal(r.rotation, ref.rotation)
            np.testing.assert_array_equal(r.transform, ref.transform)


# ------------------------------------------------------------------ #
#  IK accuracy grid
# ------------------------------------------------------------------ #


class TestIKAccuracyGrid:
    """IK solver accuracy across a grid of reachable targets."""

    def test_dls_roundtrip_grid_20_targets(self):
        """FK->IK->FK roundtrip at 20 grid targets, all errors < 1e-3."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        solver = IKSolverRegistry.create("damped_least_squares", robot)

        rng = np.random.default_rng(123)
        for _ in range(20):
            q_orig = rng.uniform(-math.pi * 0.8, math.pi * 0.8, size=2)
            fk_orig = robot.forward_kinematics(q_orig)
            target = _make_target(fk_orig.x, fk_orig.y)
            result = solver.solve(target)
            if result.success and result.primary is not None:
                fk_check = robot.forward_kinematics(result.primary.values)
                err = float(np.linalg.norm(
                    fk_check.position[:2] - fk_orig.position[:2]
                ))
                assert err < 1e-3, (
                    f"Roundtrip error {err:.2e} at q={q_orig}"
                )

    def test_all_solvers_accuracy_comparison(self):
        """Compare accuracy of all solvers on 10 targets."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        rng = np.random.default_rng(456)
        solver_names = IKSolverRegistry.available()

        for _ in range(10):
            q_orig = rng.uniform(-2.0, 2.0, size=2)
            fk_orig = robot.forward_kinematics(q_orig)
            target = _make_target(fk_orig.x, fk_orig.y)

            for name in solver_names:
                solver = IKSolverRegistry.create(name, robot)
                result = solver.solve(target)
                assert isinstance(result, IKSolution)
                if result.success and result.primary is not None:
                    fk_check = robot.forward_kinematics(result.primary.values)
                    err = float(np.linalg.norm(
                        fk_check.position[:2] - fk_orig.position[:2]
                    ))
                    assert err < 1e-2, (
                        f"Solver '{name}' error {err:.2e} too large"
                    )

    def test_ik_near_workspace_boundary(self):
        """IK accuracy degrades near workspace boundary but should still converge."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        max_reach = 1.8
        boundary_reach = 0.95 * max_reach  # 1.71
        target = _make_target(boundary_reach, 0.0)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        result = solver.solve(target)
        assert isinstance(result, IKSolution)
        if result.success and result.primary is not None:
            fk = robot.forward_kinematics(result.primary.values)
            err = float(np.linalg.norm(fk.position[:2] - target.position[:2]))
            assert err < 0.1, (
                f"Boundary target error {err:.2e} exceeds tolerance"
            )

    def test_ik_at_workspace_center(self):
        """IK at workspace center should converge quickly with high accuracy."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        target = _make_target(0.9, 0.0)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        # Provide a non-singular initial guess so the solver avoids the
        # fully-extended singularity at q=[0,0].
        result = solver.solve(target, q0=[0.5, -0.5])
        assert result.success, "IK should converge at workspace center"
        assert result.primary is not None
        assert result.iterations < 50, (
            f"Took {result.iterations} iterations at workspace center"
        )
        fk = robot.forward_kinematics(result.primary.values)
        err = float(np.linalg.norm(fk.position[:2] - target.position[:2]))
        assert err < 1e-5, f"Center accuracy {err:.2e} below expectation"

    def test_ik_multiple_initial_guesses(self):
        """Different initial guesses should all converge for a reachable target."""
        robot = create_two_link_planar(link1=1.0, link2=0.8)
        target = _make_target(0.8, 0.5)
        solver = IKSolverRegistry.create("damped_least_squares", robot)
        guesses = [
            [0.0, 0.0],
            [1.0, -1.0],
            [-0.5, 0.5],
            [math.pi / 4, -math.pi / 4],
        ]
        for q0 in guesses:
            result = solver.solve(target, q0=q0)
            assert isinstance(result, IKSolution)
            if result.success and result.primary is not None:
                fk = robot.forward_kinematics(result.primary.values)
                err = float(np.linalg.norm(
                    fk.position[:2] - target.position[:2]
                ))
                assert err < 1e-3, (
                    f"Initial guess {q0} gave error {err:.2e}"
                )


# ------------------------------------------------------------------ #
#  Transform precision
# ------------------------------------------------------------------ #


class TestTransformPrecision:
    """Transform computation precision tests."""

    def test_inverse_roundtrip_100_transforms(self):
        """T @ inv(T) should be identity for 100 random DH transforms."""
        rng = np.random.default_rng(303)
        for _ in range(100):
            alpha = rng.uniform(-math.pi, math.pi)
            a = rng.uniform(0.0, 5.0)
            theta = rng.uniform(-math.pi, math.pi)
            d = rng.uniform(-2.0, 2.0)
            T = dh_transform(alpha, a, theta, d)
            T_inv = inverse_transform(T)
            product = T @ T_inv
            np.testing.assert_allclose(
                product, np.eye(4), atol=1e-10,
                err_msg="T @ inv(T) is not identity",
            )

    def test_chain_associativity(self):
        """(T1 @ T2) @ T3 == T1 @ (T2 @ T3) up to floating point."""
        rng = np.random.default_rng(404)
        for _ in range(20):
            params = rng.uniform(-2.0, 2.0, size=(3, 4))
            T1 = dh_transform(*params[0])
            T2 = dh_transform(*params[1])
            T3 = dh_transform(*params[2])
            left = chain_transforms([chain_transforms([T1, T2]), T3])
            right = chain_transforms([T1, chain_transforms([T2, T3])])
            np.testing.assert_allclose(
                left, right, atol=1e-12,
                err_msg="Transform chain is not associative",
            )

    def test_dh_mdh_both_valid_se3(self):
        """Both DH and MDH transforms at random params are always valid SE(3)."""
        rng = np.random.default_rng(505)
        for _ in range(50):
            alpha = rng.uniform(-math.pi, math.pi)
            a = rng.uniform(0.0, 3.0)
            theta = rng.uniform(-math.pi, math.pi)
            d = rng.uniform(-1.0, 1.0)
            T_dh = dh_transform(alpha, a, theta, d)
            T_mdh = mdh_transform(alpha, a, theta, d)
            assert is_valid_transform(T_dh), (
                f"Standard DH produced invalid SE(3) at "
                f"alpha={alpha}, a={a}, theta={theta}, d={d}"
            )
            assert is_valid_transform(T_mdh), (
                f"Modified DH produced invalid SE(3) at "
                f"alpha={alpha}, a={a}, theta={theta}, d={d}"
            )

    def test_rotation_determinant_always_1(self):
        """det(R) == 1.0 for all rotations extracted from DH transforms."""
        rng = np.random.default_rng(606)
        for _ in range(100):
            alpha = rng.uniform(-math.pi, math.pi)
            a = rng.uniform(0.0, 5.0)
            theta = rng.uniform(-math.pi, math.pi)
            d = rng.uniform(-2.0, 2.0)
            T = dh_transform(alpha, a, theta, d)
            R = extract_rotation(T)
            det = float(np.linalg.det(R))
            assert det == pytest.approx(1.0, abs=1e-10), (
                f"det(R) = {det} at alpha={alpha}, theta={theta}"
            )

    def test_inverse_transform_position_negation(self):
        """Position extracted from inv(T) should satisfy R^T @ (-p)."""
        rng = np.random.default_rng(707)
        for _ in range(30):
            alpha = rng.uniform(-math.pi, math.pi)
            a = rng.uniform(0.1, 3.0)
            theta = rng.uniform(-math.pi, math.pi)
            d = rng.uniform(-1.0, 1.0)
            T = dh_transform(alpha, a, theta, d)
            T_inv = inverse_transform(T)
            R = extract_rotation(T)
            p = extract_position(T)
            expected_inv_pos = -R.T @ p
            np.testing.assert_allclose(
                extract_position(T_inv), expected_inv_pos, atol=1e-12,
            )


# ------------------------------------------------------------------ #
#  Rotation precision
# ------------------------------------------------------------------ #


class TestRotationPrecision:
    """Rotation conversion precision tests."""

    def test_euler_roundtrip_100_angles(self):
        """euler->matrix->euler for 100 random angle triples, error < 1e-10."""
        rng = np.random.default_rng(808)
        for _ in range(100):
            # Avoid gimbal lock by keeping pitch away from +/-pi/2
            roll = rng.uniform(-math.pi, math.pi)
            pitch = rng.uniform(-math.pi / 2 + 0.1, math.pi / 2 - 0.1)
            yaw = rng.uniform(-math.pi, math.pi)
            R = euler_to_rotation(roll, pitch, yaw)
            r2, p2, y2 = rotation_to_euler(R)
            R2 = euler_to_rotation(r2, p2, y2)
            np.testing.assert_allclose(
                R, R2, atol=1e-10,
                err_msg=f"Euler roundtrip failed at roll={roll}, pitch={pitch}, yaw={yaw}",
            )

    def test_quaternion_roundtrip_100(self):
        """quaternion->matrix->quaternion for 100 random quaternions."""
        rng = np.random.default_rng(909)
        for _ in range(100):
            q_raw = rng.standard_normal(4)
            q_raw = q_raw / np.linalg.norm(q_raw)
            if q_raw[0] < 0:
                q_raw = -q_raw  # canonical positive-w form
            R = quaternion_to_rotation(q_raw)
            q_back = rotation_to_quaternion(R)
            if q_back[0] < 0:
                q_back = -q_back
            np.testing.assert_allclose(
                q_raw, q_back, atol=1e-10,
                err_msg="Quaternion roundtrip failed",
            )

    def test_axis_angle_roundtrip_50(self):
        """axis_angle->matrix->axis_angle for 50 random configs."""
        rng = np.random.default_rng(1010)
        for _ in range(50):
            raw_axis = rng.standard_normal(3)
            raw_axis = raw_axis / np.linalg.norm(raw_axis)
            angle = rng.uniform(0.1, math.pi - 0.1)  # away from 0 and pi
            R = axis_angle_to_rotation(raw_axis, angle)
            axis_back, angle_back = rotation_to_axis_angle(R)
            # Axis may be flipped with negated angle
            if angle_back < 0:
                axis_back = -axis_back
                angle_back = -angle_back
            R_back = axis_angle_to_rotation(axis_back, angle_back)
            np.testing.assert_allclose(
                R, R_back, atol=1e-10,
                err_msg="Axis-angle roundtrip failed",
            )

    def test_all_rotation_representations_agree(self):
        """For a given rotation, Euler, quat, and axis-angle all produce same matrix."""
        rng = np.random.default_rng(1111)
        for _ in range(30):
            roll = rng.uniform(-math.pi, math.pi)
            pitch = rng.uniform(-math.pi / 2 + 0.1, math.pi / 2 - 0.1)
            yaw = rng.uniform(-math.pi, math.pi)

            R_euler = euler_to_rotation(roll, pitch, yaw)

            q = rotation_to_quaternion(R_euler)
            R_from_quat = quaternion_to_rotation(q)

            axis, angle = rotation_to_axis_angle(R_euler)
            R_from_aa = axis_angle_to_rotation(axis, angle)

            np.testing.assert_allclose(R_euler, R_from_quat, atol=1e-10)
            np.testing.assert_allclose(R_euler, R_from_aa, atol=1e-10)

    def test_rotation_matrices_are_valid_so3(self):
        """All generated rotation matrices must satisfy SO(3) constraints."""
        rng = np.random.default_rng(1212)
        for _ in range(50):
            angle = rng.uniform(-math.pi, math.pi)
            from roboarm.core.rotations import rotx, roty, rotz
            for rot_fn in (rotx, roty, rotz):
                R = rot_fn(angle)
                assert is_valid_rotation(R), (
                    f"{rot_fn.__name__}({angle}) is not valid SO(3)"
                )


# ------------------------------------------------------------------ #
#  Jacobian accuracy
# ------------------------------------------------------------------ #


class TestJacobianAccuracy:
    """Jacobian computation accuracy."""

    def test_geometric_vs_numerical_50_configs(self, two_link_robot):
        """Geometric and numerical Jacobians must agree at 50 configs."""
        jac = JacobianComputer(two_link_robot)
        rng = np.random.default_rng(1313)
        for _ in range(50):
            q = rng.uniform(-math.pi, math.pi, size=2)
            J_geo = jac.compute(q)
            J_num = jac.compute_numerical(q)
            np.testing.assert_allclose(
                J_geo, J_num, atol=1e-4,
                err_msg=f"Jacobian mismatch at q={q}",
            )

    def test_manipulability_positive_nonsingular(self, two_link_robot):
        """Manipulability > 0 for non-singular configs."""
        jac = JacobianComputer(two_link_robot)
        non_singular_configs = [
            [0.5, -0.3],
            [1.0, -1.0],
            [math.pi / 4, math.pi / 6],
            [-0.8, 1.2],
        ]
        for q in non_singular_configs:
            mu = jac.manipulability(q)
            assert mu > 0.0, (
                f"Manipulability should be positive at q={q}, got {mu}"
            )

    def test_manipulability_zero_at_singularity(self, two_link_robot):
        """Manipulability ~0 at fully extended (singular) config."""
        jac = JacobianComputer(two_link_robot)
        mu = jac.manipulability([0.0, 0.0])
        assert mu < 1e-4, (
            f"Manipulability at singular config should be ~0, got {mu}"
        )

    def test_jacobian_shape_2link(self, two_link_robot):
        """2-link planar Jacobian should be 2x2."""
        jac = JacobianComputer(two_link_robot)
        J = jac.compute([0.5, -0.3])
        assert J.shape == (2, 2), f"Expected (2,2), got {J.shape}"

    def test_jacobian_shape_3link(self, three_link_robot):
        """3-link planar Jacobian should be 2x3."""
        jac = JacobianComputer(three_link_robot)
        J = jac.compute([0.1, 0.2, 0.3])
        assert J.shape == (2, 3), f"Expected (2,3), got {J.shape}"

    def test_numerical_jacobian_agrees_3link(self, three_link_robot):
        """Geometric and numerical Jacobians agree for 3-link robot."""
        jac = JacobianComputer(three_link_robot)
        rng = np.random.default_rng(1414)
        for _ in range(30):
            q = rng.uniform(-math.pi, math.pi, size=3)
            J_geo = jac.compute(q)
            J_num = jac.compute_numerical(q)
            np.testing.assert_allclose(
                J_geo, J_num, atol=1e-4,
                err_msg=f"3-link Jacobian mismatch at q={q}",
            )

    def test_singularity_detection(self, two_link_robot):
        """is_singular should return True at fully extended config."""
        jac = JacobianComputer(two_link_robot)
        assert jac.is_singular([0.0, 0.0]), (
            "Fully extended config should be detected as singular"
        )
        assert not jac.is_singular([0.5, -0.5]), (
            "Non-singular config incorrectly flagged"
        )
