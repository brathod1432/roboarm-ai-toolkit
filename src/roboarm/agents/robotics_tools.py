"""Robotics-specific tool factory.

Builds a :class:`ToolRegistry` populated with tools for forward / inverse
kinematics, Jacobian analysis, robot description, and solver comparison.
All tools operate on a given :class:`RobotArm` instance.
"""

from __future__ import annotations

import logging

import numpy as np

from roboarm.agents.tools import ToolDefinition, ToolRegistry
from roboarm.core.robot import RobotArm
from roboarm.core.types import EndEffectorPose

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Individual tool implementations
# ------------------------------------------------------------------

def _describe_robot(robot: RobotArm) -> str:
    """Return a human-readable description of the robot."""
    lines = [
        f"Robot: {robot.name}",
        f"  DOF: {robot.n_dof}",
        f"  Joints: {robot.n_joints}",
        f"  Joint names: {robot.joint_names}",
    ]
    limits = robot.joint_limits
    for i, (name, lim) in enumerate(zip(robot.joint_names, limits)):
        if lim is not None:
            lines.append(
                f"  {name}: [{lim.lower:.4f}, {lim.upper:.4f}] rad"
            )
        else:
            lines.append(f"  {name}: no limits")
    return "\n".join(lines)


def _compute_fk(robot: RobotArm, angles: list[float]) -> str:
    """Compute forward kinematics and return a formatted result."""
    q = [float(a) for a in angles]
    pose = robot.forward_kinematics(q)
    lines = [
        "Forward Kinematics Result:",
        f"  Input angles (rad): {[round(a, 6) for a in q]}",
        "  End-effector position:",
        f"    x = {pose.x:.6f}",
        f"    y = {pose.y:.6f}",
        f"    z = {pose.z:.6f}",
    ]
    return "\n".join(lines)


def _solve_ik(
    robot: RobotArm,
    target_x: float,
    target_y: float,
    target_z: float | None = None,
    solver_name: str | None = None,
) -> str:
    """Solve inverse kinematics for a target position."""
    # Lazy import to avoid hard failure if the solvers layer is not yet
    # fully wired up.
    try:
        from roboarm.kinematics.solvers.registry import IKSolverRegistry
    except ImportError:
        return (
            "Error: IK solver registry is not available. "
            "Ensure the kinematics.solvers package is installed."
        )

    z = target_z if target_z is not None else 0.0
    position = np.array([target_x, target_y, z], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    target_pose = EndEffectorPose(
        position=position,
        rotation=np.eye(3, dtype=np.float64),
        transform=transform,
    )

    chosen = solver_name or "damped_least_squares"

    try:
        registry = IKSolverRegistry()
        solver = registry.create(chosen, robot)
    except Exception as exc:
        return (
            f"Error creating solver {chosen!r}: {exc}"
        )

    try:
        result = solver.solve(target_pose)
    except Exception as exc:
        return f"Error during IK solve: {exc}"

    lines = [
        f"Inverse Kinematics Result (solver: {chosen}):",
        f"  Target position: x={target_x:.4f}, y={target_y:.4f}, z={z:.4f}",
        f"  Success: {result.success}",
    ]
    if result.success and result.primary is not None:
        angles = result.primary.values
        lines.append(
            f"  Solution (rad): {[round(float(v), 6) for v in angles]}"
        )
        # FK verification — confirm the solution actually reaches the target
        try:
            fk_check = robot.forward_kinematics(angles)
            verified_error = float(np.linalg.norm(
                position[:2 if robot.is_planar else 3]
                - fk_check.position[:2 if robot.is_planar else 3]
            ))
            lines.append(f"  Verified FK error: {verified_error:.6e}")
        except Exception:
            pass  # verification is best-effort; never block the response
    elif not result.success and result.best_attempt is not None:
        lines.append(
            f"  Best attempt (rad): {[round(float(v), 6) for v in result.best_attempt.values]}"
        )
    lines.extend([
        f"  Iterations: {result.iterations}",
        f"  Residual error: {result.residual_error:.6e}",
        f"  Computation time: {result.computation_time_ms:.2f} ms",
    ])
    if result.messages:
        lines.append(f"  Messages: {'; '.join(result.messages)}")
    return "\n".join(lines)


def _compute_jacobian(
    robot: RobotArm,
    angles: list[float],
    _jac_cache: dict[int, object] | None = None,
) -> str:
    """Compute the Jacobian matrix and manipulability.

    *_jac_cache* is injected by :func:`build_robotics_tools` so the
    :class:`JacobianComputer` is reused across calls instead of being
    re-instantiated (and re-running planarity detection) every time.
    """
    from roboarm.kinematics.jacobian import JacobianComputer

    q = [float(a) for a in angles]
    # Use cached JacobianComputer if supplied, otherwise create a new one
    if _jac_cache is not None and id(robot) in _jac_cache:
        jc = _jac_cache[id(robot)]
    else:
        jc = JacobianComputer(robot)
        if _jac_cache is not None:
            _jac_cache[id(robot)] = jc
    jacobian = jc.compute(q)
    mu = jc.manipulability(q)
    singular = jc.is_singular(q)

    lines = [
        "Jacobian Analysis:",
        f"  Input angles (rad): {[round(a, 6) for a in q]}",
        f"  Jacobian matrix ({jacobian.shape[0]}x{jacobian.shape[1]}):",
    ]
    for row_idx in range(jacobian.shape[0]):
        row_vals = "  ".join(f"{v:+.6f}" for v in jacobian[row_idx])
        lines.append(f"    [{row_vals}]")
    lines.extend([
        f"  Manipulability index: {mu:.6e}",
        f"  Near singularity: {singular}",
    ])
    return "\n".join(lines)


def _compare_solvers(
    robot: RobotArm,
    target_x: float,
    target_y: float,
    target_z: float | None = None,
) -> str:
    """Compare all available IK solvers for a given target."""
    try:
        from roboarm.kinematics.solvers.registry import IKSolverRegistry
    except ImportError:
        return (
            "Error: IK solver registry is not available. "
            "Ensure the kinematics.solvers package is installed."
        )

    z = target_z if target_z is not None else 0.0
    position = np.array([target_x, target_y, z], dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    target_pose = EndEffectorPose(
        position=position,
        rotation=np.eye(3, dtype=np.float64),
        transform=transform,
    )

    registry = IKSolverRegistry()
    available = registry.available()

    if not available:
        return "No IK solvers are registered."

    lines = [
        "IK Solver Comparison:",
        f"  Target: x={target_x:.4f}, y={target_y:.4f}, z={z:.4f}",
        f"  Solvers tested: {len(available)}",
        "",
        f"  {'Solver':<28} {'Success':<10} {'Error':<14} "
        f"{'Time (ms)':<12} {'Iterations':<12}",
        f"  {'-' * 76}",
    ]

    for solver_name in available:
        try:
            solver = registry.create(solver_name, robot)
            result = solver.solve(target_pose)
            # Use the solver's self-reported time (excludes instantiation overhead
            # and is consistent with what individual solve_ik calls report).
            lines.append(
                f"  {solver_name:<28} "
                f"{'Yes' if result.success else 'No':<10} "
                f"{result.residual_error:<14.6e} "
                f"{result.computation_time_ms:<12.2f} "
                f"{result.iterations:<12}"
            )
        except Exception as exc:
            logger.warning(
                "Solver %r failed during comparison: %s",
                solver_name, exc,
            )
            lines.append(
                f"  {solver_name:<28} {'ERROR':<10} "
                f"{str(exc)[:40]}"
            )

    return "\n".join(lines)


# ------------------------------------------------------------------
# Public factory
# ------------------------------------------------------------------

def build_robotics_tools(robot: RobotArm) -> ToolRegistry:
    """Create a :class:`ToolRegistry` with FK, IK, Jacobian, describe,
    and compare tools pre-registered.

    A shared :class:`JacobianComputer` cache is created once and
    injected into the ``compute_jacobian`` tool so the planner's
    planarity detection only runs on first use.

    Args:
        robot: The robot arm model the tools will operate on.

    Returns:
        Fully populated :class:`ToolRegistry`.
    """
    # Shared cache: {id(robot): JacobianComputer} — avoids re-instantiation
    _jac_cache: dict[int, object] = {}
    registry = ToolRegistry()

    registry.register(ToolDefinition(
        name="describe_robot",
        description=(
            "Describe the robot arm: name, DOF, joints, joint names, "
            "and joint limits."
        ),
        parameters={},
        function=lambda: _describe_robot(robot),
    ))

    registry.register(ToolDefinition(
        name="compute_fk",
        description=(
            "Compute forward kinematics for the given joint angles "
            "(list of floats in radians) and return the end-effector "
            "position."
        ),
        parameters={
            "angles": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Joint angles in radians.",
            },
        },
        function=lambda angles: _compute_fk(robot, angles),
    ))

    registry.register(ToolDefinition(
        name="solve_ik",
        description=(
            "Solve inverse kinematics for a target Cartesian position. "
            "Returns joint angles that reach the target."
        ),
        parameters={
            "target_x": {
                "type": "number",
                "description": "Target X coordinate.",
            },
            "target_y": {
                "type": "number",
                "description": "Target Y coordinate.",
            },
            "target_z": {
                "type": "number",
                "description": "Target Z coordinate (optional, default 0).",
            },
            "solver_name": {
                "type": "string",
                "description": (
                    "IK solver to use (default: damped_least_squares)."
                ),
            },
        },
        function=lambda target_x, target_y, target_z=None,
                        solver_name=None: _solve_ik(
            robot, target_x, target_y, target_z, solver_name,
        ),
    ))

    registry.register(ToolDefinition(
        name="compute_jacobian",
        description=(
            "Compute the Jacobian matrix and manipulability index at "
            "the given joint configuration."
        ),
        parameters={
            "angles": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Joint angles in radians.",
            },
        },
        function=lambda angles: _compute_jacobian(robot, angles, _jac_cache),
    ))

    registry.register(ToolDefinition(
        name="compare_solvers",
        description=(
            "Run all available IK solvers on the same target and return "
            "a comparison table with success, error, time, and iterations."
        ),
        parameters={
            "target_x": {
                "type": "number",
                "description": "Target X coordinate.",
            },
            "target_y": {
                "type": "number",
                "description": "Target Y coordinate.",
            },
            "target_z": {
                "type": "number",
                "description": "Target Z coordinate (optional, default 0).",
            },
        },
        function=lambda target_x, target_y, target_z=None: (
            _compare_solvers(robot, target_x, target_y, target_z)
        ),
    ))

    logger.info(
        "Built robotics tool registry with %d tools for %r",
        len(registry), robot.name,
    )
    return registry
