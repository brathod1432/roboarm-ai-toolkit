"""Serial-link robot arm model with forward kinematics.

Defines :class:`RobotArm`, the central model class that holds a kinematic
chain and computes forward kinematics by chaining DH transforms.

Extended capabilities
---------------------
* **One-liner IK** — :meth:`solve_ik` / :meth:`ik` replace the 8-step
  boilerplate that was previously needed to run a single IK solve.
* **Named poses** — :meth:`save_pose`, :meth:`get_pose`, :meth:`fk_at`,
  :meth:`list_poses`, :meth:`delete_pose` give each robot a built-in
  configuration store (home, pick, place, …).
* **Serialization** — :meth:`to_dict`, :meth:`from_dict`, :meth:`save`,
  :meth:`load` let clients persist and reload custom robots via JSON.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from roboarm.core.exceptions import ValidationError
from roboarm.core.transform import (
    chain_transforms,
    extract_position,
    extract_rotation,
    transform_from_dh_params,
)
from roboarm.core.types import (
    DHParams,
    EndEffectorPose,
    IKSolution,
    JointConfig,
    JointLimits,
)

logger = logging.getLogger(__name__)


class RobotArm:
    """Serial-link robot arm built from a chain of DH-parameterised joints.

    Example::

        from roboarm.core.types import DHParams, JointConfig
        j1 = JointConfig(DHParams(alpha=0, a=1.0, d=0, theta=0))
        j2 = JointConfig(DHParams(alpha=0, a=1.0, d=0, theta=0))
        robot = RobotArm([j1, j2], name="2-Link Planar")

        # Forward kinematics
        pose = robot.forward_kinematics([0.5, -0.3])

        # One-liner IK (new)
        q = robot.ik(x=1.0, y=0.5)

        # Named poses (new)
        robot.save_pose("home", [0.0, 0.0])
        robot.get_pose("home")

        # Serialization (new)
        robot.save("my_arm.json")
        robot2 = RobotArm.load("my_arm.json")
    """

    def __init__(
        self,
        joints: list[JointConfig],
        name: str = "Robot",
    ) -> None:
        """Initialise the robot arm.

        Args:
            joints: Ordered list of joint configurations (base to tip).
            name: Human-readable robot name.
        """
        if not joints:
            raise ValidationError("Robot must have at least one joint")
        self._joints = list(joints)
        self.name = name
        # Named pose store: {pose_name: joint_angle_array}
        self._poses: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Basic properties
    # ------------------------------------------------------------------

    @property
    def joints(self) -> list[JointConfig]:
        """All joints in the chain."""
        return list(self._joints)

    @property
    def n_joints(self) -> int:
        """Total number of joints (including fixed offsets)."""
        return len(self._joints)

    @property
    def n_dof(self) -> int:
        """Number of actuated (variable) degrees of freedom."""
        return sum(1 for j in self._joints if j.is_variable)

    @property
    def joint_names(self) -> list[str]:
        """Names of all variable joints."""
        return [j.name or f"J{i}" for i, j in enumerate(self._joints) if j.is_variable]

    @property
    def joint_limits(self) -> list[JointLimits | None]:
        """Limits for each variable joint."""
        return [j.limits for j in self._joints if j.is_variable]

    @property
    def is_planar(self) -> bool:
        """``True`` when every joint has ``alpha == 0`` and ``d == 0``.

        A planar robot operates entirely in the XY plane.  This property
        is the canonical planarity check; all sub-modules delegate to it
        rather than replicating the logic locally.
        """
        return all(
            abs(jc.dh_params.alpha) <= 1e-9 and abs(jc.dh_params.d) <= 1e-9
            for jc in self._joints
        )

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(self, q: Sequence[float]) -> EndEffectorPose:
        """Compute end-effector pose from joint angles.

        Args:
            q: Joint angles in radians (length must equal :attr:`n_dof`).

        Returns:
            End-effector position and orientation.

        Raises:
            ValidationError: If ``q`` has the wrong length.
        """
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        if q_arr.size != self.n_dof:
            raise ValidationError(
                f"Expected {self.n_dof} joint angles, got {q_arr.size}"
            )

        transforms = self._compute_link_transforms(q_arr)
        T = chain_transforms(transforms)
        return EndEffectorPose(
            position=extract_position(T),
            rotation=extract_rotation(T),
            transform=T,
        )

    def joint_transforms(self, q: Sequence[float]) -> list[np.ndarray]:
        """Compute the cumulative transform up to each joint frame.

        The returned list has ``n_joints + 1`` entries: ``[T_base, T_01,
        T_02, ..., T_0n]`` where ``T_base`` is the identity.

        Args:
            q: Joint angles in radians.

        Returns:
            List of 4x4 cumulative transforms.
        """
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        if q_arr.size != self.n_dof:
            raise ValidationError(
                f"Expected {self.n_dof} joint angles, got {q_arr.size}"
            )

        link_transforms = self._compute_link_transforms(q_arr)
        cumulative = [np.eye(4, dtype=np.float64)]
        T = np.eye(4, dtype=np.float64)
        for lt in link_transforms:
            T = T @ lt
            cumulative.append(T.copy())
        return cumulative

    def joint_positions(self, q: Sequence[float]) -> np.ndarray:
        """Compute the 3-D position of every joint origin.

        Args:
            q: Joint angles in radians.

        Returns:
            ``(n_joints + 1, 3)`` array of positions (base through TCP).
        """
        cumulative = self.joint_transforms(q)
        return np.array([extract_position(T) for T in cumulative])

    # ------------------------------------------------------------------
    # Inverse kinematics shortcuts (UC1)
    # ------------------------------------------------------------------

    def solve_ik(
        self,
        position: Sequence[float],
        solver_name: str = "damped_least_squares",
        q0: Sequence[float] | None = None,
    ) -> IKSolution:
        """Solve inverse kinematics for a Cartesian target position.

        Convenience wrapper that handles solver import, registry lookup,
        and :class:`EndEffectorPose` construction automatically.

        Args:
            position: Target ``[x, y]`` or ``[x, y, z]`` position in metres.
            solver_name: Registered IK solver name.  Defaults to
                ``"damped_least_squares"``.
            q0: Optional initial joint-angle guess.

        Returns:
            :class:`IKSolution` with convergence details.

        Raises:
            ConfigurationError: If *solver_name* is not registered.

        Example::

            result = robot.solve_ik([1.0, 0.5])
            if result.success:
                print(result.primary.values)
        """
        # Lazy import avoids circular dependency (kinematics → core).
        import roboarm.kinematics.solvers  # noqa: F401  (triggers registration)
        from roboarm.kinematics.solvers.registry import IKSolverRegistry

        pos_arr = np.asarray(position, dtype=np.float64).ravel()
        # Pad to 3-D if only 2-D supplied
        if pos_arr.size == 2:
            pos_arr = np.append(pos_arr, 0.0)

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = pos_arr
        target = EndEffectorPose(
            position=pos_arr,
            rotation=np.eye(3, dtype=np.float64),
            transform=T,
        )

        solver = IKSolverRegistry.create(solver_name, self)
        return solver.solve(target, q0=q0)

    def ik(
        self,
        position: Sequence[float] | None = None,
        solver_name: str = "damped_least_squares",
        q0: Sequence[float] | None = None,
        *,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> np.ndarray:
        """Solve IK and return the joint angle array directly.

        Compared with :meth:`solve_ik`, this method:

        * Accepts either a positional ``[x, y, z]`` argument or keyword
          arguments ``x=``, ``y=``, ``z=``.
        * Returns the joint angle array on success.
        * Raises :class:`~roboarm.core.exceptions.KinematicsError` on
          failure instead of returning an :class:`IKSolution`.

        Args:
            position: Target as ``[x, y]`` or ``[x, y, z]``.
            solver_name: IK solver registry name.
            q0: Initial guess.
            x: Target x coordinate (alternative to *position*).
            y: Target y coordinate (alternative to *position*).
            z: Target z coordinate (default 0 when using keyword form).

        Returns:
            ``(n_dof,)`` float64 array of joint angles in radians.

        Raises:
            KinematicsError: If the solver fails to converge.
            ValidationError: If neither *position* nor *x*/*y* are given.

        Example::

            q = robot.ik(x=1.0, y=0.5)
            q = robot.ik([1.2, 0.8])
        """
        from roboarm.core.exceptions import KinematicsError

        if position is not None:
            pos = list(position)
        elif x is not None and y is not None:
            pos = [x, y] if z is None else [x, y, z]
        else:
            raise ValidationError(
                "Provide either a position sequence or x= and y= keyword arguments."
            )

        result = self.solve_ik(pos, solver_name=solver_name, q0=q0)
        if not result.success or result.primary is None:
            raise KinematicsError(
                f"IK solver {solver_name!r} failed to converge "
                f"(residual={result.residual_error:.4e}). "
                "Use solve_ik() to inspect the full result."
            )
        return result.primary.values

    # ------------------------------------------------------------------
    # Named pose store (UC3)
    # ------------------------------------------------------------------

    def save_pose(self, name: str, q: Sequence[float]) -> None:
        """Store a named joint configuration.

        Args:
            name: Pose identifier (e.g. ``"home"``, ``"pick"``).
            q: Joint angles in radians.

        Raises:
            ValidationError: If *q* has the wrong length.

        Example::

            robot.save_pose("home", [0.0, 0.0])
            robot.save_pose("pick", [0.5, -0.3])
        """
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        if q_arr.size != self.n_dof:
            raise ValidationError(
                f"Expected {self.n_dof} joint values for pose {name!r}, "
                f"got {q_arr.size}"
            )
        self._poses[name] = q_arr
        logger.debug("Saved pose %r: %s", name, q_arr.tolist())

    def get_pose(self, name: str) -> np.ndarray:
        """Retrieve a stored joint configuration by name.

        Args:
            name: Pose identifier.

        Returns:
            ``(n_dof,)`` joint angle array in radians.

        Raises:
            KeyError: If *name* is not stored.
        """
        if name not in self._poses:
            available = sorted(self._poses)
            raise KeyError(
                f"Pose {name!r} not found. Available: {available}"
            )
        return self._poses[name].copy()

    def fk_at(self, name: str) -> EndEffectorPose:
        """Compute FK at a named pose.

        Args:
            name: Pose identifier.

        Returns:
            :class:`EndEffectorPose` at the stored joint configuration.
        """
        return self.forward_kinematics(self.get_pose(name))

    def list_poses(self) -> list[str]:
        """Return a sorted list of all stored pose names."""
        return sorted(self._poses)

    def delete_pose(self, name: str) -> None:
        """Remove a stored pose.

        Args:
            name: Pose identifier to remove.

        Raises:
            KeyError: If *name* is not stored.
        """
        if name not in self._poses:
            raise KeyError(f"Pose {name!r} not found.")
        del self._poses[name]
        logger.debug("Deleted pose %r", name)

    # ------------------------------------------------------------------
    # Serialization (UC2)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the robot to a plain Python dictionary.

        The returned dict contains only JSON-compatible types (str, float,
        bool, None, list, dict) and can be round-tripped via
        :meth:`from_dict`.

        Returns:
            Dictionary representation including joints, limits, and poses.
        """
        joints_data = []
        for jc in self._joints:
            dh = jc.dh_params
            lim: dict[str, float | None] | None = None
            if jc.limits is not None:
                lim = {
                    "lower": jc.limits.lower,
                    "upper": jc.limits.upper,
                    "velocity_max": jc.limits.velocity_max,
                    "acceleration_max": jc.limits.acceleration_max,
                }
            joints_data.append({
                "dh_params": {
                    "alpha": dh.alpha,
                    "a": dh.a,
                    "d": dh.d,
                    "theta": dh.theta,
                    "convention": dh.convention,
                },
                "limits": lim,
                "name": jc.name,
                "is_variable": jc.is_variable,
            })

        poses_data = {name: q.tolist() for name, q in self._poses.items()}

        return {
            "name": self.name,
            "joints": joints_data,
            "poses": poses_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotArm:
        """Reconstruct a :class:`RobotArm` from a dictionary.

        Args:
            data: Dictionary as produced by :meth:`to_dict`.

        Returns:
            A fully reconstructed :class:`RobotArm` with joints and poses.

        Raises:
            ValidationError: If required keys are missing.
        """
        try:
            name = data["name"]
            joints_data = data["joints"]
        except KeyError as exc:
            raise ValidationError(
                f"Missing required key in robot dict: {exc}"
            ) from exc

        joints: list[JointConfig] = []
        for j in joints_data:
            dh_d = j["dh_params"]
            dh = DHParams(
                alpha=float(dh_d["alpha"]),
                a=float(dh_d["a"]),
                d=float(dh_d["d"]),
                theta=float(dh_d["theta"]),
                convention=str(dh_d.get("convention", "standard")),
            )
            lim: JointLimits | None = None
            if j.get("limits") is not None:
                ld = j["limits"]
                lim = JointLimits(
                    lower=float(ld["lower"]),
                    upper=float(ld["upper"]),
                    velocity_max=(
                        float(ld["velocity_max"])
                        if ld.get("velocity_max") is not None
                        else None
                    ),
                    acceleration_max=(
                        float(ld["acceleration_max"])
                        if ld.get("acceleration_max") is not None
                        else None
                    ),
                )
            joints.append(JointConfig(
                dh_params=dh,
                limits=lim,
                name=str(j.get("name", "")),
                is_variable=bool(j.get("is_variable", True)),
            ))

        robot = cls(joints, name=str(name))

        for pose_name, q_list in data.get("poses", {}).items():
            robot.save_pose(pose_name, q_list)

        return robot

    def save(self, path: str | Path, indent: int = 2) -> None:
        """Serialise this robot to a JSON file.

        Args:
            path: File path (created or overwritten).
            indent: JSON indentation spaces.

        Example::

            robot.save("my_arm.json")
        """
        out = Path(path)
        out.write_text(
            json.dumps(self.to_dict(), indent=indent),
            encoding="utf-8",
        )
        logger.info("Robot %r saved to %s", self.name, out)

    @classmethod
    def load(cls, path: str | Path) -> RobotArm:
        """Load a :class:`RobotArm` from a JSON file.

        Args:
            path: Path to a JSON file produced by :meth:`save`.

        Returns:
            A fully reconstructed :class:`RobotArm`.

        Example::

            robot = RobotArm.load("my_arm.json")
        """
        src = Path(path)
        data = json.loads(src.read_text(encoding="utf-8"))
        robot = cls.from_dict(data)
        logger.info("Robot %r loaded from %s", robot.name, src)
        return robot

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_link_transforms(self, q: np.ndarray) -> list[np.ndarray]:
        """Build per-link transforms inserting variable joint angles.

        Args:
            q: Array of variable joint angles (length == n_dof).

        Returns:
            List of 4x4 per-link transforms.
        """
        transforms: list[np.ndarray] = []
        q_idx = 0
        for jc in self._joints:
            if jc.is_variable:
                T = transform_from_dh_params(jc.dh_params, q[q_idx])
                q_idx += 1
            else:
                T = transform_from_dh_params(jc.dh_params, 0.0)
            transforms.append(T)
        return transforms

    def __repr__(self) -> str:
        return (
            f"RobotArm(name={self.name!r}, "
            f"n_joints={self.n_joints}, n_dof={self.n_dof})"
        )
