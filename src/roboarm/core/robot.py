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


# Maximum joints allowed in deserialized robots (DoS protection)
_MAX_JOINTS: int = 32


def _validate_dh_float(value: object, field: str) -> float:
    """Convert to float and reject non-finite values (NaN/Inf injection guard).

    Args:
        value: Raw value from untrusted input (JSON dict, etc.).
        field: Field name for the error message.

    Returns:
        Validated finite float.

    Raises:
        ValidationError: If the value is not finite or not a number.
    """
    import math
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"DH param '{field}' must be a number, got {value!r}") from exc
    if not math.isfinite(f):
        raise ValidationError(
            f"DH param '{field}' must be finite (got {f}). "
            "NaN/Inf values are not permitted."
        )
    return f


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
        if len(joints) > _MAX_JOINTS:
            raise ValidationError(
                f"Robot must have at most {_MAX_JOINTS} joints, got {len(joints)}"
            )
        self._joints = list(joints)
        self.name = name
        # Named pose store: {pose_name: joint_angle_array}
        self._poses: dict[str, np.ndarray] = {}
        # Solver cache: keyed by solver name, populated lazily on first use.
        # Avoids re-instantiating the solver (registry lookup + JacobianComputer
        # setup) on every call to solve_ik() — critical for real-time control loops.
        self._solver_cache: dict[str, object] = {}

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
        # Guard: NaN/Inf in output transform means DH parameters or joint
        # angles contained non-finite values — surface this immediately
        # rather than allowing silent NaN propagation downstream.
        if not np.all(np.isfinite(T[:3, :])):
            raise ValidationError(
                "forward_kinematics() produced non-finite end-effector position. "
                "This usually means DH parameters (a, d, alpha) contain NaN or Inf. "
                "Check that all robot parameters are valid finite numbers."
            )
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

        # Security: reject non-finite or astronomically large coordinates
        import math
        for i, coord in enumerate(pos_arr):
            if not math.isfinite(coord):
                raise ValidationError(
                    f"Position coordinate [{i}] is not finite: {coord!r}"
                )
            if abs(coord) > 1e6:
                raise ValidationError(
                    f"Position coordinate [{i}] = {coord:.4g} exceeds the maximum "
                    f"allowed magnitude of 1e6 m. Did you mix up units?"
                )

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = pos_arr
        target = EndEffectorPose(
            position=pos_arr,
            rotation=np.eye(3, dtype=np.float64),
            transform=T,
        )

        # Use cached solver instance; create on first call for this name.
        if solver_name not in self._solver_cache:
            self._solver_cache[solver_name] = IKSolverRegistry.create(solver_name, self)
        solver = self._solver_cache[solver_name]
        return solver.solve(target, q0=q0)  # type: ignore[union-attr]

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
        from roboarm.core.exceptions import IKFailedError

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
            raise IKFailedError(
                f"IK solver {solver_name!r} failed to converge "
                f"(residual={result.residual_error:.4e}). "
                "Use solve_ik() to inspect the full result.",
                residual_error=result.residual_error,
                best_attempt=result.best_attempt,
                solver_name=solver_name,
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

        # --- Security: validate structure before processing ---
        _MAX_JOINTS = 32
        if not isinstance(joints_data, list):
            raise ValidationError("'joints' must be a list")
        if len(joints_data) > _MAX_JOINTS:
            raise ValidationError(
                f"Robot definition exceeds maximum of {_MAX_JOINTS} joints "
                f"(got {len(joints_data)}). Possible DoS attempt."
            )

        joints: list[JointConfig] = []
        for j in joints_data:
            dh_d = j["dh_params"]
            dh = DHParams(
                alpha=_validate_dh_float(dh_d.get("alpha", 0.0), "alpha"),
                a=_validate_dh_float(dh_d.get("a", 0.0), "a"),
                d=_validate_dh_float(dh_d.get("d", 0.0), "d"),
                theta=_validate_dh_float(dh_d.get("theta", 0.0), "theta"),
                convention=str(dh_d.get("convention", "standard")),
            )
            if dh.convention not in ("standard", "modified"):
                raise ValidationError(
                    f"Invalid DH convention {dh.convention!r}. "
                    "Must be 'standard' or 'modified'."
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
        out = Path(path).resolve()
        cwd = Path.cwd().resolve()
        try:
            out.relative_to(cwd)
        except ValueError:
            logger.warning(
                "robot.save(): writing outside current working directory: %s "
                "(cwd=%s). Verify this is intentional.",
                out, cwd,
            )
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
        src = Path(path).resolve()
        cwd = Path.cwd().resolve()
        try:
            src.relative_to(cwd)
        except ValueError:
            logger.warning(
                "RobotArm.load(): reading from outside current working directory: %s "
                "(cwd=%s). Verify this is intentional.",
                src, cwd,
            )
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

    def __eq__(self, other: object) -> bool:
        """Two robots are equal if they have the same serialized definition.

        Named poses are also compared.

        Args:
            other: Object to compare with.

        Returns:
            ``True`` if *other* is a :class:`RobotArm` with identical
            joints and poses.
        """
        if not isinstance(other, RobotArm):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        """Hash based on the serialized joint chain (ignoring poses).

        Poses are mutable and therefore excluded from the hash to keep
        the ``hash(robot)`` consistent even after :meth:`save_pose` calls.
        """
        import hashlib
        import json as _json
        # Hash only joints and name, not poses (poses are mutable)
        d = {"name": self.name, "joints": self.to_dict()["joints"]}
        blob = _json.dumps(d, sort_keys=True).encode()
        return int(hashlib.sha256(blob).hexdigest(), 16) % (2**61)

    def copy(self) -> RobotArm:
        """Return a deep copy of this robot arm including joints and poses.

        The copy is fully independent — modifying the original's joints
        or poses does not affect the copy and vice versa.

        Returns:
            A new :class:`RobotArm` with identical configuration.

        Example::

            robot_copy = robot.copy()
            robot_copy.save_pose("variant", [0.5, 0.5])  # does not affect robot
        """
        new_robot = RobotArm.from_dict(self.to_dict())
        logger.debug("Copied robot %r", self.name)
        return new_robot

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks.

        Returns an HTML table showing all joints with their DH parameters
        and joint limits, making the robot model immediately readable in
        a notebook cell output.
        """
        rows = []
        var_idx = 0
        for jc in self._joints:
            dh = jc.dh_params
            kind = "revolute" if jc.is_variable else "fixed"
            lim_str = "—"
            if jc.limits is not None:
                lo = f"{jc.limits.lower:.3f}"
                hi = f"{jc.limits.upper:.3f}"
                lim_str = f"[{lo}, {hi}] rad"
            rows.append(
                f"<tr>"
                f"<td>{jc.name or ('J' + str(var_idx + 1) if jc.is_variable else 'fixed')}</td>"
                f"<td>{kind}</td>"
                f"<td>{dh.alpha:.4f}</td>"
                f"<td>{dh.a:.4f}</td>"
                f"<td>{dh.d:.4f}</td>"
                f"<td>{dh.convention}</td>"
                f"<td>{lim_str}</td>"
                f"</tr>"
            )
            if jc.is_variable:
                var_idx += 1

        header = (
            "<th>Name</th><th>Type</th>"
            "<th>alpha (rad)</th><th>a (m)</th><th>d (m)</th>"
            "<th>Convention</th><th>Limits</th>"
        )
        table = (
            f'<table border="1" style="border-collapse:collapse;font-family:monospace">'
            f"<caption><b>{self.name}</b> | {self.n_dof} DOF | "
            f"{self.n_joints} joints</caption>"
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
        )
        return table

    def gravity_torques(
        self,
        q: Sequence[float],
        link_masses: Sequence[float] | None = None,
        payload_mass: float = 0.0,
        gravity: float = 9.81,
    ) -> np.ndarray:
        """Estimate gravitational joint torques using simplified statics.

        Each joint torque is approximated as the sum of gravitational
        moments from all links distal to that joint.  Link masses are
        approximated as point masses located at the midpoint of each link.

        This is a *simplified* estimate — it ignores link inertia tensors
        and assumes all joints are revolute rotating about their local
        z-axis.  For precise dynamics use a full Newton-Euler or
        Lagrangian formulation.

        Args:
            q: Joint angles in radians.
            link_masses: Mass of each variable link in kg.  Length must
                equal ``n_dof``.  If ``None``, all links are assumed
                massless and only the payload contributes.
            payload_mass: Mass of the tool/payload at the end-effector
                (kg).  Default is 0.
            gravity: Gravitational acceleration magnitude (m/s²).
                Default is 9.81.

        Returns:
            ``(n_dof,)`` array of gravitational torques in N·m.  Positive
            torque acts in the direction of increasing joint angle.

        Example::

            masses = [0.5, 0.3]       # 0.5 kg link 1, 0.3 kg link 2
            tau = robot.gravity_torques([0.5, -0.3], link_masses=masses,
                                        payload_mass=0.1)
        """
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        if q_arr.size != self.n_dof:
            raise ValidationError(
                f"Expected {self.n_dof} joint angles, got {q_arr.size}"
            )

        if link_masses is None:
            masses = np.zeros(self.n_dof, dtype=np.float64)
        else:
            masses = np.asarray(link_masses, dtype=np.float64).ravel()
            if masses.size != self.n_dof:
                raise ValidationError(
                    f"link_masses length {masses.size} != n_dof {self.n_dof}"
                )

        # Get all joint positions
        all_positions = self.joint_positions(q_arr)
        # all_positions[0] = base (origin), all_positions[-1] = end-effector

        # Gravity vector (pointing down in world z)
        g_vec = np.array([0.0, 0.0, -gravity], dtype=np.float64)

        torques = np.zeros(self.n_dof, dtype=np.float64)

        for joint_idx in range(self.n_dof):
            # Joint origin position
            joint_pos = all_positions[joint_idx]

            # Torque from each distal link's centre-of-mass
            for link_idx in range(joint_idx, self.n_dof):
                if masses[link_idx] == 0.0:
                    continue
                # Approximate CoM at midpoint between link origin and next joint
                com = 0.5 * (all_positions[link_idx] + all_positions[link_idx + 1])
                r = com - joint_pos  # moment arm

                # Gravitational force on this link mass
                f = masses[link_idx] * g_vec

                # Torque = r × F, projected onto joint z-axis
                # Joint z-axis is world z for planar arms; for 3-D use the
                # joint's local z from the cumulative transform
                torque_vec = np.cross(r, f)
                torques[joint_idx] += torque_vec[2]  # z-component

            # Torque from end-effector payload
            if payload_mass > 0.0:
                r_ee = all_positions[-1] - joint_pos
                f_ee = payload_mass * g_vec
                torque_ee = np.cross(r_ee, f_ee)
                torques[joint_idx] += torque_ee[2]

        return torques

    async def solve_ik_async(
        self,
        position: Sequence[float],
        solver_name: str = "damped_least_squares",
        q0: Sequence[float] | None = None,
    ) -> IKSolution:
        """Asynchronous inverse kinematics solve.

        Runs :meth:`solve_ik` in a thread pool so it does not block the
        event loop when used in an ``asyncio``-based server or control loop.

        Args:
            position: Target ``[x, y]`` or ``[x, y, z]`` position in metres.
            solver_name: Registered IK solver name.
            q0: Optional initial joint-angle guess.

        Returns:
            :class:`IKSolution` with convergence details.

        Example::

            import asyncio

            async def main():
                result = await robot.solve_ik_async([1.0, 0.5])
                print(result.success)

            asyncio.run(main())
        """
        import asyncio
        import functools
        return await asyncio.get_event_loop().run_in_executor(
            None,
            functools.partial(self.solve_ik, position, solver_name, q0),
        )

    def fk_batch(
        self,
        q_array: np.ndarray,
        full: bool = False,
    ) -> np.ndarray | list[EndEffectorPose]:
        """Compute forward kinematics for many joint configurations at once.

        Convenience wrapper around :func:`~roboarm.kinematics.batch.batch_fk`
        that is discoverable directly on the robot model.

        Args:
            q_array: ``(N, n_dof)`` array of joint configurations in radians.
            full: If ``False`` (default), return an ``(N, 3)`` position array.
                If ``True``, return a list of :class:`EndEffectorPose` objects.

        Returns:
            ``(N, 3)`` float64 position array, or list of
            :class:`EndEffectorPose` when *full* is ``True``.

        Example::

            import numpy as np
            Q = np.random.uniform(-np.pi, np.pi, (1000, robot.n_dof))
            positions = robot.fk_batch(Q)   # (1000, 3)
        """
        from roboarm.kinematics.batch import batch_fk
        return batch_fk(self, q_array, full=full)

    def ik_batch(
        self,
        targets: Sequence[Sequence[float]],
        solver_name: str = "damped_least_squares",
        q0_list: Sequence[Sequence[float]] | None = None,
        warm_start: bool = True,
    ) -> list[IKSolution]:
        """Solve inverse kinematics for many target positions at once.

        Convenience wrapper around :func:`~roboarm.kinematics.batch.batch_ik`.

        Args:
            targets: Sequence of ``[x, y]`` or ``[x, y, z]`` target positions.
            solver_name: IK solver registry name.
            q0_list: Optional per-target initial guesses.
            warm_start: If ``True`` (default), use the previous solution as
                the initial guess for the next target (warm-start chaining).

        Returns:
            List of :class:`IKSolution` objects in the same order as *targets*.

        Example::

            targets = [(1.0, 0.5), (0.8, 0.6), (1.2, 0.3)]
            results = robot.ik_batch(targets)
        """
        from roboarm.kinematics.batch import batch_ik
        return batch_ik(
            self,
            targets,
            solver_name=solver_name,
            q0_list=q0_list,
            warm_start=warm_start,
        )
