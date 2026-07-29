"""Trajectory safety validation.

Checks a joint-space trajectory for joint-limit violations, kinematic
singularities, excessive joint acceleration, and user-defined forbidden
spatial zones before execution on hardware.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from roboarm.core.robot import RobotArm

logger = logging.getLogger(__name__)


@dataclass
class LimitViolation:
    """A single joint-limit violation in a trajectory.

    Attributes:
        step: Waypoint index (0-based).
        joint: Joint index (0-based).
        joint_name: Human-readable joint name.
        value: Actual angle value (radians).
        lower: Lower limit (radians).
        upper: Upper limit (radians).
    """

    step: int
    joint: int
    joint_name: str
    value: float
    lower: float
    upper: float


@dataclass
class SingularityWarning:
    """A near-singular waypoint in a trajectory.

    Attributes:
        step: Waypoint index (0-based).
        manipulability: Yoshikawa index at this configuration.
        threshold: The threshold below which the point is flagged.
    """

    step: int
    manipulability: float
    threshold: float


@dataclass
class AccelerationViolation:
    """A joint acceleration that exceeds the hardware limit.

    Attributes:
        step: Inter-step interval index (0-based); the acceleration is
            computed between waypoints *step* and *step + 2*.
        joint: Joint index (0-based).
        joint_name: Human-readable joint name.
        value: Estimated joint acceleration in rad/s².
        limit: The acceleration limit that was exceeded (rad/s²).
    """

    step: int
    joint: int
    joint_name: str
    value: float
    limit: float


@dataclass
class SafeZoneViolation:
    """An end-effector position that entered a forbidden zone.

    Attributes:
        step: Waypoint index (0-based).
        position: End-effector position ``[x, y, z]`` at this waypoint.
        zone_label: Human-readable label for the violated zone.
    """

    step: int
    position: np.ndarray
    zone_label: str


@dataclass
class TrajectoryReport:
    """Full validation report for a trajectory.

    Attributes:
        is_safe: ``True`` when no limit violations, singularities,
            acceleration violations, or safe-zone violations were found.
        limit_violations: All joint-limit violations found.
        singularities: All near-singular waypoints found.
        acceleration_violations: Waypoints where estimated joint
            acceleration exceeded the limit.
        safe_zone_violations: Waypoints where the end-effector entered
            a forbidden Cartesian zone.
        max_joint_step_rad: Maximum angle change between adjacent steps
            across all joints (useful for detecting velocity spikes).
        n_steps: Total number of waypoints checked.
        n_dof: Degrees of freedom.
    """

    is_safe: bool
    limit_violations: list[LimitViolation] = field(default_factory=list)
    singularities: list[SingularityWarning] = field(default_factory=list)
    acceleration_violations: list[AccelerationViolation] = field(default_factory=list)
    safe_zone_violations: list[SafeZoneViolation] = field(default_factory=list)
    max_joint_step_rad: float = 0.0
    n_steps: int = 0
    n_dof: int = 0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines = [
            f"Trajectory validation ({self.n_steps} steps, {self.n_dof} DOF)",
            f"  Safe: {self.is_safe}",
            f"  Limit violations: {len(self.limit_violations)}",
            f"  Singular waypoints: {len(self.singularities)}",
            f"  Acceleration violations: {len(self.acceleration_violations)}",
            f"  Safe-zone violations: {len(self.safe_zone_violations)}",
            f"  Max joint step: {self.max_joint_step_rad:.4f} rad",
        ]
        for v in self.limit_violations[:5]:
            lines.append(
                f"    Limit step={v.step} joint={v.joint_name} "
                f"value={v.value:.4f} limits=[{v.lower:.4f},{v.upper:.4f}]"
            )
        if len(self.limit_violations) > 5:
            lines.append(f"    … and {len(self.limit_violations) - 5} more")
        for s in self.singularities[:5]:
            lines.append(f"    Singularity step={s.step} mu={s.manipulability:.6e}")
        if len(self.singularities) > 5:
            lines.append(f"    … and {len(self.singularities) - 5} more")
        for a in self.acceleration_violations[:3]:
            lines.append(
                f"    Accel step={a.step} joint={a.joint_name} "
                f"value={a.value:.4f} limit={a.limit:.4f} rad/s²"
            )
        for z in self.safe_zone_violations[:3]:
            lines.append(
                f"    SafeZone step={z.step} zone={z.zone_label!r} "
                f"pos=({z.position[0]:.3f},{z.position[1]:.3f},{z.position[2]:.3f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal zone helper
# ---------------------------------------------------------------------------

@dataclass
class _ForbiddenZone:
    """Internal representation of a forbidden Cartesian zone."""

    label: str
    kind: str  # "sphere" or "box"
    center: np.ndarray
    # sphere → radius; box → half-extents (3-D)
    params: np.ndarray

    def contains(self, point: np.ndarray) -> bool:
        """Return True if *point* is inside this zone."""
        diff = point - self.center
        if self.kind == "sphere":
            return bool(np.dot(diff, diff) <= self.params[0] ** 2)
        # box: axis-aligned
        return bool(np.all(np.abs(diff) <= self.params))


class TrajectoryValidator:
    """Validates a joint-space trajectory for safety before execution.

    Checks performed (all optional, controlled by constructor arguments):

    * **Joint limits** — every angle at every waypoint must lie within
      the configured mechanical limits.
    * **Singularities** — configurations where the manipulability index
      falls below *singularity_threshold* are flagged.
    * **Acceleration** — when *dt* is provided, the finite-difference
      acceleration between adjacent waypoints is compared against each
      joint's ``acceleration_max`` limit.
    * **Safe zones** — user-defined forbidden Cartesian regions (spheres
      or axis-aligned boxes) are checked via FK at every waypoint.

    Args:
        robot: The robot arm model.
        singularity_threshold: Manipulability below which a configuration
            is flagged as singular.  Default ``1e-4``.
        check_singularities: If ``True`` (default), compute manipulability
            at every waypoint.  Disable for speed on long trajectories.
        max_joint_step_rad: If set, flag any adjacent-waypoint joint step
            exceeding this value (rad).  ``None`` means no check.
        dt: Time step in seconds between consecutive waypoints.  Required
            for acceleration checking; ignored if ``None``.

    Example::

        validator = TrajectoryValidator(robot, dt=0.02)
        validator.add_forbidden_sphere([0.5, 0.5, 0.0], radius=0.2,
                                       label="obstacle")
        report = validator.check(trajectory_array)
        if not report.is_safe:
            print(report.summary())
    """

    def __init__(
        self,
        robot: RobotArm,
        singularity_threshold: float = 1e-4,
        check_singularities: bool = True,
        max_joint_step_rad: float | None = None,
        dt: float | None = None,
    ) -> None:
        self._robot = robot
        self._threshold = singularity_threshold
        self._check_sing = check_singularities
        self._max_step = max_joint_step_rad
        self._dt = dt
        self._zones: list[_ForbiddenZone] = []

    # ------------------------------------------------------------------
    # Safe zone configuration
    # ------------------------------------------------------------------

    def add_forbidden_sphere(
        self,
        center: list[float] | np.ndarray,
        radius: float,
        label: str = "sphere",
    ) -> None:
        """Declare a spherical forbidden zone.

        The end-effector must not enter a sphere of *radius* centred at
        *center*.

        Args:
            center: ``[x, y, z]`` centre of the sphere (metres).
            radius: Sphere radius (metres).  Must be positive.
            label: Human-readable label for error messages.

        Raises:
            ValueError: If *radius* is not positive or *center* is not
                a 3-element sequence.
        """
        c = np.asarray(center, dtype=np.float64).ravel()
        if c.size != 3:
            raise ValueError(f"center must be a 3-element sequence, got {c.size}")
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self._zones.append(
            _ForbiddenZone(
                label=label,
                kind="sphere",
                center=c,
                params=np.array([radius], dtype=np.float64),
            )
        )
        logger.debug(
            "Added forbidden sphere: center=%s, r=%.4f, label=%r",
            c.tolist(), radius, label,
        )

    def add_forbidden_box(
        self,
        center: list[float] | np.ndarray,
        dimensions: list[float] | np.ndarray,
        label: str = "box",
    ) -> None:
        """Declare an axis-aligned box forbidden zone.

        The end-effector must not enter the box defined by ``center ± dimensions/2``.

        Args:
            center: ``[x, y, z]`` centre of the box (metres).
            dimensions: ``[dx, dy, dz]`` full side lengths (metres).
                The zone spans ``[cx-dx/2, cx+dx/2]`` on each axis.
            label: Human-readable label for error messages.

        Raises:
            ValueError: If *center* or *dimensions* are not 3-element
                sequences, or any dimension is not positive.
        """
        c = np.asarray(center, dtype=np.float64).ravel()
        d = np.asarray(dimensions, dtype=np.float64).ravel()
        if c.size != 3 or d.size != 3:
            raise ValueError("center and dimensions must each be 3-element sequences")
        if np.any(d <= 0):
            raise ValueError(f"All dimensions must be positive, got {d.tolist()}")
        half = d / 2.0
        self._zones.append(
            _ForbiddenZone(label=label, kind="box", center=c, params=half)
        )
        logger.debug(
            "Added forbidden box: center=%s, dims=%s, label=%r",
            c.tolist(), d.tolist(), label,
        )

    def clear_zones(self) -> None:
        """Remove all previously added forbidden zones."""
        self._zones.clear()
        logger.debug("All forbidden zones cleared")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def check(
        self,
        trajectory: np.ndarray,
        dt: float | None = None,
    ) -> TrajectoryReport:
        """Validate every waypoint in *trajectory*.

        Args:
            trajectory: ``(n_steps, n_dof)`` joint angle array in radians.
            dt: Time step override (seconds).  Uses the constructor value
                if not provided here.

        Returns:
            :class:`TrajectoryReport` with full validation details.
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[1] != self._robot.n_dof:
            raise ValueError(
                f"Expected (n_steps, {self._robot.n_dof}) trajectory, "
                f"got {traj.shape}"
            )

        n_steps, n_dof = traj.shape
        effective_dt = dt if dt is not None else self._dt
        limits = self._robot.joint_limits
        joint_names = self._robot.joint_names

        violations: list[LimitViolation] = []
        singularities: list[SingularityWarning] = []
        accel_violations: list[AccelerationViolation] = []
        zone_violations: list[SafeZoneViolation] = []

        jac_computer = None
        if self._check_sing:
            from roboarm.kinematics.jacobian import JacobianComputer
            jac_computer = JacobianComputer(self._robot)

        check_zones = bool(self._zones)
        check_accel = effective_dt is not None and effective_dt > 0.0

        for i in range(n_steps):
            q = traj[i]

            # 1. Joint limit check
            for j, (lim, jname) in enumerate(zip(limits, joint_names)):
                if lim is not None:
                    if q[j] < lim.lower - 1e-9 or q[j] > lim.upper + 1e-9:
                        violations.append(LimitViolation(
                            step=i, joint=j, joint_name=jname,
                            value=float(q[j]),
                            lower=lim.lower, upper=lim.upper,
                        ))

            # 2. Singularity check
            if jac_computer is not None:
                mu = jac_computer.manipulability(q)
                if mu < self._threshold:
                    singularities.append(SingularityWarning(
                        step=i, manipulability=mu, threshold=self._threshold,
                    ))

            # 3. Safe-zone check (FK needed)
            if check_zones:
                try:
                    pose = self._robot.forward_kinematics(q)
                    ee_pos = pose.position
                    for zone in self._zones:
                        if zone.contains(ee_pos):
                            zone_violations.append(SafeZoneViolation(
                                step=i,
                                position=ee_pos.copy(),
                                zone_label=zone.label,
                            ))
                except Exception as exc:
                    logger.warning("Safe-zone FK failed at step %d: %s", i, exc)

        # 4. Acceleration check (second finite difference over the trajectory)
        if check_accel and n_steps >= 3:
            accel = np.diff(traj, n=2, axis=0) / (effective_dt ** 2)  # type: ignore[operator]
            for i in range(accel.shape[0]):
                for j, (lim, jname) in enumerate(zip(limits, joint_names)):
                    if lim is not None and lim.acceleration_max is not None:
                        a_val = abs(float(accel[i, j]))
                        if a_val > lim.acceleration_max:
                            accel_violations.append(AccelerationViolation(
                                step=i,
                                joint=j,
                                joint_name=jname,
                                value=float(accel[i, j]),
                                limit=lim.acceleration_max,
                            ))

        # 5. Max joint step
        max_step = 0.0
        if n_steps > 1:
            diffs = np.abs(np.diff(traj, axis=0))
            max_step = float(np.max(diffs))
            if self._max_step is not None and max_step > self._max_step:
                logger.warning(
                    "Trajectory max joint step %.4f rad exceeds threshold %.4f rad",
                    max_step, self._max_step,
                )

        is_safe = (
            len(violations) == 0
            and len(singularities) == 0
            and len(accel_violations) == 0
            and len(zone_violations) == 0
        )

        report = TrajectoryReport(
            is_safe=is_safe,
            limit_violations=violations,
            singularities=singularities,
            acceleration_violations=accel_violations,
            safe_zone_violations=zone_violations,
            max_joint_step_rad=max_step,
            n_steps=n_steps,
            n_dof=n_dof,
        )

        if is_safe:
            logger.info("Trajectory validation PASSED (%d steps, %d DOF)", n_steps, n_dof)
        else:
            logger.warning(
                "Trajectory FAILED: %d limit, %d singular, %d accel, %d zone violations",
                len(violations), len(singularities),
                len(accel_violations), len(zone_violations),
            )

        return report
