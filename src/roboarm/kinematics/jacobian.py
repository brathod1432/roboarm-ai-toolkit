"""Jacobian computation for serial-link robot arms.

Provides :class:`JacobianComputer` which computes the geometric Jacobian,
numerical Jacobian, manipulability index, and singularity detection for a
given :class:`RobotArm`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from roboarm.core.robot import RobotArm

logger = logging.getLogger(__name__)


class JacobianComputer:
    """Computes Jacobian matrices and related kinematic quantities.

    For planar robots (all DH ``alpha == 0`` and all ``d == 0``) the Jacobian
    is returned as a compact ``2 x N`` matrix covering only the ``(x, y)``
    linear-velocity rows.  For general 3-D robots a full ``6 x N`` geometric
    Jacobian is returned (3 linear + 3 angular velocity rows).

    Args:
        robot: The robot arm model to compute Jacobians for.

    Example::

        from roboarm.robots.two_link_planar import create_two_link_planar
        jc = JacobianComputer(create_two_link_planar())
        J = jc.compute([0.5, -0.3])
    """

    def __init__(self, robot: RobotArm) -> None:
        self._robot = robot
        self._is_planar = robot.is_planar
        logger.debug(
            "JacobianComputer created for %s (planar=%s)",
            robot.name,
            self._is_planar,
        )

    @property
    def is_planar(self) -> bool:
        """Whether the robot is a planar manipulator."""
        return self._is_planar

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, q: Sequence[float]) -> np.ndarray:
        """Compute the geometric Jacobian at configuration *q*.

        For revolute joint *i* the geometric Jacobian columns are:

        * **Linear part:** ``z_{i-1} x (o_n - o_{i-1})``
        * **Angular part:** ``z_{i-1}``

        where ``z_{i-1}`` is the z-axis of frame *i-1* and ``o`` is the
        frame origin extracted from the cumulative transforms.

        Returns:
            ``(2, n_dof)`` array for planar robots or ``(6, n_dof)`` for
            3-D robots.
        """
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        cumulative = self._robot.joint_transforms(q_arr)

        n_dof = self._robot.n_dof
        o_n = cumulative[-1][:3, 3]  # end-effector origin

        J_full = np.zeros((6, n_dof), dtype=np.float64)

        var_idx = 0
        for jc, T_prev in zip(self._robot.joints, cumulative[:-1]):
            if not jc.is_variable:
                continue
            z = T_prev[:3, 2]  # z-axis of frame i-1
            o = T_prev[:3, 3]  # origin of frame i-1
            J_full[:3, var_idx] = np.cross(z, o_n - o)  # linear
            J_full[3:, var_idx] = z                       # angular
            var_idx += 1

        if self._is_planar:
            logger.debug("Returning 2xN planar Jacobian")
            return J_full[:2, :]

        logger.debug("Returning 6xN full Jacobian")
        return J_full

    def compute_numerical(
        self,
        q: Sequence[float],
        delta: float = 1e-7,
    ) -> np.ndarray:
        """Compute a numerical Jacobian via central finite differences.

        Only the *position* part of the FK output is differentiated:
        ``(x, y)`` for planar robots, ``(x, y, z)`` for 3-D robots.

        Args:
            q: Joint angles in radians.
            delta: Perturbation step size.

        Returns:
            ``(2, n_dof)`` or ``(3, n_dof)`` numerical Jacobian.
        """
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        n_dof = self._robot.n_dof
        task_dim = 2 if self._is_planar else 3

        J = np.zeros((task_dim, n_dof), dtype=np.float64)

        for i in range(n_dof):
            q_plus = q_arr.copy()
            q_minus = q_arr.copy()
            q_plus[i] += delta
            q_minus[i] -= delta

            pos_plus = self._robot.forward_kinematics(q_plus).position
            pos_minus = self._robot.forward_kinematics(q_minus).position

            diff = (pos_plus - pos_minus) / (2.0 * delta)
            J[:, i] = diff[:task_dim]

        return J

    def manipulability(self, q: Sequence[float]) -> float:
        """Yoshikawa manipulability index at configuration *q*.

        Defined as ``sqrt(det(J @ J^T))``.  A value near zero indicates
        proximity to a kinematic singularity.

        Args:
            q: Joint angles in radians.

        Returns:
            Non-negative manipulability measure.
        """
        J = self.compute(q)
        JJT = J @ J.T
        det_val = np.linalg.det(JJT)
        # Clamp to zero to guard against tiny negative floating-point values
        mu = float(np.sqrt(max(det_val, 0.0)))
        logger.debug("Manipulability at q=%s: %.6e", list(q), mu)
        return mu

    def is_singular(
        self,
        q: Sequence[float],
        threshold: float = 1e-4,
    ) -> bool:
        """Check whether the robot is near a kinematic singularity.

        Args:
            q: Joint angles in radians.
            threshold: Manipulability value below which the configuration
                is considered singular.

        Returns:
            ``True`` if the manipulability is below *threshold*.
        """
        mu = self.manipulability(q)
        singular = mu < threshold
        if singular:
            logger.warning(
                "Configuration q=%s is singular (mu=%.6e < %.6e)",
                list(q), mu, threshold,
            )
        else:
            logger.debug(
                "Configuration q=%s is non-singular (mu=%.6e)",
                list(q), mu,
            )
        return singular

    def joint_velocities(
        self,
        q: Sequence[float],
        ee_velocity: Sequence[float],
        damping: float = 0.0,
    ) -> np.ndarray:
        """Compute joint velocities from a desired end-effector velocity.

        Implements resolved-rate motion control: given a desired Cartesian
        velocity ``dx`` at the end-effector, compute the joint velocities
        ``dq = J^+ @ dx`` (or a damped variant for singularity robustness).

        Args:
            q: Current joint angles in radians.
            ee_velocity: Desired end-effector velocity.  For planar robots:
                ``[vx, vy]`` in m/s.  For spatial robots: ``[vx, vy, vz]``
                in m/s (position part only).
            damping: Damping factor λ for the Damped Least Squares inverse.
                ``0.0`` (default) uses the plain Moore-Penrose pseudo-inverse.
                Use ``0.01``–``0.1`` near singularities.

        Returns:
            ``(n_dof,)`` joint velocity array in rad/s.

        Example::

            jc = JacobianComputer(robot)
            dq = jc.joint_velocities([0.5, -0.3], [0.05, 0.0])
        """
        J = self.compute(q)
        dx = np.asarray(ee_velocity, dtype=np.float64).ravel()
        task_dim = J.shape[0]
        if dx.size != task_dim:
            raise ValueError(
                f"ee_velocity must have {task_dim} elements for this robot, "
                f"got {dx.size}"
            )
        if damping == 0.0:
            return np.linalg.pinv(J) @ dx
        # Damped least squares: J^T (J J^T + λ²I)^{-1} dx
        JJT = J @ J.T + (damping ** 2) * np.eye(task_dim, dtype=np.float64)
        return J.T @ np.linalg.solve(JJT, dx)

    def manipulability_gradient(self, q: Sequence[float]) -> np.ndarray:
        """Compute the gradient of the manipulability measure w.r.t. joint angles.

        The gradient ``∂μ/∂q`` (where μ = sqrt(det(JJ^T))) points in the
        joint-space direction that maximally increases manipulability.
        Useful for null-space redundancy resolution: a redundant robot can
        add a component along this gradient to drift toward more dexterous
        configurations while tracking a Cartesian target.

        Args:
            q: Joint angles in radians.

        Returns:
            ``(n_dof,)`` gradient vector.  The sign convention is such that
            moving joints in the direction of the gradient *increases*
            manipulability.
        """
        delta = 1e-5
        q_arr = np.asarray(q, dtype=np.float64).ravel()
        n_dof = self._robot.n_dof
        grad = np.zeros(n_dof, dtype=np.float64)
        for i in range(n_dof):
            q_plus = q_arr.copy()
            q_plus[i] += delta
            q_minus = q_arr.copy()
            q_minus[i] -= delta
            mu_plus = self.manipulability(q_plus)
            mu_minus = self.manipulability(q_minus)
            grad[i] = (mu_plus - mu_minus) / (2.0 * delta)
        return grad

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
