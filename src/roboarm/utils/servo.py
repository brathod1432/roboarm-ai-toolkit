"""Servo PWM mapping utilities for hobbyist hardware integration.

Provides :class:`ServoConfig` for mapping robot joint angles (radians) to
PWM microsecond values suitable for standard RC servos, and
:class:`ServoChain` for managing multiple servos at once.

Typical usage with an Arduino/Raspberry Pi servo driver::

    servo = ServoConfig(
        zero_offset_rad=0.0,
        scale_us_per_rad=500.0,    # 500 μs/rad → ±π gives ±1571 μs
        center_us=1500,
        min_us=500,
        max_us=2500,
    )
    pwm = servo.angle_to_pwm(0.785)   # π/4 → ~1893 μs
    angle = servo.pwm_to_angle(1893)  # back to ~0.786 rad
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ServoConfig:
    """Calibration parameters for a single RC servo.

    Maps joint angles (radians) to PWM pulse widths (microseconds) using
    a linear model:

    .. code-block::

        pwm_us = center_us + (angle_rad - zero_offset_rad) * scale_us_per_rad

    The result is clamped to ``[min_us, max_us]`` before output.

    Attributes:
        zero_offset_rad: Joint angle (radians) that corresponds to the
            servo's centre position (*center_us*).  Typically 0.
        scale_us_per_rad: Microseconds per radian.  Positive means larger
            angle → larger PWM (clockwise).  Negative reverses the
            direction.  A common value is ``500 μs/rad`` (which maps
            ±180° to ±1571 μs around centre).
        center_us: PWM value (μs) at the zero/neutral position.
            Typically 1500 for standard servos.
        min_us: Lower PWM limit (μs).  Pulses below this are clamped.
            Default 500.
        max_us: Upper PWM limit (μs).  Pulses above this are clamped.
            Default 2500.
    """

    zero_offset_rad: float = 0.0
    scale_us_per_rad: float = 500.0
    center_us: int = 1500
    min_us: int = 500
    max_us: int = 2500

    def __post_init__(self) -> None:
        if self.min_us >= self.max_us:
            raise ValueError(
                f"min_us ({self.min_us}) must be less than max_us ({self.max_us})"
            )
        if self.scale_us_per_rad == 0.0:
            raise ValueError("scale_us_per_rad must be non-zero")

    def angle_to_pwm(self, angle_rad: float) -> int:
        """Convert a joint angle to a PWM pulse width.

        Args:
            angle_rad: Joint angle in radians.

        Returns:
            PWM pulse width in microseconds, clamped to ``[min_us, max_us]``.
        """
        raw = self.center_us + (angle_rad - self.zero_offset_rad) * self.scale_us_per_rad
        return int(max(self.min_us, min(self.max_us, round(raw))))

    def pwm_to_angle(self, pwm_us: int) -> float:
        """Convert a PWM pulse width back to a joint angle.

        Args:
            pwm_us: PWM pulse width in microseconds.

        Returns:
            Joint angle in radians.
        """
        return (pwm_us - self.center_us) / self.scale_us_per_rad + self.zero_offset_rad

    @property
    def angle_range_rad(self) -> tuple[float, float]:
        """The joint angle range reachable within ``[min_us, max_us]``."""
        lo = self.pwm_to_angle(self.min_us)
        hi = self.pwm_to_angle(self.max_us)
        return (min(lo, hi), max(lo, hi))


@dataclass
class ServoChain:
    """A collection of :class:`ServoConfig` objects for a multi-joint arm.

    Provides batch angle-to-PWM conversion for sending joint angles to a
    multi-channel servo driver (e.g. PCA9685).

    Attributes:
        servos: Ordered list of :class:`ServoConfig`, one per joint.
    """

    servos: list[ServoConfig]

    def angles_to_pwm(self, angles_rad: np.ndarray | list[float]) -> list[int]:
        """Convert joint angles to PWM values for all servos.

        Args:
            angles_rad: Sequence of joint angles in radians.  Length must
                equal the number of servos.

        Returns:
            List of integer PWM values, one per servo.

        Raises:
            ValueError: If the length of *angles_rad* does not match the
                number of servos.
        """
        angles = list(angles_rad)
        if len(angles) != len(self.servos):
            raise ValueError(
                f"Expected {len(self.servos)} angles, got {len(angles)}"
            )
        return [servo.angle_to_pwm(a) for servo, a in zip(self.servos, angles)]

    def pwm_to_angles(self, pwm_values: list[int]) -> list[float]:
        """Convert PWM values back to joint angles for all servos.

        Args:
            pwm_values: List of integer PWM values.

        Returns:
            List of joint angles in radians.
        """
        if len(pwm_values) != len(self.servos):
            raise ValueError(
                f"Expected {len(self.servos)} PWM values, got {len(pwm_values)}"
            )
        return [servo.pwm_to_angle(p) for servo, p in zip(self.servos, pwm_values)]
