"""Trajectory file I/O — CSV and NumPy NPZ export/import.

Lets clients save trajectories to disk and reload them for replay,
sharing, or analysis in external tools (MATLAB, Excel, ROS, Arduino).

CSV format (human-readable, universally importable)
---------------------------------------------------
Header row: ``time_s, J1_rad, J2_rad, ..., x_m, y_m, z_m``
One data row per waypoint.  The FK columns (x, y, z) are optional and
included only when a *robot* is supplied to :func:`save_trajectory_csv`.

NPZ format (compact, fast, lossless)
--------------------------------------
A NumPy ``.npz`` archive with arrays ``trajectory``, ``timestamps``,
and a ``metadata`` dict encoded as JSON.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Characters that trigger formula execution in spreadsheet software
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _safe_csv_cell(value: object) -> object:
    """Protect against CSV formula injection attacks.

    Spreadsheet applications (Excel, LibreOffice, Google Sheets) execute
    formulas in cells that start with ``=``, ``+``, ``-``, or ``@``.  A
    maliciously crafted joint name like ``=HYPERLINK(...)`` could exfiltrate
    data when the CSV is opened.

    This function prepends a tab character to any string cell that starts with
    a formula trigger character, which is the OWASP-recommended mitigation and
    is harmless for all legitimate robot parameter names.

    Args:
        value: The cell value to protect.

    Returns:
        The original value if it is not a string or does not start with a
        formula prefix; otherwise a prefixed safe string.
    """
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "\t" + value
    return value


def save_trajectory_csv(
    path: str | Path,
    trajectory: np.ndarray,
    timestamps: Sequence[float] | None = None,
    joint_names: Sequence[str] | None = None,
    robot: object | None = None,
) -> None:
    """Save a joint-space trajectory to a CSV file.

    Args:
        path: Output file path (created or overwritten).
        trajectory: ``(n_steps, n_dof)`` joint angle array in radians.
        timestamps: Optional 1-D sequence of time values in seconds
            (length == n_steps).  If ``None``, integer step indices are
            used.
        joint_names: Column headers for joint columns.  Defaults to
            ``["J1_rad", "J2_rad", ...]``.
        robot: Optional :class:`~roboarm.core.robot.RobotArm`.  When
            supplied, FK is computed at each waypoint and the Cartesian
            position is appended as extra columns (``x_m``, ``y_m``,
            ``z_m``).

    Example::

        import numpy as np
        from roboarm.trajectory.io import save_trajectory_csv

        t = np.linspace(0, 2.0, 50)
        traj = np.zeros((50, 2))
        save_trajectory_csv("motion.csv", traj, timestamps=t)
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2:
        raise ValueError(f"trajectory must be 2-D, got shape {traj.shape}")
    n_steps, n_dof = traj.shape

    if timestamps is None:
        t_arr = np.arange(n_steps, dtype=np.float64)
    else:
        t_arr = np.asarray(timestamps, dtype=np.float64).ravel()

    if t_arr.size != n_steps:
        raise ValueError(
            f"timestamps length {t_arr.size} does not match "
            f"trajectory length {n_steps}"
        )

    if joint_names is None:
        jnames = [f"J{i + 1}_rad" for i in range(n_dof)]
    else:
        jnames = list(joint_names)

    header = ["time_s"] + jnames
    include_fk = robot is not None
    if include_fk:
        header += ["x_m", "y_m", "z_m"]

    out = Path(path)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([_safe_csv_cell(h) for h in header])
        for i in range(n_steps):
            row: list[float] = [t_arr[i]] + traj[i].tolist()
            if include_fk:
                pose = robot.forward_kinematics(traj[i])  # type: ignore[union-attr]
                row += [pose.x, pose.y, pose.z]
            writer.writerow(row)

    logger.info("Trajectory saved to %s (%d steps, %d DOF)", out, n_steps, n_dof)


def load_trajectory_csv(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a trajectory from a CSV file.

    Args:
        path: Path to a CSV file produced by :func:`save_trajectory_csv`.

    Returns:
        Tuple ``(trajectory, timestamps, metadata)`` where:

        * ``trajectory`` — ``(n_steps, n_dof)`` joint angle array
        * ``timestamps`` — ``(n_steps,)`` time array
        * ``metadata`` — dict with ``joint_names`` and optionally
          ``cartesian_columns`` keys
    """
    src = Path(path)
    rows: list[list[str]] = []
    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    data = np.array([[float(v) for v in row] for row in rows], dtype=np.float64)
    timestamps = data[:, 0]

    # Identify joint columns (not time, not cartesian)
    cart_cols = {"x_m", "y_m", "z_m"}
    joint_cols = [h for h in header[1:] if h not in cart_cols]
    joint_indices = [header.index(h) for h in joint_cols]

    trajectory = data[:, joint_indices]

    metadata: dict[str, Any] = {"joint_names": joint_cols}
    if any(h in header for h in cart_cols):
        metadata["cartesian_columns"] = [h for h in header if h in cart_cols]

    logger.info("Trajectory loaded from %s (%d steps)", src, len(rows))
    return trajectory, timestamps, metadata


def save_trajectory_npz(
    path: str | Path,
    trajectory: np.ndarray,
    timestamps: Sequence[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a trajectory to a compressed NumPy .npz file.

    Args:
        path: Output file path (the ``.npz`` extension is added if absent).
        trajectory: ``(n_steps, n_dof)`` joint angle array.
        timestamps: Optional time array in seconds.
        metadata: Optional dict of extra fields (serialised as JSON).

    Example::

        save_trajectory_npz("motion.npz", traj, timestamps=t,
                            metadata={"robot": "2-link", "solver": "dls"})
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    n_steps = traj.shape[0]
    t_arr = (
        np.asarray(timestamps, dtype=np.float64).ravel()
        if timestamps is not None
        else np.arange(n_steps, dtype=np.float64)
    )
    meta_json = json.dumps(metadata or {})
    # Normalize: strip trailing .npz if present, then np.savez_compressed adds it
    path_str = str(path)
    if path_str.endswith(".npz"):
        path_str = path_str[:-4]
    np.savez_compressed(
        path_str,
        trajectory=traj,
        timestamps=t_arr,
        metadata_json=np.array(meta_json),
    )
    logger.info("Trajectory saved to %s.npz (%d steps)", path_str, n_steps)


def load_trajectory_npz(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a trajectory from a .npz file.

    Args:
        path: Path produced by :func:`save_trajectory_npz`.

    Returns:
        ``(trajectory, timestamps, metadata)`` tuple.
    """
    path_str = str(path)
    if not path_str.endswith(".npz"):
        path_str = path_str + ".npz"
    archive = np.load(path_str, allow_pickle=False)
    traj = archive["trajectory"]
    timestamps = archive["timestamps"]
    meta_json = str(archive["metadata_json"])
    metadata: dict[str, Any] = json.loads(meta_json)
    logger.info("Trajectory loaded from %s (%d steps)", path_str, len(traj))
    return traj, timestamps, metadata
