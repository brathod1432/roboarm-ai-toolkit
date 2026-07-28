# Trajectory Planning

Trajectory planning answers the question: *given a start configuration and a goal configuration, how should the joints move through time to reach the goal smoothly?* A good trajectory respects the physics of the arm — joints should not jump discontinuously, velocities should not spike, and the motion should feel natural rather than jerky.

This toolkit provides two families of trajectory generators in [`src/roboarm/trajectory/`](../../src/roboarm/trajectory/):

1. **Polynomial interpolation** (`interpolation.py`) — generates smooth paths between two configurations using 1st-, 3rd-, or 5th-degree polynomials.
2. **LSPB — Linear Segment with Parabolic Blend** (`lspb.py`) — generates trapezoidal velocity profiles, the standard motion primitive used in industrial robot controllers.

Both families work in **joint space**: the inputs and outputs are joint angles, not Cartesian poses.

---

## Polynomial Interpolation

Polynomial interpolation parameterises a joint trajectory as a function of a normalised time variable `s ∈ [0, 1]`, where `s=0` is the start and `s=1` is the goal. Each family member trades off boundary conditions against smoothness.

All three functions in `interpolation.py` share the same signature:

```python
trajectory = interpolation_function(q_start, q_end, n_steps=50)
# Returns: (n_steps, n_dof) array, one row per time step
```

### Linear Interpolation

The simplest scheme — joint angle changes at a constant rate:

```
q(s) = q0 + s * (q1 - q0)
```

**Boundary conditions:** position continuity only. Velocity is discontinuous at the start and end (the joint "snaps" from rest to full speed instantly).

```python
from roboarm.trajectory.interpolation import linear_interpolation

q_start = [0.0, 0.0]
q_end   = [1.0, -0.5]

traj = linear_interpolation(q_start, q_end, n_steps=50)
# traj.shape == (50, 2)
# traj[0]  == [0.0, 0.0]   — start
# traj[-1] == [1.0, -0.5]  — end
```

Use linear interpolation only when motion smoothness is not important, for example when checking reachability across a coarse grid.

### Cubic Interpolation

A third-degree polynomial that enforces zero velocity at both endpoints:

```
q(s) = q0 + (3s^2 - 2s^3) * (q1 - q0)
```

**Boundary conditions:** position and velocity continuity (velocity is zero at start and end). The resulting motion accelerates from rest, reaches peak speed at the midpoint, then decelerates symmetrically — sometimes called an *S-curve* profile.

```python
from roboarm.trajectory.interpolation import cubic_interpolation

traj = cubic_interpolation(q_start, q_end, n_steps=100)
```

Cubic interpolation is the right default for most point-to-point movements where the arm starts and stops at rest.

### Quintic Interpolation

A fifth-degree polynomial that additionally enforces zero acceleration at both endpoints:

```
q(s) = q0 + (10s^3 - 15s^4 + 6s^5) * (q1 - q0)
```

**Boundary conditions:** position, velocity, and acceleration continuity (all zero at start and end). This produces the smoothest possible single-segment motion and is appropriate when the trajectory feeds into a dynamics controller that is sensitive to torque spikes.

```python
from roboarm.trajectory.interpolation import quintic_interpolation

traj = quintic_interpolation(q_start, q_end, n_steps=100)
```

### Comparison

| Method | Polynomial degree | Zero velocity at ends | Zero acceleration at ends | Typical use |
|--------|------------------|-----------------------|---------------------------|-------------|
| Linear | 1 | No | No | Debugging, coarse motion |
| Cubic | 3 | Yes | No | Standard point-to-point |
| Quintic | 5 | Yes | Yes | Dynamics-sensitive or chained moves |

---

## LSPB — Trapezoidal Velocity Profile

LSPB is the standard trajectory primitive in industrial robot controllers. Instead of specifying the trajectory as a time polynomial, you specify a maximum cruise velocity and a total duration. The generator then computes three phases automatically:

```
Phase 1 (acceleration):  parabolic blend from rest to v_max
Phase 2 (cruise):        linear segment at constant v_max
Phase 3 (deceleration):  parabolic blend from v_max back to rest
```

The parabolic blends ensure that velocity is continuous everywhere, and the constant-acceleration design makes the motion easy to reason about mechanically.

### Single-Joint LSPB

```python
from roboarm.trajectory.lspb import lspb

positions, velocities, times = lspb(
    q0=0.0,       # start angle (rad)
    qf=1.5,       # goal angle (rad)
    t_total=2.0,  # total duration (s)
    v_max=None,   # auto: 1.5 * delta_q / t_total
    n_steps=100,
)
# positions.shape == velocities.shape == times.shape == (100,)
```

When `v_max=None` (the default), the cruise velocity is set to `1.5 * (qf - q0) / t_total`, which allocates one third of the total time to acceleration and one third to deceleration. You can override this with any feasible value.

**Feasibility constraint:** `v_max` must be large enough that the constant-velocity segment has positive duration. If not, a `ValueError` is raised:

```python
# This will raise ValueError — v_max too slow to complete the move in t_total
lspb(q0=0.0, qf=1.5, t_total=2.0, v_max=0.5)
```

### Multi-Joint LSPB

For a robot arm, all joints must finish at the same time. `multi_joint_lspb` runs an independent LSPB for each joint over a shared total duration:

```python
from roboarm.trajectory.lspb import multi_joint_lspb

traj = multi_joint_lspb(
    q_start=[0.0, 0.0],
    q_end=[1.5, -0.5],
    t_total=2.0,
    v_max=None,   # auto per joint
    n_steps=100,
)
# traj.shape == (100, 2)
```

Each joint reaches its goal position at the same final time, but the cruise velocity and blend duration may differ per joint (because the total angular displacement differs).

---

## Choosing a Method

| Criterion | Recommended method |
|-----------|--------------------|
| Maximum smoothness (zero velocity + acceleration at endpoints) | Quintic interpolation |
| Standard point-to-point, minimal overshoot | Cubic interpolation or LSPB |
| Known maximum velocity, industrial-style profile | LSPB |
| Fastest to compute, no smoothness needed | Linear interpolation |
| Chained multi-point trajectory | Quintic between each pair of waypoints |

---

## Full Example

```python
import numpy as np
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.trajectory.interpolation import cubic_interpolation
from roboarm.trajectory.lspb import multi_joint_lspb

robot = create_two_link_planar()

q_start = [0.0, 0.0]
q_end   = [1.0, -0.5]
n_steps = 100

# --- Cubic interpolation ---
traj_cubic = cubic_interpolation(q_start, q_end, n_steps)

# Compute FK at each waypoint
print("Cubic trajectory — end-effector path:")
for i in [0, 25, 50, 75, 99]:
    pose = robot.forward_kinematics(traj_cubic[i])
    print(f"  step {i:3d}: x={pose.x:.4f}, y={pose.y:.4f}")

# --- LSPB ---
traj_lspb = multi_joint_lspb(q_start, q_end, t_total=2.0, n_steps=n_steps)

print("\nLSPB trajectory — joint 1 profile (first 5 steps):")
for i in range(5):
    print(f"  t={i*2.0/(n_steps-1):.3f}s  q1={traj_lspb[i, 0]:.4f} rad")
```

---

## See Also

- [Forward Kinematics](forward_kinematics.md) — applying FK along a trajectory to get the Cartesian path
- [Inverse Kinematics](inverse_kinematics.md) — planning in joint space from task-space waypoints
- [`src/roboarm/trajectory/interpolation.py`](../../src/roboarm/trajectory/interpolation.py) — linear, cubic, quintic implementations
- [`src/roboarm/trajectory/lspb.py`](../../src/roboarm/trajectory/lspb.py) — LSPB single-joint and multi-joint implementations
