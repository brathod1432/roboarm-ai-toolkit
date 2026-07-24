# Jacobian Matrix

The Jacobian matrix is the fundamental link between joint-space velocities and task-space (Cartesian) velocities. It answers the question: *if I change the joint angles by a small amount, how does the end-effector move?*

## Definition

For an *n*-DOF robot, the Jacobian `J(q)` is a matrix that maps joint velocities to end-effector velocities:

```
dx = J(q) * dq
```

where `dq` is an `(n x 1)` vector of joint velocities and `dx` is the task-space velocity vector. For a full 3-D robot the Jacobian is `6 x n` (3 linear + 3 angular velocity components). For a planar robot the toolkit returns a compact `2 x n` matrix covering only the `(vx, vy)` linear velocities.

## Geometric Jacobian

The **geometric Jacobian** is computed from the kinematic chain using the z-axis cross-product formula. For each revolute joint *i*:

- **Linear velocity column:** `J_v_i = z_{i-1} x (o_n - o_{i-1})`
- **Angular velocity column:** `J_w_i = z_{i-1}`

where:
- `z_{i-1}` is the z-axis of the coordinate frame attached to joint *i-1* (extracted from the cumulative transform: `T[:3, 2]`)
- `o_{i-1}` is the origin of frame *i-1* (`T[:3, 3]`)
- `o_n` is the end-effector origin
- `x` denotes the vector cross product

Stacking all columns produces:

```
J = | z_0 x (o_n - o_0)   z_1 x (o_n - o_1)   ...   z_{n-1} x (o_n - o_{n-1}) |
    |       z_0                   z_1            ...          z_{n-1}             |
```

This is exactly what `JacobianComputer.compute(q)` calculates in [`src/roboarm/kinematics/jacobian.py`](../../src/roboarm/kinematics/jacobian.py).

```python
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.kinematics.jacobian import JacobianComputer

robot = create_two_link_planar()
jc = JacobianComputer(robot)

J = jc.compute([0.5, -0.3])
print(J.shape)  # (2, 2) for a planar robot
print(J)
```

## Numerical Jacobian

When the geometric formulation is inconvenient or you want to verify it, the **numerical Jacobian** uses central finite differences:

```
J[:, i] = (FK(q + delta*e_i) - FK(q - delta*e_i)) / (2 * delta)
```

where `e_i` is the *i*-th unit vector and `delta` is a small perturbation (default `1e-7`).

```python
J_num = jc.compute_numerical([0.5, -0.3], delta=1e-7)
```

The numerical Jacobian only differentiates the position output (`x, y` for planar, `x, y, z` for 3-D), so its shape is `(2, n)` or `(3, n)` rather than the full `(6, n)`.

## Manipulability Index

The **Yoshikawa manipulability index** quantifies how well the robot can move in all task-space directions at a given configuration:

```
mu(q) = sqrt(det(J(q) * J(q)^T))
```

- A **large** value means the robot can move freely in all directions -- good dexterity.
- A value **near zero** means the robot is close to a kinematic singularity and has lost the ability to move in at least one direction.

```python
mu = jc.manipulability([0.5, -0.3])
print(f"Manipulability: {mu:.6f}")
```

## Singularity Detection

A configuration is **singular** when the manipulability drops below a threshold (default `1e-4`). At a singularity:

- The Jacobian loses rank -- some task-space directions become unachievable.
- Jacobian-based IK solvers (pseudoinverse, DLS) may produce very large or erratic joint velocities.
- Physical implications include loss of controllability and potential mechanical lockup.

Common singular configurations include fully extended arms (elbow at 0 or 180 degrees) and aligned wrist axes.

```python
# Fully extended arm (q1=0, q2=0) is near-singular for a 2-link planar arm
is_sing = jc.is_singular([0.0, 0.0])
print(f"Singular: {is_sing}")  # True -- arm fully stretched
```

The `is_singular()` method wraps the manipulability check with a configurable threshold:

```python
is_sing = jc.is_singular([0.5, -0.3], threshold=1e-3)
```

## Planar vs. Spatial Robots

`JacobianComputer` automatically detects whether a robot is planar (all `alpha == 0` and `d == 0`) and adjusts its output accordingly:

| Robot type | `compute()` shape | `compute_numerical()` shape |
|------------|-------------------|-----------------------------|
| Planar     | `(2, n_dof)` -- `vx, vy` only | `(2, n_dof)` |
| Spatial    | `(6, n_dof)` -- `vx, vy, vz, wx, wy, wz` | `(3, n_dof)` -- position only |

## Full Example

```python
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.kinematics.jacobian import JacobianComputer
import numpy as np

robot = create_two_link_planar(link1=1.0, link2=0.8)
jc = JacobianComputer(robot)

q = [0.7, -0.4]

# Geometric Jacobian
J = jc.compute(q)
print("Jacobian:\n", np.round(J, 4))

# Manipulability
mu = jc.manipulability(q)
print(f"Manipulability: {mu:.6f}")

# Singularity check
print(f"Near singularity: {jc.is_singular(q)}")

# Verify against numerical Jacobian
J_num = jc.compute_numerical(q)
print(f"Max difference from numerical: {np.max(np.abs(J - J_num)):.2e}")
```

---

## See Also

- [Forward Kinematics](forward_kinematics.md) -- the FK computation that the Jacobian differentiates
- [Inverse Kinematics](inverse_kinematics.md) -- IK solvers that use the Jacobian (pseudoinverse, DLS)
- [DH Parameters](dh_parameters.md) -- the kinematic model underlying the Jacobian
- [`src/roboarm/kinematics/jacobian.py`](../../src/roboarm/kinematics/jacobian.py) -- `JacobianComputer` implementation
- [`src/roboarm/core/robot.py`](../../src/roboarm/core/robot.py) -- `joint_transforms()` used internally by the Jacobian
