# Denavit-Hartenberg Parameters

Denavit-Hartenberg (DH) parameters are a minimal set of four numbers that describe the geometric relationship between consecutive links in a serial-link robot arm. Every rigid connection between two joint axes can be fully characterised by these four values, making DH parameters the standard language for encoding robot kinematics.

## The Four Parameters

Each link *i* in the kinematic chain is described by:

| Symbol | Name | Description |
|--------|------|-------------|
| `alpha` | Link twist | Angle between the z-axes of frames *i-1* and *i*, measured about the common perpendicular (x-axis). |
| `a` | Link length | Distance between the z-axes of frames *i-1* and *i*, measured along the common perpendicular. |
| `d` | Link offset | Distance between the x-axes of frames *i-1* and *i*, measured along the z-axis of frame *i-1*. For prismatic joints this is the variable. |
| `theta` | Joint angle | Angle between the x-axes of frames *i-1* and *i*, measured about the z-axis of frame *i-1*. For revolute joints this is the variable. |

In this toolkit, these parameters are stored in the `DHParams` dataclass:

```python
from roboarm.core.types import DHParams

link = DHParams(alpha=0.0, a=1.0, d=0.0, theta=0.0, convention="standard")
```

## Standard DH Convention

The **standard** (classical) DH convention builds the 4x4 homogeneous transformation matrix for link *i* as:

```
T_i = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)
```

Each primitive is a rotation or translation about/along a single axis. When multiplied out, the resulting 4x4 matrix is:

```
T = | cos(th)   -sin(th)*cos(al)    sin(th)*sin(al)   a*cos(th) |
    | sin(th)    cos(th)*cos(al)   -cos(th)*sin(al)   a*sin(th) |
    |   0            sin(al)            cos(al)            d     |
    |   0              0                  0                1     |
```

where `th = theta` and `al = alpha`.

This is implemented in `dh_transform()` within [`src/roboarm/core/transform.py`](../../src/roboarm/core/transform.py).

## Modified DH (Craig) Convention

The **modified** DH convention, introduced by John Craig, attaches the coordinate frame to the *preceding* link rather than the current one. The transformation order becomes:

```
T_i = Rx(alpha_{i-1}) * Tx(a_{i-1}) * Rz(theta_i) * Tz(d_i)
```

The resulting 4x4 matrix is:

```
T = | cos(th)          -sin(th)          0       a           |
    | sin(th)*cos(al)   cos(th)*cos(al)  -sin(al)  -d*sin(al) |
    | sin(th)*sin(al)   cos(th)*sin(al)   cos(al)   d*cos(al) |
    |   0                  0                0         1        |
```

where `th = theta_i` and `al = alpha_{i-1}`.

This is implemented in `mdh_transform()` within [`src/roboarm/core/transform.py`](../../src/roboarm/core/transform.py).

## When to Use Each Convention

| Criterion | Standard DH | Modified DH (Craig) |
|-----------|-------------|---------------------|
| Frame attachment | Frame *i* is on the *distal* end of link *i* | Frame *i* is on the *proximal* end of link *i* |
| Base frame | May not coincide with joint 1 | Naturally coincides with joint 1 |
| Common usage | Textbook robotics (Spong, Siciliano) | Craig's *Introduction to Robotics*; many industrial robots |
| Toolkit default | `convention="standard"` | `convention="modified"` |

Choose **standard** DH when following classical textbooks or when existing DH tables use that convention. Choose **modified** DH when modelling industrial arms (the 6-DOF robot in this toolkit uses modified DH) or when you want the base frame aligned with joint 1.

You select the convention when constructing `DHParams`:

```python
# Standard convention (default)
standard_link = DHParams(alpha=0.0, a=1.0, d=0.0, theta=0.0, convention="standard")

# Modified (Craig) convention
modified_link = DHParams(alpha=1.5708, a=0.0, d=0.0, theta=0.0, convention="modified")
```

The function `transform_from_dh_params()` in [`src/roboarm/core/transform.py`](../../src/roboarm/core/transform.py) automatically dispatches to the correct matrix builder based on `params.convention`.

## Example: 2-Link Planar Arm

A simple planar arm with two revolute joints and link lengths `L1` and `L2` uses standard DH:

| Link | alpha | a   | d | theta |
|------|-------|-----|---|-------|
| 1    | 0     | L1  | 0 | q1    |
| 2    | 0     | L2  | 0 | q2    |

Since `alpha = 0` and `d = 0` for all links, the z-axes of every frame point out of the plane and the arm moves entirely in the XY plane.

## Example: 6-DOF Arm (Modified DH)

The 6-DOF robot in this toolkit uses the modified DH convention. See [`src/roboarm/robots/six_dof_mdh.py`](../../src/roboarm/robots/six_dof_mdh.py) for the full parameter table. An excerpt:

| Link | alpha_{i-1} (deg) | a_{i-1} (cm) | theta_i (deg) | d_i (cm) |
|------|--------------------|--------------|---------------|----------|
| 0->1 | 0   | 0    | q1 | 0    |
| 1->2 | 90  | 0    | q2 | 0    |
| 2->3 | 0   | 15   | q3 | 0    |
| 3->4 | 90  | 7.2  | q4 | 0    |
| 4->5 | 90  | 0    | q5 | 13.2 |
| 5->6 | 90  | 0    | q6 | 3    |

Note that non-zero `alpha` and `d` values create a spatial (3-D) kinematic chain, unlike the planar example above.

## Implementation Reference

The core transform functions live in [`src/roboarm/core/transform.py`](../../src/roboarm/core/transform.py):

- `dh_transform(alpha, a, theta, d)` -- standard DH 4x4 matrix
- `mdh_transform(alpha, a, theta, d)` -- modified DH 4x4 matrix
- `transform_from_dh_params(params, q)` -- auto-dispatching wrapper that adds the joint variable `q` to `theta`
- `chain_transforms(transforms)` -- multiplies a list of 4x4 matrices left-to-right
- `is_valid_transform(T)` -- validates SE(3) membership (orthonormal rotation, proper bottom row)

---

## See Also

- [Forward Kinematics](forward_kinematics.md) -- how DH transforms are chained to compute end-effector pose
- [Inverse Kinematics](inverse_kinematics.md) -- recovering joint angles from a desired pose
- [`src/roboarm/core/transform.py`](../../src/roboarm/core/transform.py) -- implementation of DH and MDH transforms
- [`src/roboarm/core/types.py`](../../src/roboarm/core/types.py) -- `DHParams`, `JointConfig`, and related dataclasses
