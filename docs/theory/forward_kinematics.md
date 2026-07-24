# Forward Kinematics

Forward kinematics (FK) answers the fundamental question: *given a set of joint angles, where is the end-effector?* It maps joint-space coordinates to a Cartesian pose (position and orientation) by chaining the DH transformation matrices of every link in the serial chain.

## Definition

For an *n*-joint serial robot, forward kinematics computes the end-effector pose as:

```
T_0n = T_01(q1) * T_12(q2) * ... * T_{n-1,n}(qn)
```

where each `T_{i-1,i}(qi)` is the 4x4 homogeneous transform built from the DH parameters of link *i* with the joint variable `qi` added to `theta` (for revolute joints). The final matrix `T_0n` encodes:

- **Position**: the 3-element translation vector `[x, y, z]` in the last column (`T_0n[:3, 3]`).
- **Orientation**: the 3x3 rotation sub-matrix (`T_0n[:3, :3]`).

This chain multiplication is implemented in `chain_transforms()` from [`src/roboarm/core/transform.py`](../../src/roboarm/core/transform.py) and exposed through `RobotArm.forward_kinematics()` in [`src/roboarm/core/robot.py`](../../src/roboarm/core/robot.py).

## 2-Link Planar Example

Consider a planar arm with link lengths `L1 = 1.0` and `L2 = 1.0` (standard DH):

| Link | alpha | a   | d | theta |
|------|-------|-----|---|-------|
| 1    | 0     | 1.0 | 0 | q1    |
| 2    | 0     | 1.0 | 0 | q2    |

Multiplying the two transforms yields:

```
x = L1 * cos(q1) + L2 * cos(q1 + q2)
y = L1 * sin(q1) + L2 * sin(q1 + q2)
```

Since both `alpha` and `d` are zero, the arm operates entirely in the XY plane and z is always zero.

```python
from roboarm.robots.two_link_planar import create_two_link_planar

robot = create_two_link_planar(link1=1.0, link2=1.0)
pose = robot.forward_kinematics([0.5, -0.3])

print(f"x = {pose.x:.4f}")  # x = 1.4553
print(f"y = {pose.y:.4f}")  # y = 0.2739
```

## 6-DOF Modified DH Example

The toolkit includes a 6-DOF arm defined with modified DH parameters. Its DH table is specified in [`src/roboarm/robots/six_dof_mdh.py`](../../src/roboarm/robots/six_dof_mdh.py):

| Link  | alpha_{i-1} (deg) | a_{i-1} (cm) | theta_i | d_i (cm) |
|-------|--------------------|--------------|---------|----------|
| 0->1  | 0   | 0    | q1 | 0    |
| 1->2  | 90  | 0    | q2 | 0    |
| 2->3  | 0   | 15   | q3 | 0    |
| 3->4  | 90  | 7.2  | q4 | 0    |
| 4->5  | 90  | 0    | q5 | 13.2 |
| 5->6  | 90  | 0    | q6 | 3    |
| 6->TCP| 0   | 0    | 0 (fixed) | 7.5 |

The arm has 6 variable joints plus a fixed TCP offset (7 links total, 6 DOF). Its home pose is `[0, 90, 0, 0, 180, 0]` degrees.

```python
import math
from roboarm.robots.six_dof_mdh import create_six_dof_mdh, HOME_POSE_RAD

robot = create_six_dof_mdh()
pose = robot.forward_kinematics(HOME_POSE_RAD)

print(f"End-effector: ({pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f})")
```

## How Chain Multiplication Works

Internally, `RobotArm.forward_kinematics()` performs these steps:

1. **Build per-link transforms** -- for each `JointConfig` in the chain, call `transform_from_dh_params(dh, q_i)` to get the 4x4 matrix. The function adds the joint variable to `theta` and dispatches to `dh_transform()` or `mdh_transform()` based on the `convention` field.

2. **Accumulate** -- multiply the per-link matrices left-to-right via `chain_transforms()`:
   ```python
   result = np.eye(4)
   for T_i in link_transforms:
       result = result @ T_i
   ```

3. **Extract pose** -- return an `EndEffectorPose` containing the position vector, rotation matrix, and the full 4x4 transform.

## Intermediate Joint Frames

In addition to the final end-effector pose, you can retrieve every intermediate frame using `robot.joint_transforms(q)`. This returns a list of `n_joints + 1` cumulative 4x4 matrices, from the base identity to the tool tip:

```python
frames = robot.joint_transforms([0.5, -0.3])
# frames[0] = identity (base)
# frames[1] = T_01
# frames[2] = T_02 (= T_01 @ T_12) -- end-effector for 2-link arm
```

For extracting only positions (useful for plotting), use `robot.joint_positions(q)` which returns an `(n_joints + 1, 3)` array.

## Pre-Built Robot Factories

The toolkit provides ready-made robot constructors in [`src/roboarm/robots/`](../../src/roboarm/robots/):

| Factory function | File | DOF | Convention |
|------------------|------|-----|------------|
| `create_two_link_planar()` | [`two_link_planar.py`](../../src/roboarm/robots/two_link_planar.py) | 2 | Standard |
| `create_three_link_planar()` | [`three_link_planar.py`](../../src/roboarm/robots/three_link_planar.py) | 3 | Standard |
| `create_six_dof_mdh()` | [`six_dof_mdh.py`](../../src/roboarm/robots/six_dof_mdh.py) | 6 | Modified |

Each factory returns a `RobotArm` instance ready for FK computation.

---

## See Also

- [DH Parameters](dh_parameters.md) -- theory behind the 4x4 link transforms
- [Inverse Kinematics](inverse_kinematics.md) -- computing joint angles from a desired pose
- [Jacobian Matrix](jacobian.md) -- velocity-level kinematics derived from FK
- [`src/roboarm/core/robot.py`](../../src/roboarm/core/robot.py) -- `RobotArm` class with `forward_kinematics()` method
- [`src/roboarm/core/transform.py`](../../src/roboarm/core/transform.py) -- DH transform builders and chain multiplication
- [`src/roboarm/robots/`](../../src/roboarm/robots/) -- pre-built robot configurations
