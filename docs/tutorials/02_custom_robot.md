# Building a Custom Robot

This tutorial shows you how to define a robot arm from scratch using DH parameters, assemble it into a `RobotArm`, and verify it with forward kinematics. By the end you will have a custom 4-DOF arm ready for IK solving and visualisation.

## Core Building Blocks

Every robot in this toolkit is assembled from three dataclasses defined in [`src/roboarm/core/types.py`](../../src/roboarm/core/types.py):

1. **`DHParams`** -- the four DH parameters (`alpha`, `a`, `d`, `theta`) plus the convention (`"standard"` or `"modified"`).
2. **`JointConfig`** -- wraps `DHParams` with optional mechanical limits, a human-readable name, and a flag indicating whether the joint is actuated.
3. **`JointLimits`** -- lower and upper angle bounds (radians), with optional velocity and acceleration limits.

## Step 1: Define the DH Parameters

Suppose you want to build a 4-DOF arm with the following standard DH table:

| Link | alpha (rad) | a (m) | d (m) | theta |
|------|-------------|-------|-------|-------|
| 1    | 0           | 0.4   | 0     | q1    |
| 2    | 0           | 0.3   | 0     | q2    |
| 3    | 0           | 0.2   | 0     | q3    |
| 4    | 0           | 0.1   | 0     | q4    |

This is a 4-link planar arm (all `alpha = 0`, all `d = 0`), where each link gets progressively shorter.

```python
from roboarm.core.types import DHParams

dh_link1 = DHParams(alpha=0.0, a=0.4, d=0.0, theta=0.0, convention="standard")
dh_link2 = DHParams(alpha=0.0, a=0.3, d=0.0, theta=0.0, convention="standard")
dh_link3 = DHParams(alpha=0.0, a=0.2, d=0.0, theta=0.0, convention="standard")
dh_link4 = DHParams(alpha=0.0, a=0.1, d=0.0, theta=0.0, convention="standard")
```

The `theta` field is set to `0.0` because for revolute joints the actual angle is added at runtime by the FK engine. If you had a fixed angular offset on a joint, you would put it here.

## Step 2: Add Joint Limits

Define mechanical limits for each joint. These are used for validation and visualisation, and can be checked during IK solving:

```python
import math
from roboarm.core.types import JointLimits

limits_wide = JointLimits(lower=-math.pi, upper=math.pi)
limits_narrow = JointLimits(lower=-math.pi / 2, upper=math.pi / 2)
```

## Step 3: Create Joint Configurations

Combine DH parameters, limits, and metadata into `JointConfig` objects:

```python
from roboarm.core.types import JointConfig

joints = [
    JointConfig(dh_params=dh_link1, limits=limits_wide, name="Base"),
    JointConfig(dh_params=dh_link2, limits=limits_wide, name="Shoulder"),
    JointConfig(dh_params=dh_link3, limits=limits_narrow, name="Elbow"),
    JointConfig(dh_params=dh_link4, limits=limits_narrow, name="Wrist"),
]
```

Each `JointConfig` defaults to `is_variable=True`, meaning the joint angle is actuated. Set `is_variable=False` for fixed offsets (like a tool-centre-point link):

```python
tcp = JointConfig(
    dh_params=DHParams(alpha=0.0, a=0.05, d=0.0, theta=0.0),
    name="TCP",
    is_variable=False,
)
```

## Step 4: Assemble the Robot

Pass the ordered list of joints to `RobotArm`:

```python
from roboarm.core.robot import RobotArm

robot = RobotArm(joints, name="Custom 4-DOF Planar")
print(robot)
# RobotArm(name='Custom 4-DOF Planar', n_joints=4, n_dof=4)

print(f"DOF: {robot.n_dof}")
print(f"Joint names: {robot.joint_names}")
```

## Step 5: Test with Forward Kinematics

Verify the robot by computing FK at a known configuration:

```python
# All joints at zero -- arm fully extended along X
pose_zero = robot.forward_kinematics([0.0, 0.0, 0.0, 0.0])
print(f"Fully extended: ({pose_zero.x:.4f}, {pose_zero.y:.4f})")
# Expected: x = 0.4 + 0.3 + 0.2 + 0.1 = 1.0, y = 0.0

# Apply some joint angles
pose = robot.forward_kinematics([0.3, -0.5, 0.8, -0.2])
print(f"Pose: ({pose.x:.4f}, {pose.y:.4f})")
```

You can also inspect intermediate joint positions:

```python
positions = robot.joint_positions([0.3, -0.5, 0.8, -0.2])
for i, pos in enumerate(positions):
    print(f"  Frame {i}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
```

## Choosing Standard vs. Modified DH

The DH convention is set per-link via the `convention` field. You can mix conventions if needed, but in practice you should use one consistently:

- **Standard DH** (`convention="standard"`): simpler for planar arms and textbook examples. Frame *i* sits at the distal end of link *i*. Used by the 2-link and 3-link planar robots.
- **Modified DH** (`convention="modified"`): frame *i* sits at the proximal end. Preferred for industrial robots. Used by the 6-DOF arm in this toolkit.

See [`src/roboarm/robots/six_dof_mdh.py`](../../src/roboarm/robots/six_dof_mdh.py) for a full modified-DH example with unit conversion helpers:

```python
from roboarm.robots.six_dof_mdh import create_six_dof_mdh

robot_6dof = create_six_dof_mdh()
print(f"6-DOF: {robot_6dof.n_joints} joints, {robot_6dof.n_dof} DOF")
# 7 joints (6 variable + 1 fixed TCP), 6 DOF
```

## Complete Example

Here is the full script to build and test the custom 4-DOF arm:

```python
import math
from roboarm.core.types import DHParams, JointConfig, JointLimits
from roboarm.core.robot import RobotArm
from roboarm.visualization.arm_plot import ArmVisualizer

# DH parameters
links = [
    DHParams(alpha=0.0, a=0.4, d=0.0, theta=0.0),
    DHParams(alpha=0.0, a=0.3, d=0.0, theta=0.0),
    DHParams(alpha=0.0, a=0.2, d=0.0, theta=0.0),
    DHParams(alpha=0.0, a=0.1, d=0.0, theta=0.0),
]

# Joint configurations
joints = [
    JointConfig(dh_params=links[0], limits=JointLimits(-math.pi, math.pi), name="Base"),
    JointConfig(dh_params=links[1], limits=JointLimits(-math.pi, math.pi), name="Shoulder"),
    JointConfig(dh_params=links[2], limits=JointLimits(-math.pi/2, math.pi/2), name="Elbow"),
    JointConfig(dh_params=links[3], limits=JointLimits(-math.pi/2, math.pi/2), name="Wrist"),
]

# Build the robot
robot = RobotArm(joints, name="Custom 4-DOF Planar")

# Test FK
q = [0.3, -0.5, 0.8, -0.2]
pose = robot.forward_kinematics(q)
print(f"End-effector: ({pose.x:.4f}, {pose.y:.4f})")

# Visualise
viz = ArmVisualizer(robot)
viz.plot_2d(q, show_workspace=True, title="Custom 4-DOF Arm")
```

---

## See Also

- [DH Parameters Theory](../theory/dh_parameters.md) -- detailed explanation of standard and modified DH
- [Forward Kinematics Theory](../theory/forward_kinematics.md) -- how chain multiplication works
- [Getting Started](01_getting_started.md) -- basic installation and first steps
- [`src/roboarm/core/types.py`](../../src/roboarm/core/types.py) -- `DHParams`, `JointConfig`, `JointLimits` dataclasses
- [`src/roboarm/core/robot.py`](../../src/roboarm/core/robot.py) -- `RobotArm` class
- [`src/roboarm/robots/`](../../src/roboarm/robots/) -- reference robot implementations
