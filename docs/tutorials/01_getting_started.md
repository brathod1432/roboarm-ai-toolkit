# Getting Started

This tutorial walks you through installing the toolkit, creating your first robot, computing forward and inverse kinematics, and visualising the result. By the end you will have a working understanding of the core API.

## Installation

Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/yourusername/roboarm-ai-toolkit.git
cd roboarm-ai-toolkit
pip install -e ".[dev]"
```

This installs the `roboarm` package along with testing tools (`pytest`, `coverage`) and visualisation dependencies (`matplotlib`).

## Creating Your First Robot

The fastest way to get a robot is to use one of the built-in factory functions:

```python
from roboarm.robots.two_link_planar import create_two_link_planar

robot = create_two_link_planar(link1=1.0, link2=1.0)
print(robot)
# RobotArm(name='2-Link Planar (L1=1.0, L2=1.0)', n_joints=2, n_dof=2)
```

The robot has two revolute joints operating in the XY plane, each with a link length of 1.0. You can inspect its properties:

```python
print(f"Degrees of freedom: {robot.n_dof}")
print(f"Joint names: {robot.joint_names}")
print(f"Joint limits: {robot.joint_limits}")
```

Other pre-built robots are available in [`src/roboarm/robots/`](../../src/roboarm/robots/):

- `create_three_link_planar()` -- a 3-DOF redundant planar arm
- `create_six_dof_mdh()` -- a 6-DOF spatial arm using modified DH parameters

## Computing Forward Kinematics

Forward kinematics converts joint angles to an end-effector pose. Pass a list of joint angles (in radians) to `forward_kinematics()`:

```python
import math

pose = robot.forward_kinematics([math.pi / 4, -math.pi / 6])

print(f"End-effector position:")
print(f"  x = {pose.x:.4f}")
print(f"  y = {pose.y:.4f}")
print(f"  z = {pose.z:.4f}")
```

The returned `EndEffectorPose` object also contains the 3x3 rotation matrix (`pose.rotation`) and the full 4x4 homogeneous transform (`pose.transform`).

You can also retrieve the positions of every joint in the chain:

```python
positions = robot.joint_positions([0.5, -0.3])
print(positions)
# Array of shape (3, 3): base, elbow, and end-effector [x, y, z]
```

## Solving Inverse Kinematics

Inverse kinematics finds joint angles that place the end-effector at a desired position. The toolkit provides several solvers; the Damped Least Squares (DLS) solver is a good default:

```python
import numpy as np
from roboarm.core.types import EndEffectorPose
from roboarm.kinematics.solvers.registry import IKSolverRegistry

# Import solver modules to register them
import roboarm.kinematics.solvers.damped_least_squares

# Create the solver
solver = IKSolverRegistry.create("damped_least_squares", robot)

# Define the target position
target = EndEffectorPose(
    position=np.array([1.2, 0.8, 0.0]),
    rotation=np.eye(3),
    transform=np.eye(4),
)

# Solve
result = solver.solve(target)

if result.success:
    angles = result.primary.values
    print(f"Solution: q1={angles[0]:.4f}, q2={angles[1]:.4f}")
    print(f"Residual error: {result.residual_error:.2e}")
    print(f"Iterations: {result.iterations}")
    print(f"Solve time: {result.computation_time_ms:.2f} ms")
else:
    print(f"IK failed: {result.messages}")
```

## Visualising the Robot

The `ArmVisualizer` class creates matplotlib plots of the robot configuration:

```python
from roboarm.visualization.arm_plot import ArmVisualizer

viz = ArmVisualizer(robot)

# Plot the arm at a specific configuration
ax = viz.plot_2d([0.5, -0.3])

# Optionally show the workspace boundary
ax = viz.plot_2d([0.5, -0.3], show_workspace=True,
                  title="My First Robot Plot")
```

For an overview of joint angles relative to their limits, use the configuration-space plot:

```python
ax = viz.plot_configuration_space([0.5, -0.3])
```

## Computing the Jacobian

The Jacobian relates joint velocities to end-effector velocities and is useful for analysing manipulability:

```python
from roboarm.kinematics.jacobian import JacobianComputer

jc = JacobianComputer(robot)

J = jc.compute([0.5, -0.3])
print(f"Jacobian shape: {J.shape}")

mu = jc.manipulability([0.5, -0.3])
print(f"Manipulability: {mu:.4f}")

print(f"Near singularity: {jc.is_singular([0.5, -0.3])}")
```

## Running the Tests

Verify that everything is working correctly by running the test suite:

```bash
pytest tests/ -v
```

For a coverage report:

```bash
pytest tests/ --cov=roboarm --cov-report=term-missing
```

## What to Read Next

Now that you have the basics, explore these topics:

- [Building a Custom Robot](02_custom_robot.md) -- define your own robot from DH parameters
- [Using the AI Agent Layer](03_ai_agents.md) -- interact with kinematics via natural language
- [DH Parameters Theory](../theory/dh_parameters.md) -- understand the math behind the transforms
- [Forward Kinematics Theory](../theory/forward_kinematics.md) -- deeper dive into FK chain multiplication
- [Inverse Kinematics Theory](../theory/inverse_kinematics.md) -- all five solvers explained

---

## See Also

- [`src/roboarm/core/robot.py`](../../src/roboarm/core/robot.py) -- `RobotArm` class
- [`src/roboarm/robots/`](../../src/roboarm/robots/) -- pre-built robot factories
- [`src/roboarm/kinematics/jacobian.py`](../../src/roboarm/kinematics/jacobian.py) -- `JacobianComputer`
- [`src/roboarm/visualization/arm_plot.py`](../../src/roboarm/visualization/arm_plot.py) -- `ArmVisualizer`
- [`src/roboarm/kinematics/solvers/`](../../src/roboarm/kinematics/solvers/) -- IK solver implementations
