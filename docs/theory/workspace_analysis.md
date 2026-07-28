# Workspace Analysis

The workspace of a robot arm is the set of all end-effector positions it can physically reach. Understanding the workspace is essential before deploying any robot: it tells you whether a given task is feasible, which configurations are near the limits of reachability, and how much "room to move" the arm has at any given location.

This toolkit provides workspace analysis through the `WorkspaceAnalyzer` class in [`src/roboarm/workspace/analysis.py`](../../src/roboarm/workspace/analysis.py). All methods use **Monte Carlo sampling** — random joint configurations are evaluated via forward kinematics, and the resulting end-effector positions are collected to characterise the reachable space. Results improve with larger sample counts.

---

## Monte Carlo Approach

Analytical workspace computation is only tractable for simple arm geometries. For general serial-link robots, a practical alternative is:

1. Sample `n` random joint configurations uniformly within each joint's limits (or `[-π, π]` if no limits are defined).
2. Evaluate FK at every sample to get an end-effector position.
3. Aggregate the resulting point cloud to answer reachability questions.

The accuracy of every estimate below scales with `sqrt(n_samples)` — doubling the sample count halves the estimation error. Typical values of `n_samples=500` to `n_samples=5000` are sufficient for most purposes.

---

## Creating an Analyser

```python
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.workspace.analysis import WorkspaceAnalyzer

robot = create_two_link_planar(link1=1.0, link2=1.0)
analyzer = WorkspaceAnalyzer(robot)
```

`WorkspaceAnalyzer` accepts any `RobotArm`, planar or spatial.

---

## Sampling the Workspace

`sample_workspace(n_samples)` returns the raw point cloud as an `(n_samples, 3)` array of `[x, y, z]` positions:

```python
points = analyzer.sample_workspace(n_samples=2000)
# points.shape == (2000, 3)

# Scatter plot with matplotlib
import matplotlib.pyplot as plt
plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.3)
plt.axis("equal")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Workspace point cloud — 2-link planar arm")
plt.show()
```

For a 2-link planar arm with L1=L2=1, the workspace is an annulus with inner radius 0 (if the arm can fold completely) and outer radius `L1 + L2 = 2.0`.

---

## Reachability Check

`is_reachable(target_position, n_samples, tolerance)` returns `True` if any sampled point is within `tolerance` metres of `target_position`:

```python
# A point clearly inside the workspace
print(analyzer.is_reachable([1.5, 0.3, 0.0]))   # True

# A point outside the workspace (beyond max reach of 2.0)
print(analyzer.is_reachable([5.0, 5.0, 0.0]))   # False
```

**Parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_samples` | 500 | Higher values increase confidence in the answer |
| `tolerance` | 0.05 | The search radius around the target (metres) |

The function logs the minimum observed distance to the target, which can help calibrate the tolerance for borderline cases.

---

## Workspace Bounding Box

`workspace_bounds(n_samples)` returns the axis-aligned bounding box of the sampled workspace as a dictionary:

```python
bounds = analyzer.workspace_bounds(n_samples=3000)
print(bounds)
# {'x': (-2.0, 2.0), 'y': (-2.0, 2.0), 'z': (0.0, 0.0)}

x_min, x_max = bounds['x']
y_min, y_max = bounds['y']
```

For a planar arm, `z_min == z_max == 0.0`. For a spatial arm, all three axes have non-trivial bounds.

The bounding box is a conservative over-approximation: the arm cannot necessarily reach every point inside it (for example, a 2-link arm cannot reach the point (0, 0, 0) when both links are fully extended, but (0,0,0) is inside the bounding box).

---

## Joint Limits and Their Effect

If a `JointLimits` object is set on a joint, `WorkspaceAnalyzer` samples only within those limits. If a joint has no limits, it samples over the full range `[-π, π]`. This means:

- **Limiting joints** shrinks the workspace.
- **Removing limits** maximises the sampled workspace volume.

```python
import math
from roboarm.core.types import DHParams, JointConfig, JointLimits
from roboarm.core.robot import RobotArm
from roboarm.workspace.analysis import WorkspaceAnalyzer

# Restricted elbow: only 0 to 90 degrees
joints = [
    JointConfig(DHParams(0, 1.0, 0, 0), limits=JointLimits(-math.pi, math.pi), name="J1"),
    JointConfig(DHParams(0, 1.0, 0, 0), limits=JointLimits(0, math.pi/2), name="J2"),
]
robot_restricted = RobotArm(joints, name="Restricted 2-link")
analyzer_r = WorkspaceAnalyzer(robot_restricted)

bounds_r = analyzer_r.workspace_bounds(2000)
print("Restricted workspace:", bounds_r)
```

---

## Full Example

```python
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.workspace.analysis import WorkspaceAnalyzer

robot = create_two_link_planar(link1=1.0, link2=0.6)
analyzer = WorkspaceAnalyzer(robot)

# Point cloud
points = analyzer.sample_workspace(3000)
print(f"Sampled {len(points)} workspace points")

# Bounding box
bounds = analyzer.workspace_bounds(3000)
print(f"x range: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}]")
print(f"y range: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}]")

# Reachability queries
test_points = [
    ([1.2, 0.4, 0.0], "Inside workspace"),
    ([1.6, 0.0, 0.0], "Near outer edge"),
    ([3.0, 3.0, 0.0], "Outside workspace"),
]
for pos, label in test_points:
    result = analyzer.is_reachable(pos, n_samples=1000, tolerance=0.1)
    print(f"  {label}: {result}")
```

---

## Limitations

- **Approximate, not exact.** Because the approach is sampling-based, the results are statistical. A `False` result from `is_reachable` with a small sample count might reverse with more samples for borderline points.
- **No obstacle awareness.** The workspace is purely kinematic. Physical obstacles, self-collision, or joint velocity constraints are not modelled.
- **Uniform joint sampling.** All joint configurations are equally likely in the sample. This means the point cloud is denser near the centre of the workspace (where many configurations converge) and sparser near the boundary.

---

## See Also

- [Forward Kinematics](forward_kinematics.md) — the FK computation that workspace sampling relies on
- [Trajectory Planning](trajectory_planning.md) — planning joint-space paths within the reachable workspace
- [`src/roboarm/workspace/analysis.py`](../../src/roboarm/workspace/analysis.py) — `WorkspaceAnalyzer` implementation
- [`src/roboarm/visualization/workspace_plot.py`](../../src/roboarm/visualization/workspace_plot.py) — plotting the sampled workspace point cloud
