# Inverse Kinematics

Inverse kinematics (IK) is the reverse of forward kinematics: given a desired end-effector pose (position and/or orientation), find the joint angles that achieve it. IK is essential for task-space control -- you specify *where* the tool should go, and the solver figures out *how* the joints should move.

## Why IK Is Harder Than FK

Forward kinematics is a straightforward matrix multiplication with a unique answer. Inverse kinematics, by contrast, is challenging for several reasons:

- **Multiple solutions** -- a 2-link arm reaching a target may have both an elbow-up and an elbow-down configuration. A 6-DOF arm can have up to 16 solutions.
- **No solution** -- the target may lie outside the reachable workspace.
- **Singularities** -- at certain configurations the Jacobian loses rank and the robot cannot move in all task-space directions, causing numerical solvers to diverge or oscillate.
- **Redundancy** -- when the robot has more DOF than the task space (e.g., a 3-link planar arm in 2-D), infinitely many solutions exist.

## Solver Approaches

This toolkit provides five IK solvers, each registered with the `IKSolverRegistry` so they can be created by name at runtime. All solvers live in [`src/roboarm/kinematics/solvers/`](../../src/roboarm/kinematics/solvers/).

### 1. Analytical (Closed-Form)

**File:** [`analytical.py`](../../src/roboarm/kinematics/solvers/analytical.py)
**Registry name:** `"analytical_2link"`

For a 2-link planar arm, the law of cosines yields an exact, constant-time solution:

```
cos(q2) = (x^2 + y^2 - L1^2 - L2^2) / (2 * L1 * L2)
q2 = atan2(+/- sqrt(1 - cos^2(q2)), cos(q2))
q1 = atan2(y, x) - atan2(L2*sin(q2), L1 + L2*cos(q2))
```

The solver returns both the elbow-up and elbow-down solutions. It only works for exactly 2-DOF planar arms.

### 2. Jacobian Pseudoinverse

**File:** [`jacobian_ik.py`](../../src/roboarm/kinematics/solvers/jacobian_ik.py)
**Registry name:** `"jacobian_pseudoinverse"`

An iterative method that computes the position error and applies:

```
delta_q = pinv(J) * error
```

where `pinv(J)` is the Moore-Penrose pseudoinverse of the Jacobian. Simple and general, but numerically unstable near singularities where the Jacobian becomes ill-conditioned.

### 3. Damped Least Squares (DLS)

**File:** [`damped_least_squares.py`](../../src/roboarm/kinematics/solvers/damped_least_squares.py)
**Registry name:** `"damped_least_squares"`

Adds a damping factor to regularise the pseudoinverse:

```
delta_q = J^T * (J * J^T + lambda^2 * I)^{-1} * error
```

The damping term `lambda^2 * I` prevents numerical blow-up near singularities at the cost of slightly slower convergence away from them. This is the default solver used by the AI agent layer.

### 4. Cyclic Coordinate Descent (CCD)

**File:** [`ccd.py`](../../src/roboarm/kinematics/solvers/ccd.py)
**Registry name:** `"ccd"`

A heuristic method that sweeps through joints from the tip back to the base. For each joint it computes the rotation that best aligns the end-effector-to-joint vector with the target-to-joint vector. CCD is simple to implement, handles arbitrary chain lengths, and requires no Jacobian computation.

### 5. FABRIK (Forward And Backward Reaching Inverse Kinematics)

**File:** [`fabrik.py`](../../src/roboarm/kinematics/solvers/fabrik.py)
**Registry name:** `"fabrik"`

A position-based iterative solver that operates directly on joint positions rather than angles. Each iteration has two phases:

1. **Forward reach** -- move the end-effector to the target, then adjust each joint toward the base while preserving link lengths.
2. **Backward reach** -- pin the base and adjust each joint toward the tip while preserving link lengths.

Joint angles are recovered from the final positions via `atan2`. FABRIK converges quickly and produces natural-looking poses.

## Solver Comparison

| Solver | Type | Singularity handling | Speed | Multi-solution | Works for any DOF |
|--------|------|---------------------|-------|----------------|-------------------|
| Analytical 2-link | Closed-form | N/A | Fastest | Yes (2 solutions) | No (2-DOF only) |
| Jacobian Pseudoinverse | Iterative | Poor | Moderate | No | Yes |
| Damped Least Squares | Iterative | Good (damping) | Moderate | No | Yes |
| CCD | Iterative/heuristic | Good | Fast | No | Yes |
| FABRIK | Iterative/heuristic | Good | Fast | No | Yes |

## Using the Solver Registry

The `IKSolverRegistry` pattern allows you to select solvers by name without importing their classes directly:

```python
from roboarm.robots.two_link_planar import create_two_link_planar
from roboarm.kinematics.solvers.registry import IKSolverRegistry

# Import solver modules to trigger registration
import roboarm.kinematics.solvers.damped_least_squares
import roboarm.kinematics.solvers.ccd

robot = create_two_link_planar()

# List available solvers
print(IKSolverRegistry.available())
# ['ccd', 'damped_least_squares']

# Create and use a solver
solver = IKSolverRegistry.create("damped_least_squares", robot)
```

## Solving an IK Problem

Every solver implements the same `solve(target, q0)` interface defined by `IKSolverBase` in [`src/roboarm/kinematics/inverse.py`](../../src/roboarm/kinematics/inverse.py):

```python
import numpy as np
from roboarm.core.types import EndEffectorPose

target = EndEffectorPose(
    position=np.array([1.2, 0.8, 0.0]),
    rotation=np.eye(3),
    transform=np.eye(4),
)

result = solver.solve(target)

if result.success:
    print(f"Joint angles: {result.primary.values}")
    print(f"Residual error: {result.residual_error:.2e}")
    print(f"Iterations: {result.iterations}")
    print(f"Time: {result.computation_time_ms:.2f} ms")
else:
    print(f"IK failed: {result.messages}")
```

The `IKSolution` dataclass includes fields for `success`, `primary` solution, `alternatives`, `iterations`, `residual_error`, and `computation_time_ms`, giving you full diagnostic insight into solver behaviour.

## Solver Configuration

Iterative solvers accept an `IKConfig` or direct keyword arguments:

```python
solver = IKSolverRegistry.create(
    "damped_least_squares",
    robot,
    max_iterations=1000,
    tolerance=1e-8,
    damping=0.05,
    step_size=0.8,
)
```

---

## See Also

- [Forward Kinematics](forward_kinematics.md) -- the forward problem that IK inverts
- [Jacobian Matrix](jacobian.md) -- the Jacobian used by pseudoinverse and DLS solvers
- [DH Parameters](dh_parameters.md) -- the kinematic model that underlies all solvers
- [`src/roboarm/kinematics/solvers/`](../../src/roboarm/kinematics/solvers/) -- all solver implementations
- [`src/roboarm/kinematics/inverse.py`](../../src/roboarm/kinematics/inverse.py) -- `IKSolverBase` and `IKConfig`
- [`src/roboarm/kinematics/solvers/registry.py`](../../src/roboarm/kinematics/solvers/registry.py) -- `IKSolverRegistry`
