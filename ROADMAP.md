# Roadmap -- roboarm-ai-toolkit

---

## Phase 1: Core Foundation (Week 1-2)
- [x] Project scaffolding & structure
- [ ] `core/types.py` -- fundamental dataclasses
- [ ] `core/exceptions.py` -- custom exception hierarchy
- [ ] `core/transform.py` -- DH & MDH transforms
- [ ] `core/rotations.py` -- SO(3) utilities
- [ ] `core/robot.py` -- robot model with FK
- [ ] `robots/two_link_planar.py` -- simplest test robot
- [ ] `robots/three_link_planar.py` -- redundant robot
- [ ] `visualization/arm_plot.py` -- 2D arm rendering
- [ ] Unit tests for transforms, rotations, FK
- [ ] Example: `01_two_link_fk.py`

## Phase 2: IK Solvers (Week 3-4)
- [ ] `kinematics/inverse.py` -- IKSolverBase + Registry
- [ ] `kinematics/jacobian.py` -- Geometric & numerical Jacobian
- [ ] `solvers/analytical.py` -- Closed-form 2-link
- [ ] `solvers/jacobian_ik.py` -- Pseudoinverse method
- [ ] `solvers/damped_least_squares.py` -- DLS (from GroupProject heritage)
- [ ] `solvers/ccd.py` -- Cyclic Coordinate Descent
- [ ] `solvers/fabrik.py` -- FABRIK
- [ ] FK<->IK roundtrip tests
- [ ] Solver comparison benchmarks
- [ ] Example: `03_ik_solver_comparison.py`

## Phase 3: AI Agents (Week 5-6)
- [ ] `agents/tools.py` -- Tool definitions & registry
- [ ] `agents/robotics_tools.py` -- Kinematics wrapped as tools
- [ ] `agents/fk_agent.py` -- FK specialist agent
- [ ] `agents/ik_agent.py` -- IK specialist agent
- [ ] `agents/coordinator.py` -- Multi-agent router
- [ ] Agent conversation tests
- [ ] Example: `08_ai_agent_fk.py`

## Phase 4: Polish & Advanced (Week 7-8)
- [ ] `robots/six_dof_mdh.py` -- 6-DOF from GroupProject
- [ ] `trajectory/` -- Joint-space interpolation, LSPB
- [ ] `workspace/analysis.py` -- Reachability visualization
- [ ] Trajectory animation (GIF export)
- [ ] Comprehensive documentation
- [ ] 80%+ test coverage
- [ ] CI/CD pipeline
- [ ] PyPI-ready packaging

## Future
- [ ] Neural IK solver (PyTorch)
- [ ] RL-based planner
- [ ] LLM integration (OpenAI/Anthropic tool-calling)
- [ ] 3D visualization (Three.js or Plotly)
- [ ] ROS2 bridge package
