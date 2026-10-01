# WaLTRL MJLab: Python Style Guide and Project Conventions

This is the repository's coding style guide. Apply it to new and modified project code; preserve external package APIs and avoid unrelated rewrites of upstream code.

Read [MIGRATION_PLAN.md](MIGRATION_PLAN.md) for the architecture, inspected source revisions, migration findings, and implementation phases. This guide owns coding conventions; the plan owns migration sequencing and design decisions. Keep both consistent when decisions change.

Current status (October 1, 2026): the initial FS MJLab environment, shared modeling submodule, heuristic joystick policy, and minimal PPO entry point are implemented. Interactive playback uses MJLab's Viser viewer. The project adopts MJLab's managers and training stack; it does not reproduce the old JAX/Brax implementation. See README.md and the migration plan for the decision and remaining work. The shared-core package shown below remains a future target; code examples are illustrative unless present in the implementation.

## 1. General Philosophy

Code should be written primarily for humans.

Prefer code that reads naturally as **English or mathematics** without requiring comments to understand its basic operation. Variable, function, class, and module names should communicate intent clearly.

Follow these principles:

- Prefer clarity over cleverness.
- Prefer explicit behavior over hidden behavior.
- Prefer simple functions and data flow over deeply nested abstractions.
- Avoid premature abstraction.
- Avoid duplicate logic when a clear reusable abstraction exists.
- Keep related code together.
- Make invalid or unexpected states fail clearly.
- Optimize for readability first and performance second unless profiling demonstrates a performance problem.

Comments and documentation should supplement readable code, not compensate for unreadable code.

---

## 2. Project Structure

Use a **flat package layout** rather than a `src/` layout.

Recommended structure:

```text
WaLTRL_mjLab/
├── modeling/                       # Pinned WaLTER_Modeling submodule
├── walter_mjlab/                    # Installable RL package
│   ├── __init__.py
│   ├── robots/                     # MJLab model adapter
│   ├── tasks/velocity/             # Task configuration and MDP terms
│   ├── controllers/               # RL controller adapter
│   ├── training/
│   ├── evaluation/
│   └── cli/
├── packages/walter_control/        # Future independently installable core
│   ├── pyproject.toml
│   └── walter_control/             # Flat layout within this package too
├── simulations/
├── scripts/
├── tests/
├── configs/
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

The package directory should contain reusable Python code.

Top-level directories such as `scripts/`, `simulations/`, and `tests/` should contain applications of the package rather than core package functionality.

For example:

```python
from walter_mjlab.controllers import PolicyController
```

is preferable to manipulating the Python path:

```python
import sys

sys.path.append(...)
```

**Never use `sys.path` modifications to make project imports work.**

Install projects into the active environment during development:

```bash
pip install -e .
```

Submodules or reusable supporting packages should likewise be installed as packages rather than manually added to the import path.

Use `pyproject.toml` as the authoritative location for project metadata, dependencies, build configuration, and developer tooling.

---

## 3. Naming Conventions

Follow standard Python naming conventions consistently.

| Element | Convention | Example |
|---|---|---|
| Packages | `snake_case` | `walter_modeling` |
| Modules | `snake_case.py` | `reward_functions.py` |
| Functions | `snake_case` | `compute_tracking_reward()` |
| Methods | `snake_case` | `update_state()` |
| Variables | `snake_case` | `desired_velocity` |
| Parameters | `snake_case` | `episode_length` |
| Classes | `PascalCase` | `WalterEnvironment` |
| Exceptions | `PascalCase` | `InvalidRobotStateError` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_MOTOR_TORQUE` |
| Internal members | `_leading_underscore` | `_previous_action` |

Names should be descriptive and unambiguous.

Prefer:

```python
desired_velocity
previous_action
wheel_angular_velocity
tracking_reward
termination_penalty
```

over:

```python
vel
pa
w
r1
pen
```

Avoid abbreviations unless they are standard and immediately recognizable within the field.

Acceptable examples include:

```python
dt
qpos
qvel
imu
ppo
```

Avoid inventing project-specific abbreviations simply to shorten names.

---

## 4. Mathematical and Reinforcement Learning Code

Mathematical code is an exception to the general preference for descriptive variable names.

Short mathematical variables are appropriate when:

1. They correspond directly to established notation.
2. Their scope is small.
3. Longer names would make the mathematics harder to recognize.

For example:

```python
error = desired_velocity - measured_velocity
reward = torch.exp(-0.5 * (error / sigma_v).square())
```

may be clearer than artificially verbose mathematical names.

Likewise:

```python
q
q_dot
tau
```

can be reasonable within narrowly scoped dynamics code.

At API boundaries, prefer descriptive names:

```python
def compute_joint_torque(
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    desired_position: torch.Tensor,
) -> torch.Tensor:
```

rather than:

```python
def torque(q, qd, q_des):
```

The implementation may then use conventional mathematical notation locally if doing so improves readability.

---

## 5. Type Annotations

**All function and method signatures should be typed.**

Specify argument types and return types:

```python
def compute_reward(
    desired_velocity: float,
    measured_velocity: float,
    tracking_sigma: float,
) -> float:
```

Functions returning nothing should explicitly use:

```python
def reset_controller() -> None:
```

Use appropriate array types when practical:

```python
def normalize_observation(observation: torch.Tensor) -> torch.Tensor:
```

Prefer modern Python syntax:

```python
list[str]
dict[str, float]
str | None
```

rather than older syntax when the supported Python version permits it:

```python
List[str]
Dict[str, float]
Optional[str]
```

Local variables generally do not require annotations when their type is obvious:

```python
tracking_error = desired_velocity - measured_velocity
```

Add local annotations when they clarify an otherwise ambiguous data structure.

---

## 6. Functions

Functions should do **one conceptual job**.

A function should generally be understandable from:

1. its name,
2. its parameters,
3. its return type,
4. and its implementation.

Prefer:

```python
tracking_reward = compute_velocity_tracking_reward(...)
action_cost = compute_action_rate_cost(...)
total_reward = tracking_reward - action_cost
```

over one large function performing every reward calculation, termination check, observation update, and logging operation simultaneously.

Function names should generally contain verbs:

```python
compute_reward()
generate_terrain()
update_observation()
load_model()
reset_environment()
```

Boolean-returning functions should read naturally as questions:

```python
is_terminated()
has_contact()
should_reset()
```

Avoid functions with many boolean configuration arguments:

```python
train_model(True, False, True, False)
```

Prefer named configuration objects, enums, or explicit keyword arguments.

Keep functions reasonably short, but **do not split functions solely to satisfy an arbitrary line limit**. A coherent 50-line mathematical routine can be clearer than five disconnected 10-line functions.

---

## 7. Classes

Classes should represent meaningful objects, stateful systems, or interfaces.

Examples:

```python
WalterEnvironment
TerrainGenerator
PPOTrainer
MotorModel
TrainingConfig
```

Do not create classes solely to group unrelated functions.

Prefer plain functions for stateless transformations.

Prefer `dataclass` or equivalent structured configuration objects for collections of related parameters:

```python
@dataclass
class RewardConfig:
    velocity_tracking_weight: float
    action_rate_weight: float
    termination_penalty: float
```

Avoid giant classes that manage simulation, rewards, training, logging, plotting, configuration, and file I/O simultaneously.

---

## 8. Comments

Use the following rule:

> **Code explains what happens. Comments explain why it happens.**

Do not narrate obvious code.

Avoid:

```python
# Increment the episode step
episode_step += 1
```

Instead, comment decisions, constraints, assumptions, or non-obvious behavior:

```python
# Apply the failure penalty once rather than at every subsequent
# simulation step before the environment resets.
termination_cost = FAILURE_COST if newly_terminated else 0.0
```

Good reasons for comments include:

- explaining a non-obvious algorithmic decision,
- documenting a simulator or library workaround,
- identifying an important physical assumption,
- describing why an apparently simpler implementation is incorrect,
- referencing an equation, paper, issue, or external convention,
- warning about numerical or performance implications.

Avoid comments that merely restate the next line.

### Comment Style

Use complete sentences for substantial comments:

```python
# Clamp the command before scaling to prevent extrapolation beyond the
# actuator limits used during policy training.
```

Short annotations may be fragments:

```python
# Wheel-frame velocity
```

Place comments immediately above the code they describe.

Avoid large decorative comment banners such as:

```python
##############################
#      REWARD FUNCTIONS      #
##############################
```

Prefer normal section comments when useful:

```python
# Reward terms
```

---

## 9. Docstrings

Use triple-double-quoted docstrings:

```python
"""Compute the velocity tracking reward."""
```

Public classes and non-trivial public functions should have docstrings.

Very simple private functions do not require docstrings when the name, type signature, and implementation already make their behavior obvious.

For simple functions:

```python
def compute_tracking_error(
    desired_velocity: float,
    measured_velocity: float,
) -> float:
    """Compute the signed velocity tracking error."""
```

For more complicated functions, document:

- purpose,
- important assumptions,
- unusual arguments,
- return behavior,
- side effects,
- exceptions,
- or mathematical definitions that are not obvious.

Do not repeat information already obvious from type annotations.

Avoid:

```python
"""
Args:
    velocity: A float representing velocity.
"""
```

when `velocity: float` and the function name already establish this.

Docstrings describe the **interface and behavior**. Inline comments describe implementation decisions.

---

## 10. Control Flow

Prefer shallow, readable control flow.

Use early returns when they simplify logic.

Prefer:

```python
if episode_terminated:
    return termination_penalty

return compute_nominal_reward(state)
```

over:

```python
if not episode_terminated:
    reward = compute_nominal_reward(state)
else:
    reward = termination_penalty

return reward
```

Avoid excessive nesting.

Extract complicated conditions into named variables:

```python
robot_has_failed = abs(body_pitch) > MAX_FAILURE_PITCH

if robot_has_failed:
    ...
```

instead of repeatedly writing:

```python
if abs(state.qpos[2]) > np.pi / 2:
```

Named intermediate variables often improve readability and make debugging easier.

---

## 11. Constants and Magic Numbers

Avoid unexplained magic numbers.

Prefer:

```python
MAX_FAILURE_PITCH = np.pi / 2
```

over:

```python
if abs(body_pitch) > 1.5708:
```

Configuration parameters should normally live in configuration objects or configuration files rather than being scattered throughout the implementation.

Physical constants, fixed algorithm constants, and intentional immutable module-level values should use `UPPER_SNAKE_CASE`.

---

## 12. Imports

Organize imports in three groups:

```python
import pathlib

import mujoco
import numpy as np
import torch

from walter_mjlab.controllers import PolicyController
from walter_mjlab.training import TrainingConfig
```

Order:

1. Python standard library
2. Third-party packages
3. Local project imports

Use absolute project imports whenever practical.

Prefer:

```python
from walter_mjlab.tasks.velocity.mdp.rewards import compute_tracking_reward
```

over complicated relative imports such as:

```python
from ...rewards import compute_tracking_reward
```

Avoid wildcard imports:

```python
from module import *
```

Import modules or symbols explicitly.

---

## 13. Configuration

Separate **configuration** from **algorithm implementation**.

Avoid hard-coding experiment parameters throughout training code.

Prefer:

```python
@dataclass
class PPOConfig:
    learning_rate: float = 1e-4
    discount_factor: float = 0.995
    num_envs: int = 2048
    batch_size: int = 1024
```

The training algorithm should consume configuration rather than define experimental choices internally.

This makes experiments easier to reproduce and compare.

---

## 14. Reinforcement Learning-Specific Practices

RL pipelines should keep the following concepts separate whenever practical:

```text
environment
├── state / observations
├── actions
├── dynamics / simulation
├── reward
└── termination

training
├── algorithm
├── configuration
├── curriculum
├── evaluation
└── checkpointing

experiment
├── configuration
├── random seed
├── metrics
└── outputs
```

Reward functions should preferably be decomposed into named terms:

```python
velocity_reward = compute_velocity_reward(...)
orientation_reward = compute_orientation_reward(...)
action_rate_cost = compute_action_rate_cost(...)

total_reward = velocity_reward + orientation_reward - action_rate_cost
```

This is preferable to embedding a large mathematical expression directly inside the environment step function.

Observation construction should similarly expose semantic components before concatenation:

```python
observation = torch.cat(
    [
        body_orientation,
        body_velocity,
        joint_positions,
        joint_velocities,
        previous_action,
    ],
    dim=-1,
)
```

This makes the observation definition easy to audit against documentation and experiments. In this example each field has shape `(num_envs, feature_count)`; concatenation preserves the environment dimension.

The additive reward example above illustrates decomposition, not a prescription to change existing reward math. WaLTRL's legacy tracking objective multiplies its tracking terms. Preserve the declared equation during migration and document any redesign separately.

---

## 15. Side Effects and State

Make state changes obvious.

A function named:

```python
compute_reward()
```

should not unexpectedly save files or modify unrelated state.

Functions that mutate state should make that behavior apparent from their interface or naming.

Prefer explicit data flow where practical:

```python
next_state = update_state(state, action)
```

rather than relying heavily on hidden global state.

Avoid mutable global variables.

---

## 16. Error Handling

Fail clearly when assumptions are violated.

Use:

```python
if action.shape != expected_action_shape:
    raise ValueError(f"Expected action shape {expected_action_shape}, received {action.shape}.")
```

rather than allowing an unrelated matrix operation to fail much later.

Use assertions primarily for internal invariants:

```python
assert observation.ndim == 2  # Batched training observations.
```

Use exceptions for invalid inputs or failures that users of the API may reasonably encounter.

Never silently ignore errors without a documented reason.

---

## 17. Testing

Important reusable logic should be testable independently from the complete simulation or training pipeline.

Prioritize tests for:

- reward terms,
- observation construction,
- action scaling,
- coordinate transforms,
- termination conditions,
- configuration loading,
- mathematical utilities,
- serialization/checkpointing,
- and known regression cases.

For scientific software, tests should verify both software behavior and important physical/mathematical assumptions where practical.

---

## 18. Formatting and Tooling

Code formatting should be automated.

Recommended baseline:

- **Ruff** for linting and import/style checks.
- **Ruff formatter** for automatic formatting.
- **Pyright** for static type checking.
- **pytest** for testing.

Do not manually debate formatting that can be enforced automatically.

Put tool configuration in `pyproject.toml` so all contributors and automated agents use the same rules. Use four-space indentation and a 100-character formatter line-length target for project Python code. Preserve upstream formatting when making focused changes in a dependency.

Use Python 3.12 as the initial project baseline, subject to the dependency compatibility check in the migration plan. Once packaging exists, use the committed `uv.lock` and `uv sync --locked` for reproducible environments. Add local package sources and dependency groups explicitly; an editable root install alone does not install an unconfigured sibling package or initialize Git submodules.

After tooling is configured, the standard checks are `uv run ruff check .`, `uv run ruff format --check .`, `uv run pyright`, and the relevant `uv run pytest` targets. Scope tools to project-owned code and exclude `modeling/`, generated artifacts, and virtual environments. These checks are configured for the initial implementation. CPU integration tests are included in pytest; GPU and physical joystick validation must be recorded separately.

---

## 19. File and Module Organization

A module should have a clear responsibility.

Prefer:

```text
rewards/
    velocity.py
    orientation.py
    regularization.py
```

when reward logic becomes substantial.

Do not create dozens of tiny files unnecessarily. A single `rewards.py` module is preferable while the functionality remains small and cohesive.

Split modules when doing so creates meaningful conceptual boundaries, not merely to reduce line count.

---

## 20. Public vs. Internal Interfaces

Use a leading underscore for implementation details that other modules should not depend on:

```python
def _compute_internal_contact_state(...):
```

Keep the intended public interface small.

Use `__init__.py` deliberately to expose important package-level interfaces where doing so improves usability:

```python
from .policy import PolicyController
```

This allows:

```python
from walter_mjlab.controllers import PolicyController
```

instead of exposing unnecessary internal module structure.

Do not automatically re-export every object from every module.

---

## 21. Research Code Is Still Production Code

Exploratory notebooks and temporary scripts may be messy.

Code that becomes part of the reusable RL pipeline should not remain exploratory code.

Once an experiment proves useful:

1. move reusable functionality into the package,
2. give variables and functions meaningful names,
3. add types,
4. remove dead experiments,
5. separate configuration from implementation,
6. add tests where appropriate,
7. document non-obvious assumptions.

Do not allow the main repository to become a permanent collection of experiments named:

```text
test2.py
test_new.py
working.py
working_final.py
working_final2.py
```

Use version control for history.

---

## 22. The Readability Test

Before considering code complete, ask:

> Could another researcher familiar with robotics or reinforcement learning understand the logic of this code without reading the comments?

If **no**, improve the structure and naming first.

Then ask:

> Are there assumptions, decisions, mathematical conventions, or strange implementation details that the code itself cannot communicate?

If **yes**, document those with comments or docstrings.

The goal is not comment-free code.

The goal is code where **comments carry information that code cannot express well on its own**.

---

## 23. Tensor and Simulation Conventions

Use PyTorch tensors for MJLab training code. Keep JAX/Brax code isolated to explicitly identified legacy comparison tools. The portable control core must not require PyTorch or MJLab merely to import its contracts; use lightweight types and NumPy where appropriate for the reference runtime.

- Document array shape, dtype, device, units, and frame at public numerical boundaries. Prefer `(num_envs, num_features)` for batched observations and actions; distinguish single-robot runtime arrays explicitly.
- Preserve batched execution. Avoid Python loops over environments, accidental CPU copies, `.item()` calls, or tensor-dependent Python branches in hot paths. Validate configuration at initialization and use vectorized masks for per-environment decisions.
- Allocate tensors on the intended device with the intended dtype. Avoid implicit NumPy/Torch conversions during stepping. Use inference/no-gradient contexts for rollout inference where appropriate, while preserving gradients in optimization code.
- Distinguish policy-step state updates from physics-substep feedback. A relative knee target is sampled once per policy step and held through substeps in the legacy task.
- Document ownership and mutation of buffers. Reset history, previous actions, controller state, and randomization state only for the environments being reset.
- Distinguish task termination from time-limit truncation and verify value bootstrapping. Make reward scaling by timestep explicit.
- Seed supported random sources and log their settings. Do not promise bitwise determinism across devices or simulator versions.

Readability remains the default, but batched GPU execution is an architectural requirement. Express the mathematics clearly using tensor operations rather than introducing avoidable host synchronization.

---

## 24. Robotics and Portable Controller Boundaries

The project trains RL policies, but the controller interface must also support heuristics and future control methods. Keep simulator objects and ROS messages in adapters, outside portable controller logic.

- Keep geometry and physical robot parameters in the pinned `modeling/` submodule. Install `WaLTER_Modeling` as a package and respect that existing public import spelling; do not rename external APIs to satisfy local naming style.
- Resolve assets through package resources or explicit configuration paths, never an assumed working directory. Keep terrain/task configuration separate from the physical robot description.
- Use named joint mappings and explicit permutations between policy, simulator, and hardware order. Never equate actuator IDs with generalized DOF indices or assume that names imply array order.
- Use SI units at interfaces. Document motor-side versus joint-side quantities, gear ratio conversion, joint signs/offsets, frames, quaternion order, and whether acceleration includes gravity. Put units in field names when they would otherwise be ambiguous.
- Define observation and action schemas with versions: field order, scaling, normalization, clipping, history, timing, and reset semantics are part of the interface.
- Actor inputs must have a corresponding hardware sensor or validated estimator. Keep simulation-only truth in critic observations unless an explicit deployment path exists.
- Separate normalized policy actions from physical joint commands. Specify control modes, gains, update periods, timestamps, and validity. Do not apply action scaling or observation normalization twice.
- Keep training checkpoints distinct from deployment bundles. Export provenance, schema versions, model/config hashes, and inference parity fixtures alongside the policy.
- Test controller switching, stale inputs, limits, and robot-specific fallback behavior in adapters. Do not infer hardware readiness from a successful simulated rollout.

Avoid speculative abstractions: start with concrete typed contracts and the minimum adapters required for the next migration phase.

---

## 25. Experiment Provenance and Future Reference

Record resolved experiment configuration, seeds, code revision, modeling gitlink, physical profile, dependency versions/lock, and relevant generated model hashes with each run. Record dirty working-tree state or a patch when applicable; do not label uncommitted work solely by a commit hash.

Keep generated checkpoints, videos, logs, and large models outside ordinary source control. Track small conformance fixtures and schemas deliberately. Logging services should be optional, and importing a module must not launch a simulation, open a GUI, log into a service, or start training. Put entry-point side effects behind explicit functions and main guards.

For migration changes, identify whether the change preserves legacy behavior, fixes an identified bug, or redesigns the task. The known source-level concerns are recorded in [MIGRATION_PLAN.md](MIGRATION_PLAN.md): model/profile mismatches, torque indexing, previous-action timing, action ordering, motor units, and reward scaling. Do not turn static findings into claims of reproduced failures without testing them.

Keep durable decisions in repository documents, not temporary checkouts or conversation memory. Update the plan with completed phases, actual validation results, and remaining blockers. Record deviations where they occur rather than adding competing instruction files or duplicating the entire plan here.

Run checks appropriate to the change and report what was actually run. Prioritize regression tests for numerical contracts and physical assumptions; documentation-only changes need consistency review rather than a training run. GPU or hardware checks that were unavailable must be reported as unverified.

---

## Summary Rules

When in doubt:

1. **Use descriptive `snake_case` names.**
2. **Use `PascalCase` for classes.**
3. **Use `UPPER_SNAKE_CASE` for constants.**
4. **Type every function signature.**
5. **Write small, cohesive functions.**
6. **Names explain what; comments explain why.**
7. **Use mathematical notation when it genuinely improves mathematical readability.**
8. **Avoid hidden state and surprising side effects.**
9. **Keep configuration separate from algorithms.**
10. **Install packages properly; never manipulate `sys.path`.**
11. **Use a flat package layout with `pyproject.toml`.**
12. **Automate formatting, linting, typing, and testing.**
13. **Use version control instead of retaining obsolete code.**
14. **Prefer readable, obvious code over clever code.**
15. **Treat reusable research code as software, not as a temporary experiment.**
