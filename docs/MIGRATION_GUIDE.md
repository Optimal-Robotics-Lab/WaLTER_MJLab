# MJLab migration guide

This guide is the map from the old Brax/Playground training stack to the files in this repository. [MIGRATION_PLAN.md](../MIGRATION_PLAN.md) records design decisions and later phases. This page records where to edit the current environment.

The old project was one environment class that owned physics, observations, rewards, resets, and the PPO loop. This project is an MJLab `ManagerBasedRlEnv`. Each of those jobs is a manager term. Training and the joystick both call `make_env_cfg()` and then `env.step()`.

## Where things moved

| Old idea | Where it lives now |
| --- | --- |
| `environment/BaseEnv.py`, `environment/RoughTerrainEnv.py` | `walter_mjlab/tasks/velocity/config.py` assembles the environment. Behavior is split across `walter_mjlab/tasks/velocity/mdp/`. |
| Embedded `modeling/GenModel.py` | Pinned submodule `modeling/`. The adapter is `walter_mjlab/robots/walter.py`. |
| Actuator PD / velocity loops inside the env step | `IdealPdActuatorCfg` in `get_robot_cfg()`. Action scaling is `walter_mjlab/tasks/velocity/mdp/actions.py`. |
| Reward expression in `_get_reward` | One function per term in `walter_mjlab/tasks/velocity/mdp/terms.py`. Weights are in `make_env_cfg()`. |
| Command sampling | `walter_mjlab/tasks/velocity/mdp/commands.py` |
| Rough terrain generation | `make_env_cfg()` when `terrain` is `"rough"`. Row selection is `set_terrain_level()` in `mdp/diagnostics.py`. |
| Episode reset and timeout | Reset events and termination terms in `make_env_cfg()`. |
| Curriculum | `update_difficulty()` in `mdp/diagnostics.py`, enabled by `curriculum` in the task JSON. |
| Brax PPO and checkpoints | `walter_mjlab/training/train.py` uses MJLab's RSL-RL runner. Checkpoints land in `artifacts/training/`. |
| Interactive joystick script that wrote motor commands | `simulations/joystick_test.py` → `walter_mjlab/evaluation/joystick.py`. The heuristic emits the same 16-dimensional action PPO uses. The window is MJLab's Viser viewer. |
| Experiment constants in `configs/base_config.py` | `configs/flat.json` and `configs/rough.json`, loaded into `VelocityTaskConfig`. |

Joint order is explicit and shared. Hips, then knees, then wheels, each in front-left, front-right, back-left, back-right order. The names are `JOINT_NAMES` in `walter_mjlab/robots/walter.py`.

## Robot model

Geometry, masses, the passive waist, and the shared motor YAML stay in the modeling submodule:

- `modeling/WaLTER_Modeling/parametric_configs/FS/model_config.yaml`
- `modeling/WaLTER_Modeling/parametric_configs/FS/motor_config.yaml`
- the `WS/` pair for the smaller morphology

`get_robot_spec()` in `walter_mjlab/robots/walter.py` builds a robot-only MuJoCo spec from those files, deletes the generator's actuators, and adds the torso IMU site. MJLab then creates the actuators. The scene, ground, and lights are not part of the robot spec.

`VelocityTaskConfig.profile` selects `FS` or `WS`. Tasks currently reject `WS` until that morphology has its own gains. FS spawn height is 0.4 m in `get_robot_cfg()`.

Change a link length, mass, or mesh in the modeling YAML, then rerun. Change a control gain in this repository, not by editing the generated XML.

## Actuators

Two layers are easy to confuse.

**Physical description.** `motor_config.yaml` still holds stall torque, no-load speed, gear ratio, rotor inertia, and passive damping. `get_robot_spec()` keeps the shared armature and passive joint damping. It does not use that file's `Kp` / `Kd` as the training controller.

**Training controller.** `get_robot_cfg()` installs MJLab `IdealPdActuatorCfg` groups:

| Joints | Stiffness | Damping | Effort limit |
| --- | --- | --- | --- |
| Hips | 150 | 50 | 504.8 N·m |
| Knees | 150 | 50 | 308.75 N·m |
| Wheels | 0 | 12 | 47.97 N·m |

Hips and knees are position targets. Wheels are velocity targets because their stiffness is zero and MJLab applies the damping term to the velocity error. Effort limits are static. This is not a torque-speed curve.

To retune a joint group, edit the `actuator_groups` tuple in `get_robot_cfg()`. To change how a normalized action becomes a target, edit `hip_scale`, `knee_scale`, and `wheel_scale` on `JointTargetsActionCfg` in `mdp/actions.py`. Defaults are 1.25 rad, 5.5 rad, and 60 rad/s. Knee targets are sampled from the measured angle once per policy step and held for the physics substeps.

Physics is 4 ms (`SimulationCfg` in `make_env_cfg()`). Decimation is 5, so the policy and the joystick run at 20 ms.

## Terrain

`configs/flat.json` and `configs/rough.json` set `"terrain": "flat"` or `"rough"`. Both flows read that field.

Flat terrain is an MJLab plane with 4 m environment spacing.

Rough terrain is an 11-row, 1-column Perlin heightfield. Nominal height grows from 0 to 0.3 m across the rows. Tile size and the noise scale are in the `TerrainGeneratorCfg` inside `make_env_cfg()`. `initial_difficulty` picks the starting row. `set_terrain_level()` repeats that choice on every reset, including when curriculum progression is off.

To change the course, edit that generator: `height_range`, `scale`, `size`, `num_rows`, or add another entry under `sub_terrains`. MJLab's heightfield types live in `mjlab.terrains`. A new terrain kind still has to be selected from `VelocityTaskConfig.terrain`, which currently accepts only `flat` and `rough`.

## Rewards

Each term is a function in `walter_mjlab/tasks/velocity/mdp/terms.py` that returns one value per environment. `make_env_cfg()` registers them with `RewardTermCfg` weights. The returned reward is the weighted sum. `scale_rewards_by_dt` is false, so weights are per policy step.

| JSON / config name | Function | Default weight | Role |
| --- | --- | --- | --- |
| `tracking` | `tracking_reward` | 10000 | Product of forward tracking, yaw tracking, and uprightness |
| `torque` | `torque_cost` | -0.0005 | Sum of squared joint-space actuator force |
| `action_rate` | `action_rate_cost` | -1000 | Squared action change, stronger when the command is near zero |
| `vertical_velocity` | `body_velocity_cost` | -5 | Body-frame vertical speed |
| `roll_velocity` | `body_velocity_cost` | -0.5 | Body-frame roll rate |
| `pitch_velocity` | `body_velocity_cost` | -0.5 | Body-frame pitch rate |
| `stationary_joints` | `stationary_joint_cost` | -10 | Joint speed while the command is near zero |
| `flipped` | `flip_cost` | -100000 | Applied on the fallen condition |
| `wheel_collision` | `wheel_collision_cost` | -10000 | Front/rear wheel-center overlap |

Change a formula in `terms.py`. Change a scale either in the `rewards` dict in `make_env_cfg()` or, for a run, under `reward_weights` in the JSON config:

```json
"reward_weights": {"tracking": 10000.0, "action_rate": -200.0}
```

Unknown names fail at startup. A new term needs both a function and a `RewardTermCfg` entry before a JSON weight can refer to it.

Joystick logs write the weighted per-step contributions to `artifacts/joystick/steps.csv`. Those columns are the values that sum to the reward, captured before an automatic reset.

## Observations, commands, and termination

**Observations.** `actor_observation()` and `critic_observation()` in `terms.py` build the vectors. The actor is 68 values: command, previous action, IMU accelerometer, IMU gyro, hip angles, knee sine/cosine, joint velocities, and joint-space actuator force. The critic appends projected gravity and base linear/angular velocity in the body frame, for 77 values. Noise scales are the `GaussianNoiseCfg` on the actor term in `make_env_cfg()`. Set `"noise": false` in the JSON to disable corruption. The critic is always clean.

**Commands.** `VelocityCommand` stores forward speed in m/s and yaw rate in rad/s. During training it resamples from the difficulty-scaled limits. The joystick calls `set_manual_command()` and that reference stays in place across resets. Limits are `max_forward_velocity` and `max_yaw_velocity` on `VelocityTaskConfig` (3 m/s and 1 rad/s unless the JSON overrides them).

**Termination.** `fallen()` ends the episode when the body up-axis drops below 0.1. `time_out` ends it after `episode_length_s` (20 s in the checked-in configs) and is marked as a timeout so value bootstrapping can treat it differently from a fall.

**Curriculum.** With `"curriculum": true`, a finished episode whose average tracking quality is above 0.7 raises difficulty by 0.1. A fall, or average quality below 0.3, lowers it by 0.1. Difficulty stays in `[0, 1]`. It scales sampled command range and selects the rough-terrain row. Manual joystick commands ignore that scale and use the requested reference directly.

## Training

```bash
python -m walter_mjlab.training.train --config configs/flat.json --num-envs 64 --iterations 1000 --device cuda:0
```

The installed command is `walter-train`. The same `VelocityTaskConfig` JSON used by the joystick is the task. PPO widths, learning rate, discount, and rollout length are in `walter_mjlab/training/train.py`. Logs and checkpoints go to a new directory under `artifacts/training/`. Old Brax checkpoints do not load here.

There is no CUDA device on this machine right now, so GPU training is still unverified. CPU smoke runs use `--device cpu` and a small `--num-envs`.

## Joystick and the Viser viewer

The heuristic itself is simulator-free. `HeuristicJoystickController` in `walter_mjlab/controllers/heuristic.py` integrates hip and knee targets and computes wheel speeds. `HeuristicActionPolicy` in `walter_mjlab/evaluation/joystick.py` turns those targets into the normalized action tensor. Rates and the hip limit are `HeuristicControllerConfig` (`max_wheel_velocity`, `knee_rate`, `hip_rate`, `hip_limit`). Override them with `--controller-config path.json`.

Start the interactive session from the repository root:

```bash
python simulations/joystick_test.py --config configs/flat.json
```

The process prints `Open the MJLab viewer at http://127.0.0.1:8080`. Open that page in a browser. Viser is MJLab's viewer: it renders the environment that `env.step()` is already advancing. Pause, single-step, reset, and speed controls are in the browser's Controls tab. Reward traces are in the viewer tabs MJLab adds for the reward manager. The gamepad is read in the Python process, not in the browser.

| Input | Effect |
| --- | --- |
| Left stick | Differential wheel drive and the forward/yaw reward reference |
| Right stick | Integrate left/right knee targets |
| D-pad up/down | Increase/decrease hip splay |
| Left bumper | Fold the front hips |
| A / B / Y / X | Neutral / wide / splay / crouch |
| Start | Reset the environment and the controller's target history |
| Back | Stop the process |

Closing the browser tab does not stop the simulation. Back does. Ctrl+C does as well.

Useful variants:

```bash
python simulations/joystick_test.py --config configs/rough.json --report-every 25
python simulations/joystick_test.py --config configs/flat.json --headless --scripted --steps 200
```

`--scripted` cycles standstill, forward, turn, and reverse without a gamepad. `--headless` skips Viser and is the repeatable check. `--command-source sampled` leaves the training command sampler in charge while the joystick still moves the joints. Every step is appended to `artifacts/joystick/steps.csv`.

First launch compiles Warp kernels and can sit there for a while. Later launches reuse the cache. CPU playback can run slower than real time; the viewer will mark the status as capped when it cannot keep up.

## Task JSON

`VelocityTaskConfig` is the shared experiment file. Fields used by both training and the joystick:

`profile`, `terrain`, `seed`, `episode_length_s`, `curriculum`, `initial_difficulty`, `noise`, `max_forward_velocity`, `max_yaw_velocity`, `reward_weights`.

The joystick entry point forces `num_envs` to 1. Training replaces it with `--num-envs`.

## What this stack does not carry over

JAX code, Brax PPO internals, and Playground environment objects are not hiding in a compatibility layer. There is no native MuJoCo window for interactive driving. Domain randomization, an ONNX export runtime, and hardware actuators are still future work, as listed in the migration plan.
