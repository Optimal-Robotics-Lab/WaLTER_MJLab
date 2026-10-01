# WaLTRL MJLab

Initial WaLTER velocity-training environment using MJLab 1.6.0 and the pinned `WaLTER_Modeling` submodule. The standalone joystick test feeds the GenZero-style heuristic into **the same `env.step()` action, reward, termination, reset, and curriculum managers used by PPO**.

This is a working first environment, not a completed hardware-transfer implementation. See [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) for where actuators, terrain, rewards, and the Viser viewer live. Design history is in [MIGRATION_PLAN.md](MIGRATION_PLAN.md). Coding conventions are in [claude.md](claude.md).

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/) are required for the commands below. From the project root:

```bash
git submodule update --init --recursive
uv sync --locked --extra joystick
```

The modeling dependency is a real submodule pinned to `cb01a463809b32e59be8d702bcb9085f31eaf9d3`. Geometry is generated from that package, not copied into this project. Run development checks with the default `dev` dependency group installed by `uv sync`.

## Standalone joystick test

```bash
uv run python simulations/joystick_test.py --config configs/flat.json
```

An installed equivalent is `uv run walter-joystick`, or `.venv/bin/walter-joystick` when `uv` is not on `PATH`. Default execution is CPU, one environment, and MJLab's Viser viewer. The process prints a local URL; open that page in a browser. For a supported NVIDIA environment, add `--device cuda:0`. First execution compiles Warp kernels and can take significantly longer than later runs. CPU debugging is supported but may not maintain real time.

Default Logitech/XInput-style mapping:

| Input | Effect |
| --- | --- |
| Left stick | Differential wheel drive and forward/yaw reward reference |
| Right stick | Integrate left/right knee targets |
| D-pad up/down | Increase/decrease hip splay |
| Left bumper | Fold front hips |
| A / B / Y / X | Neutral / wide / splay / crouch posture |
| Start | Reset environment and controller target history |
| Back | Quit |

Mappings vary by device/driver. Inspect the device's axis numbering and pass a partial JSON override via `--mapping configs/joystick.json`; keys are defined in `walter_mjlab/controllers/joystick.py`. Select another device with `--joystick-index 1`. Missing controls or disconnects fail clearly and stop stepping.

```json
{"forward_axis": 1, "yaw_axis": 0, "knee_forward_axis": 4, "knee_yaw_axis": 3, "deadzone": 0.2}
```

`--controller-config path.json` overrides `HeuristicControllerConfig` fields such as `max_wheel_velocity`, `knee_rate`, and `hip_rate`. Defaults preserve GenZero-style open-loop behavior, with increments converted to rad/s and normalized action limits enforced. Hips are limited to ±1.25 rad, so the wide posture is clipped intentionally. Positive semantic yaw follows body +Z.

The default reward command is **requested** forward speed (left-stick forward × task maximum) and yaw rate; it is never replaced by measured velocity. Open-loop differential wheel commands do not guarantee exact tracking of those references. Logs include measured velocities so this mismatch can be inspected. The heuristic and task command gains can be tuned independently.

Use `--command-source sampled` to keep the training command sampler active while operating the joints yourself. Joystick mode overrides only the command source; it leaves the rest of the environment machinery enabled.

## Inspect rewards and curriculum

Use the same JSON config for training and joystick evaluation:

```bash
uv run python simulations/joystick_test.py --config configs/rough.json --report-every 25
```

Available task settings include `terrain`, `seed`, `episode_length_s`, `curriculum`, `initial_difficulty`, `noise`, command limits, and `reward_weights`. Unknown reward names fail during configuration. Example override:

```json
{
  "terrain": "flat",
  "curriculum": false,
  "initial_difficulty": 0.5,
  "noise": false,
  "reward_weights": {"tracking": 10000.0, "action_rate": -200.0}
}
```

- `curriculum: true` adjusts per-environment difficulty on reset. Full episodes with tracking quality >0.7 increase difficulty by 0.1; falls or average tracking quality <0.3 decrease it by 0.1. Other resets keep the level. Values are clamped to [0, 1]. This is an initial curriculum, not the final training strategy.
- Difficulty scales **sampled** command limits from 25% to 100%; manual commands remain exactly the requested reference. Rough terrain uses 11 seeded Perlin heightfield rows spanning nominal height ranges from 0 to 0.3 m. Difficulty chooses the nearest row at reset. The generator samples difficulty within row intervals, so `initial_difficulty` selects a stage, not an exact obstacle height.
- `curriculum: false` freezes progression at the requested initial difficulty. This is useful for comparing reward changes on the same terrain seed and stage.
- Evaluation only forces `num_envs=1`; rewards, noise, episode length, terrain and curriculum configuration are shared with training. Reset runs the configured curriculum and clears controller history.

Every transition, including terminal ones, is written to `artifacts/joystick/steps.csv` by default. Use `--csv path.csv` to keep separate runs; reusing a path replaces that log. Columns include weighted reward contributions, total reward, episode/termination state, command, measured forward/yaw velocity, uprightness, normalized actions, joint positions and velocities. Indexed joint columns use the canonical order stored in the adjacent `.config.json` sidecar. Reward snapshots are captured **before automatic resets**, and the logged contributions sum to the returned reward. Terms are per-step contributions, not MJLab's episode-averaged metrics.

The sidecar stores resolved task/controller/mapping settings, installed versions, Git/model identity, source-file and lock hashes, and a compiled-model hash. Dirty state is recorded. The Viser page displays the same MJLab environment the policy steps. It does not open a second MuJoCo physics window.

## Repeatable checks without a joystick

```bash
uv run python simulations/joystick_test.py --config configs/flat.json --headless --scripted --steps 1100 --csv artifacts/joystick/flat.csv
uv run python simulations/joystick_test.py --config configs/rough.json --headless --scripted --steps 200 --csv artifacts/joystick/rough.csv
```

The scripted sequence cycles through standstill, forward, forward-plus-turn, and reverse every three simulated seconds. Without `--scripted`, a real joystick is required even in headless mode. Use an episode longer than three seconds to exercise motion; a default episode lasts 20 seconds.

## Training and checks

```bash
uv run walter-train --config configs/flat.json --num-envs 64 --iterations 1000 --device cuda:0
uv run walter-train --config configs/flat.json --num-envs 2 --iterations 1 --device cpu
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

PPO is a minimal initial entry point with local TensorBoard logging; it uses the identical task factory. A successful smoke iteration verifies execution, not convergence or suitable reward magnitudes. Configuration and checkpoints are saved under a new directory in `artifacts/training/`. Legacy Brax checkpoints are not loadable through this entry point.

## Current model and task boundaries

- FS is the runnable task profile. WS geometry compiles, but tasks reject it until separate gains are tuned: provisional FS gains destabilized the much smaller WS model in a CPU test.
- Shared FS geometry, mass, passive waist, damping and reflected inertia are retained. Ideal PD/velocity feedback uses provisional legacy-derived gains and static torque limits. This is not a calibrated torque-speed or hardware actuator model.
- Actions are absolute hip position, knee position relative to the measured pose at policy-step entry, and wheel velocity. Targets are sampled at 50 Hz; feedback/physics run at 250 Hz. Joint order is explicitly resolved by name.
- Actor observations have 68 values; the critic has 77. Actor effort uses correctly mapped joint-space simulated force. The actor's sensor availability is not yet validated for hardware; critic contact features from legacy WaLTRL are not yet ported.
- Noise leaves command/action history untouched, applies per-feature scales to sensor/joint data, and currently perturbs knee sine/cosine features after encoding. This differs from legacy angle-before-encoding noise. The critic is uncorrupted.
- The multiplicative tracking reward and main penalties are implemented, including legacy wheel-center proximity checks. Contact masks, shared profile parameters, corrected force indexing, body-frame velocity conventions, and previous-action timing differ from the old environment. This is not a numerically exact legacy recreation.
- No domain randomization, exported policy runtime, hardware drivers, or sim-to-real validation is claimed yet. See the migration plan for those phases.

Validated: CPU model and contact checks, flat and rough MJLab stepping, a 1,100-step scripted rollout with episode reset, and a one-iteration PPO update. A native MuJoCo window was tried for the joystick and crashed on this Wayland session; interactive viewing now goes through MJLab's Viser viewer. GPU training and policy quality remain unverified. MuJoCo Warp warns that heightfield/capsule and heightfield/box pairs currently generate at most one contact with MULTICCD; consider that limitation when interpreting rough-terrain behavior.
