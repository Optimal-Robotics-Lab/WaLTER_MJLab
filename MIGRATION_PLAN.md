# WaLTRL → MJLab: migration and controller portability plan

Status: initial MJLab environment implemented October 1, 2026. Interactive control uses MJLab's Viser viewer. Later migration phases remain planned. Prepared September 30, 2026. This document preserves design decisions and the remaining implementation sequence. Day-to-day file locations are in [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md).

## Decision — adopt MJLab, October 1, 2026

This repository is a switch onto MJLab. The implementation uses MJLab's manager-based environment, Viser play viewer, and RSL-RL training entry points. Legacy WaLTRL remains a reference for robot behavior worth keeping: sixteen driven joints, hip position / knee-relative / wheel-velocity commands, a velocity-tracking task, and flat or rough terrain.

JAX, Brax, and MuJoCo Playground execution details are not part of the new stack. Do not reproduce their array lifecycle, PPO internals, or a second native MuJoCo window for interactive driving. The heuristic joystick is an MJLab policy: it reads the gamepad and returns the task action tensor consumed by `env.step()`. MJLab's `ViserPlayViewer` is the online viewer for that loop.

## Implementation checkpoint — October 1, 2026

The project now has packaging and a dependency lock, the real pinned `modeling/` submodule, an FS robot adapter, a manager-based flat/rough velocity task, portable GenZero-style heuristic controls, and `simulations/joystick_test.py`. A minimal PPO entry point and joystick evaluation both consume `VelocityTaskConfig` through `make_env_cfg`; the joystick supplies actions through `env.step()` rather than writing motor torques directly.

CPU validation: 28 tests passed, including model compilation/contact settling, controller mappings, reward-total reconciliation at terminal steps, partial reset isolation, frozen knee targets, and rough-terrain stage selection. A 1,100-step scripted joystick rollout completed an episode/reset; one PPO iteration completed with two environments. A native MuJoCo passive viewer was tried for the joystick and segfaulted on this Wayland session; that window is retired. Interactive playback belongs to MJLab's Viser viewer. GPU execution remains unverified. These checks establish execution of the MJLab task, not policy quality or a line-by-line legacy recreation.

Implementation differences to keep visible:

- The initial task uses shared FS morphology and a passive waist, with provisional legacy-derived PD/velocity gains and static effort limits. WS morphology compiles but its task is blocked pending separate tuning after its standstill test failed with FS gains.
- Corrected joint-space force indexing, current-action history semantics, canonical name-based ordering, and positive body-Z yaw are used. Actor dimension is 68; critic dimension is 77, without legacy wheel-contact channels.
- Feature-specific observation noise leaves commands/history clean; knee sine/cosine features currently receive noise after encoding. This is an intentional initial simplification, not exact legacy noise semantics.
- Tracking remains multiplicative. Wheel overlap is the legacy center-distance heuristic. Shared model contact behavior and FS physical parameters differ from the original embedded model.
- A new tracking-based curriculum adjusts command range and rough-terrain stage on reset. A fixed-stage configuration and seeded terrain support repeatable reward inspection. Full domain randomization is still pending.
- The separately installable shared core, complete artifact export, ROS2 adapters, and physical motor calibration remain future work. The heuristic module is simulator-independent now, without prematurely creating that extra package.
- MJLab is pinned to the released 1.6.0 package; the lock resolves its actual dependencies (including RSL-RL 5.4.2). The earlier reference checkout had different dependency declarations. Implementation was tested against the installed package.

See [README.md](README.md) for runnable commands, control mapping, diagnostics, and limitations. Phase 0's executable legacy baseline and full physical parity work remain incomplete; phases 1–4 have an initial vertical implementation, not completed acceptance of every migration gate.

Coding conventions: follow [claude.md](claude.md), including flat layouts for both the RL package and the separately installable shared core. Package trees below are proposed, not implemented.

## 1. Objective and architecture decision

Build this repository as WaLTER's RL training and policy evaluation project using MJLab. Follow GenZero's broad organization: an installable controller package, a `modeling/` Git submodule, and separate simulation entry points. Keep robot descriptions independent of control methods. Establish a small, versioned controller contract so RL, GenZero heuristics, and future control methods can use the same simulation and ROS2 execution infrastructure.

The transfer unit is a controller artifact plus its explicit observation/action contract, not an MJLab environment or a training checkpoint alone. The desired workflow is:

```text
WaLTER_Modeling ── robot description, physical parameters, names, frames
       │
       ├── MJLab training ── exported policy bundle
       │                            │
       └── native MuJoCo evaluation ┤
                                    ▼
                       portable controller runtime
                                    │
                  ROS2 controller adapter / supervisor
                                    │
                         ros2_control interfaces
                           ┌────────┴────────┐
                    MuJoCo backend     robot backend
```

Interactive training and joystick evaluation stay inside MJLab, using its Viser viewer. The native MuJoCo box in this diagram is a possible later deployment backend for an exported controller, not a second training simulation and not the joystick window.

Simulation-to-hardware transfer should change the backend and deployment profile while preserving controller math and artifact identity. It still requires calibration, timing measurements, and staged validation; matching APIs alone does not establish physical equivalence.

## 2. Sources examined

Reference checkouts were inspected outside this initially empty workspace. Pin findings to these commits rather than assuming future branches remain identical:

| Repository | Inspected commit | Role |
| --- | --- | --- |
| [WaLTRL](https://github.com/dmj17b/WaLTRL/tree/2b279b08f97187fe741b595b56da422fbed2f3ec) | `2b279b08f97187fe741b595b56da422fbed2f3ec` | Existing RL task and training behavior |
| [WaLTER_GenZero](https://github.com/Optimal-Robotics-Lab/WaLTER_GenZero/tree/1f1972f1cb7f4632bd7672b858330647d4ef6728) | `1f1972f1cb7f4632bd7672b858330647d4ef6728` | Requested organization and heuristic controller reference |
| [WaLTER_Modeling](https://github.com/Optimal-Robotics-Lab/WaLTER_Modeling/tree/cb01a463809b32e59be8d702bcb9085f31eaf9d3) | `cb01a463809b32e59be8d702bcb9085f31eaf9d3` | Shared parametric robot generator |
| [MJLab](https://github.com/mujocolab/mjlab/tree/f135c1daa0f278bd19e323c2b9f256ca541ae2c5) | `f135c1daa0f278bd19e323c2b9f256ca541ae2c5` | Actual APIs and dependency declarations |

GenZero's `modeling` gitlink already pins the same modeling commit inspected above. MJLab's inspected `pyproject.toml` declares version 1.6.0; this is a source observation, not a claim that this exact revision is the best released dependency for the target machine. Select and lock a tested release/commit during bootstrap.

### Existing WaLTRL behavior to preserve deliberately

Primary files: `environment/BaseEnv.py`, `environment/RoughTerrainEnv.py`, `configs/base_config.py`, `train/base_train.py`, `modeling/GenModel.py`, and the policy evaluation scripts.

- The previous project is JAX + MuJoCo Playground/MJX with Brax PPO. The new project uses PyTorch through MJLab's managers, viewer, and RSL-RL integration. Keep the robot-level behavior described below where it still matches the intended task. Leave JAX primitives and Brax training code behind.
- Sixteen driven joints: four hips, four knees, eight wheels. The waist is passive when enabled. Actions are indexed using compiled actuator IDs, while several observation arrays use a separately assembled hip/knee/wheel order.
- Hip actions request absolute positions (`1.25 * action`); knee actions request an offset from the measured position at the start of the policy step (`q_start + 5.5 * action`), held across physics substeps; wheel actions request velocities (`60 * action`). Low-level PD/velocity laws execute every physics step.
- Configuration requests a 20 ms policy step and a 4 ms simulation step, with 1,000 policy steps per episode. Verify actual compiled model timestep and substep count before declaring a 50 Hz / 250 Hz baseline.
- Actor inputs include forward/yaw commands, previous action, accelerometer, gyro, hip positions, knee sine/cosine, joint velocities, and actuator-derived torques. Critic inputs additionally include privileged state and wheel contact measurements. The actor is nominally 68-dimensional for the 16-action model; freeze actual shapes from an executable baseline.
- Rewards include a product of linear tracking, orientation tracking, and yaw tracking, plus penalties. Preserve the product explicitly: replacing it with three additive reward terms changes the task. Orientation and T-pose penalties are computed/logged but are not both included in the returned reward expression.
- Rough terrain is generated at construction; `_reset_terrain` currently does not randomize it per episode. Do not describe the existing behavior as a terrain curriculum.
- PPO uses separate actor/critic observations and `[512, 256, 256, 128]` hidden layers, observation normalization, and W&B. The README documents a manual Brax/JAX patch, which should not carry into the new training stack.

### Compatibility issues found by static inspection

1. **The shared model is not a drop-in replacement.** Wheel actuators use names such as `fl_wheel1_joint`, while WaLTRL expects `fl_wheel1`. The shared generator lacks WaLTRL's active IMU/velocity/up-vector and wheel-touch sensor setup, has different collision behavior, and unconditionally creates the passive waist joint. WaLTRL can omit that joint.
2. **Physical profiles differ.** Legacy `2_7_Scale` and shared `FS` have similar major dimensions but different thigh/shin masses, starting height, contact parameters, and motor settings. Shared `WS` is another morphology. Establish an explicit legacy comparison profile; do not silently substitute FS or WS and attribute all changes to MJLab.
3. **Motor realism differs.** WaLTRL's `motor_model` simply returns desired torque, although its generated actuators declare static force ranges. GenZero implements a torque-speed limiter but couples it to `MjData`, plotting, and per-motor objects. It also uses a different wheel feedback gain convention. Shared motor parameters need explicit speed units and motor-shaft versus joint-side semantics before reuse.
4. **Torque indexing needs correction.** WaLTRL indexes `qfrc_actuator` with actuator IDs in observations and penalties. That array is indexed by generalized DOF, so the free base and optional waist make this mapping incorrect. Build the correct actuator→joint→DOF mapping; record the behavior change separately from the baseline.
5. **Observation timing needs a declared contract.** `step()` constructs observations before updating `prev_action`, so the returned actor observation contains an older action than a conventional next-step observation. Command resampling also occurs within the step. Capture existing behavior, then explicitly version corrected timing.
6. **Model names and order cannot be inferred from list position.** GenZero hardcodes actuator indices; MJLab can order actuators by target definition. Validate names and build explicit permutations for policy order, simulator order, and hardware order.
7. **Hardware observability is incomplete.** Simulated actuator force and contact truth are not automatically available on the robot. Every actor feature needs an identified sensor or estimator, units, frame, and timing definition. Privileged quantities can remain critic-only.

These are source findings, not executed failure reproductions. Keep a migration change ledger distinguishing exact legacy behavior, confirmed bug fixes, and intentional task redesign.

## 3. Repository layout and ownership

Proposed initial layout, retaining GenZero's top-level package and simulation organization:

```text
WaLTRL_mjLab/
├── pyproject.toml
├── uv.lock
├── .gitmodules
├── modeling/                    # WaLTER_Modeling, pinned Git submodule
├── walter_mjlab/                # RL-specific package
│   ├── robots/                 # model loading and MJLab entity adapter
│   ├── tasks/velocity/
│   │   ├── config/             # flat, rough, legacy, deployment-oriented
│   │   └── mdp/                # observations, actions, rewards, events, commands
│   ├── training/               # PPO configuration and thin runner integration
│   ├── controllers/            # RL policy controller adapter
│   ├── evaluation/             # scenarios and metrics
│   └── cli/                    # train, play, evaluate, export
├── packages/walter_control/     # separately installable, lightweight shared core
│   ├── pyproject.toml
│   └── walter_control/         # flat package: contracts, runtime, conformance
├── simulations/                # native MuJoCo controller runner and examples
├── configs/                    # task and deployment profiles, not copied geometry
├── scripts/                    # bootstrap and artifact utilities
├── tests/                      # unit, model, integration, export parity
├── docs/                       # architecture, schemas, migration change ledger
└── artifacts/                  # ignored generated runs/bundles; small fixtures tracked
```

`walter_control` is a proposed package name, not an existing repository. Incubate it here as an independently installable package, then extract it to an organization-owned shared repository when GenZero and the ROS2 stack consume it. That extraction should not change its public imports or artifact schema. Avoid building a large general-purpose plugin framework before there are multiple consumers.

Ownership boundaries:

| Owner | Owns | Must not depend on |
| --- | --- | --- |
| WaLTER_Modeling | Geometry, masses/inertias, joint topology/names, physical motor data, sensor mounting frames; future robot-description export | RL rewards, PPO, policy observation layouts, ROS execution logic |
| Shared control core | State/command schemas, policy preprocessing, action decoding, controller lifecycle, artifact loader and conformance fixtures | MJLab, Brax, training code, simulator objects, ROS messages |
| This RL repository | MJLab adapters, training tasks, randomization, rewards, PPO, RL export/evaluation | Hardware drivers in the training loop |
| Future ROS2 stack | Hardware transport/calibration, state estimation, ros2_control adapters, lifecycle/supervision, launch/configuration | Full RL training environment |
| GenZero/future controller projects | Their own controller algorithms behind the shared contract | RL-specific observation assumptions |

Use `modeling/` as a real submodule pointing at the requested repository, installed as a local package dependency. Clone with submodules, pin its gitlink, and expose a clear bootstrap error when it is absent. Prefer HTTPS for a portable submodule URL unless organization access policy requires SSH. Do not copy the generator into this project or silently track a moving modeling branch.

Use the existing `WaLTER_Modeling.GenerateModel` and `config_dir()` package API instead of working-directory-relative paths. Resolve custom profiles explicitly. Model changes needed for parity should become focused upstream changes to the modeling repository; temporary adapter transformations must be named, tested, and recorded in provenance.

Separate physical motor properties from controller tuning gains over time. Preserve compatibility with current motor YAML while adding a validated schema with SI units, gear convention, limits, and profile version. MJCF and future URDF/Xacro should derive from common physical data. Do not maintain two manually edited robot descriptions; ROS-specific hardware declarations belong in the ROS stack.

## 4. MJLab implementation strategy

The inspected framework exposes `ManagerBasedRlEnvCfg`, `SceneCfg`, `EntityCfg` with a `spec_fn`, manager dictionaries, `register_mjlab_task`, RSL-RL integration, and ONNX export. Its [migration guide](https://mujocolab.github.io/mjlab/main/source/migration_isaac_lab.html) describes manager composition, but WaLTRL requires a JAX-to-PyTorch port beyond that guide.

| Existing component | Planned destination |
| --- | --- |
| Embedded `GenModel` | Shared generator through an MJLab `EntityCfg` adapter |
| Monolithic `BaseEnv` | Manager-based config plus small MDP terms |
| `_simulation_step` | Action processing once per policy step; actuator feedback each physics substep |
| Actor/value builders | Explicit actor and privileged critic observation groups |
| `_get_reward` | Named terms, preserving multiplicative task term and actual returned penalties |
| Command sampler | Forward/yaw command term with zero-command probabilities and duration semantics |
| Reset and rough terrain subclass | Reset events, scene terrain configuration, later curriculum |
| Brax PPO and checkpoints | RSL-RL PPO and new native checkpoints |
| Interactive test scripts | MJLab `ViserPlayViewer` plus a heuristic policy that returns task actions |

Implementation details that matter:

- Construct a fresh robot-only `MjSpec` per entity build. Leave terrain and lights with the scene. Add required sensors/sites with defined semantics. Ensure generated XML actuators and MJLab-created actuators are not duplicated; choose wrapping existing actuators or replacing them in the adapter explicitly.
- Start with separate hip-position, knee-relative-position, and wheel-velocity action terms. Freeze knee targets at policy-step entry. Recomputing the relative knee target every substep would produce different behavior.
- Maintain explicit action and observation order independent of MJLab's internal order. For new policies use a documented canonical order; retain an explicit permutation for legacy comparisons.
- Validate timestep, decimation, clipping, PD update order, reset timing, and command timing with trace fixtures. Use 20 ms policy period / 4 ms physics period as the initial requested configuration, subject to baseline verification.
- MJLab scales rewards by step duration by default. For an initial legacy comparison use its inspected `scale_rewards_by_dt=False`, or an equivalently verified mapping. Later reward normalization is a separate experiment.
- Map actor/critic groups through the pinned RSL-RL configuration. Preserve normalization in export. PPO batch/rollout settings do not have a one-to-one translation from Brax; calculate samples per update and tune after correctness checks.
- Use small environment counts first; then measure throughput/memory before scaling toward 4,096 environments. Do not copy MJX contact-buffer settings blindly into Warp/MJLab configuration.
- Provide project CLIs that import/register WaLTER tasks before invoking framework runners. Do not assume an external-package auto-discovery hook exists; the inspected task registry uses explicit registration.
- Keep W&B optional with local logging available. Record resolved configs, seeds, dependency lock, robot profile, model commit, generated model hash, and code commit with every run.
- Existing Brax parameter files remain historical baselines. Do not claim native compatibility with RSL-RL checkpoints. Any future weight conversion requires matching architecture, activation, normalization, action distribution, and numerical inference tests; retraining is the default plan.

## 5. Portable controller contract

Define language-neutral schemas and Python reference types before coupling inference to ROS. The shared contract must support heuristic and optimization-based controllers as well as RL.

**Robot state:** timestamp and clock domain, ordered joint names, joint positions/velocities, optional measured/estimated effort with validity/provenance, IMU orientation/angular velocity/specific force with frames, optional base-state estimates, sensor freshness and faults. Distinguish passive joints from actuated joints.

**Task command:** forward speed and yaw rate initially, timestamp/expiry, source/mode. Expand with a versioned schema when new controllers require additional commands.

**Controller lifecycle:** configure/load, validate capabilities, reset internal state, step(state, command, dt), deactivate. Recurrent state and observation history are explicit; reset them after startup, episode reset, fault recovery, or controller switch.

**Joint command:** ordered joint names, requested control mode per joint, desired position/velocity, optional feedforward effort and gains, timestamp/validity. An RL adapter decodes network actions into this physical command. A heuristic or MPC controller produces the same command directly. Backend capability validation rejects unsupported modes.

**Coordinate and unit rules:** SI, explicit joint signs/zero offsets, base/IMU/world frames, quaternion ordering, angle wrapping, and gravity convention. Use ROS conventions at the ROS boundary; convert MuJoCo quaternion storage explicitly. IMU specific force must not be confused with gravity-compensated acceleration.

**Observation contract:** ordered fields, shape/dtype, frame/unit, scaling/clipping, noise applied during training, normalizer, history order, previous-action semantics, missing-data behavior. Actor inputs must be reproducible from robot sensors/estimators. If effort sensing is unavailable, remove that feature or train with the same validated estimate used at deployment; silently substituting zeros is not parity.

**Action contract:** exact ordering, bounds, squashing/clipping, scales/offsets, absolute versus measured-relative targets, feedback gains, command-hold policy, and update rates. The export must identify whether it emits raw normalized actions or physical commands, avoiding double scaling.

Share schemas, parameters, and conformance fixtures between batched Torch training and lightweight runtime implementations. They need not share array libraries, but they must produce numerically consistent outputs for identical recorded inputs.

## 6. Model fidelity and deployment preparation

After the legacy task is understood, create a separately versioned deployment-oriented task:

- Reconcile shared FS/WS profiles with the intended physical robot. Specify waist mode and limits as morphology settings; do not infer them from comments or a zero-valued range.
- Specify torque-speed curves in joint-side units after explicit rotor conversion, continuous/peak limits, transmission efficiency, reflected inertia, and friction. Avoid multiplying gear ratio twice. Confirm RPM versus rad/s from authoritative motor data or measurements.
- Evaluate MJLab's DC motor actuator against requirements rather than copying GenZero's limiter. Its behavior, especially braking, must match the selected actuator model. Validate saturation at zero speed, near no-load speed, reverse motion, and braking.
- Match the actual hardware command mode: drive-side position/velocity loops versus host-side effort control. Equivalent nominal gains do not guarantee equivalent dynamics; identify delay, bandwidth, saturation, and update rates.
- Randomize physically justified ranges for mass/COM, wheel contact/friction, motor strength, gain uncertainty, latency, sensor bias/noise, action delay and dropouts. Log sampled parameters. Calibrate ranges from measurements when available.
- Keep contact truth, terrain truth, and exact base velocity in the critic unless a deployment estimator supplies the corresponding actor signal.
- Benchmark standstill, forward/reverse travel, turning, command reversals, slopes, steps, payloads, perturbations, and recovery. Track tracking error, falls, effort saturation, slip, smoothness, and latency—not reward alone.

## 7. Policy artifact and ROS2 path

Export a versioned bundle containing:

```text
policy.onnx
manifest.json              # schema/version, hashes, robot/profile compatibility
observations.json          # order, preprocessing, normalization/history contract
actions.json               # physical decoding, order, modes, gains, timing
normalization.*            # only if not embedded; ownership explicitly stated
robot_profile.json         # model commit/config hashes and runtime requirements
evaluation.json            # scenarios, metrics, validation status
fixtures/                  # representative states → observations → actions/commands
```

Keep training checkpoints separate from deployable bundles. Export only the inference policy and necessary state. Verify ONNX output against the training actor and verify the complete observation-to-command path, including normalization exactly once. The lightweight runtime should install without MJLab, CUDA, Brax, W&B, or ROS; use ONNX Runtime as the initial deployment candidate, with target-device latency validation before commitment.

The future ROS2 workspace should provide a controller adapter, robot description integration, a MuJoCo simulation hardware adapter, and physical hardware interfaces. ROS2's [ros2_control architecture](https://control.ros.org/jazzy/doc/getting_started/getting_started.html) separates controller updates from hardware read/write operations; use that boundary to switch backends and control methods.

Initially run exported inference outside the strict motor update loop with timestamped command handoff if needed. Decide between this arrangement and in-process inference from measured worst-case latency, jitter, target CPU, and controller requirements. Python inference and ordinary topic delivery must not be assumed to satisfy a hard real-time motor loop.

Use one active command owner, validated controller switching, lifecycle activation checks, stale-state/command handling, and bounded physical commands. Define the robot-specific fallback behavior explicitly; zero torque or holding position is not universally a safe response. Hardware emergency-stop behavior must remain independent of the learned policy.

Validation sequence:

1. Batched MJLab evaluation with the training actor.
2. Exported actor in native MuJoCo using the shared runtime and matching physical model.
3. The same bundle and ROS2 adapter against the MuJoCo backend, with simulated clock and reset handling.
4. Recorded-data replay and hardware-in-the-loop where available; verify units, signs, calibration, freshness, and measured inference/control deadlines.
5. Restrained low-energy hardware tests followed by a progressively expanded operating envelope.

Use the same scenario definitions and metrics where possible for RL, GenZero, and future controllers. The shared control interface must not force heuristics to manufacture an RL observation vector.

## 8. Implementation phases and acceptance gates

| Phase | Deliverables | Gate before proceeding |
| --- | --- | --- |
| 0: Baseline capture | Source pins; executable legacy reference; model/observation/action inventory; fixed-seed state traces; bug/change ledger | Actual model timing, topology, contact behavior, and feature/action order documented; unresolved legacy execution problems explicit |
| 1: Bootstrap and contracts | Installable project; locked dependencies; modeling submodule; separately installable shared core; named joint maps | Clean checkout initializes reproducibly; model profiles compile; all name mappings resolve; no training dependencies in core |
| 2: Shared model integration | Robot-only spec factory; sensors; explicit actuator ownership; legacy comparison profile; focused modeling changes | CPU model checks and small MJLab batched smoke tests pass; physical/model differences explained |
| 3: Flat task port | Manager terms; low-level control; command/reset/termination handling; PPO config | Observation/action/reward fixtures match declared semantics; reset isolation and timeout bootstrap behavior checked; short training produces finite outputs |
| 4: Rough terrain and realism | Terrain scenarios/curriculum; measured motor model; noise/delay/randomization; corrected deployable observation schema | Multi-seed evaluation demonstrates defined behavior; corrected task improvements distinguished from framework effects |
| 5: Export and sim-to-sim | Portable bundle; ONNX runtime controller; native MuJoCo runner; parity suite | Same recorded inputs yield observations and commands within specified tolerances; closed-loop scenarios pass without training dependencies |
| 6: Shared ROS2 execution | Extract/shared core consumer; ROS adapter and simulated hardware interface; GenZero adapter | Same bundle runs through ROS2 simulation; switching/fault/timing tests pass; heuristic controller uses same execution interfaces |
| 7: Hardware transfer | Calibrated profile; capability and timing verification; staged trials | Hardware-specific limits, sensor availability, fallback behavior, and operating envelope validated |

Phases 0–5 belong primarily to this RL project. Phases 6–7 are coordinated future work across the shared-control, modeling, GenZero, and hardware repositories. Build the interfaces now, but do not populate this RL repository with a speculative hardware stack.

Testing should target costly integration failures: joint permutations, free-base/waist DOF indexing, unit conversion, relative-action timing, motor saturation, observation provenance, reward timestep scaling, reset/history isolation, ONNX parity, and stale-state handling. Use CPU tests for contracts/model compilation, GPU tests for vectorized stepping and training, and target-device tests for runtime deadlines. Exact long-horizon trajectories across physics backends are not expected; use short trace checks plus statistical closed-loop metrics.

## 9. Decisions to resolve at their dependency point

These do not prevent repository bootstrap or the initial flat-task port:

- Which morphology is the first hardware target: shared FS, WS, or a new calibrated profile? Default to an explicitly labeled legacy comparison profile for migration experiments.
- Is the waist fixed or passive for that target, and is its angle measured?
- What motor controllers, command modes, sensor set, encoder conventions, and effort estimates are actually available?
- What onboard computer, ROS2 distribution, control rates, and latency budget will deployment use? Do not bind this plan to a ROS distribution before checking the hardware stack.
- Which historical checkpoint and resolved configuration define the legacy baseline? Existing checkpoint names alone are insufficient provenance.
- Where should the organization publish the shared control core once its second consumer is implemented?

Next work: drive the heuristic from the Viser viewer and inspect measured-versus-requested velocity and individual reward terms, then tune the MJLab task before long PPO experiments. An executable legacy baseline is optional reference material, not a gate for using the MJLab environment. Continue export/runtime and ROS2 work only after these environment contracts are stable.
