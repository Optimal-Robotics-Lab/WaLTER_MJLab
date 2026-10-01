"""Heuristic joystick policy and MJLab Viser playback for the velocity task."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

if TYPE_CHECKING:
    import torch


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI without importing simulation or joystick dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Shared velocity task JSON configuration")
    parser.add_argument("--device", default="cpu", help="Torch/Warp device, e.g. cpu or cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--steps", type=int, default=0, help="Stop after N steps; zero runs until quit"
    )
    parser.add_argument(
        "--scripted", action="store_true", help="Use repeatable inputs without a gamepad"
    )
    parser.add_argument("--mapping", type=Path, help="Optional joystick mapping JSON")
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--controller-config", type=Path, help="Optional heuristic gain/rate JSON")
    parser.add_argument("--command-source", choices=("joystick", "sampled"), default="joystick")
    parser.add_argument("--csv", type=Path, default=Path("artifacts/joystick/steps.csv"))
    parser.add_argument("--report-every", type=int, default=100)
    return parser


def load_task_options(path: Path | None) -> dict[str, Any]:
    """Load the training configuration, overriding only evaluation batch size."""
    options = {} if path is None else json.loads(path.read_text())
    if not isinstance(options, dict):
        raise ValueError("Task configuration must be a JSON object")
    return {**options, "num_envs": 1}


def write_run_metadata(path: Path, task: Any, args: argparse.Namespace, model: Any) -> None:
    """Record resolved settings and model/source identity alongside the transition log."""
    from walter_mjlab.controllers import HeuristicControllerConfig, JoystickMapping
    from walter_mjlab.provenance import collect_provenance
    from walter_mjlab.robots import JOINT_NAMES

    controller_options = (
        {} if args.controller_config is None else json.loads(args.controller_config.read_text())
    )
    mapping = JoystickMapping.from_json(args.mapping) if args.mapping else JoystickMapping()
    metadata = {
        **collect_provenance(model),
        "joint_order": list(JOINT_NAMES),
        "heuristic_controller": asdict(HeuristicControllerConfig(**controller_options)),
        "joystick_mapping": asdict(mapping),
        "task": asdict(task),
        "evaluation": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "reward_terms": "Weighted per-step contributions, including timestep scaling from training",
    }
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1)[0].item()
    return float(value)


def diagnostic_row(
    step: int,
    episode: int,
    reward: Any,
    terminated: Any,
    truncated: Any,
    diagnostics: dict[str, Any],
) -> dict[str, float | int]:
    """Flatten a pre-reset diagnostics snapshot for the single evaluation robot."""
    command = diagnostics["command"][0]
    row = {
        "step": step,
        "episode": episode,
        "episode_step": int(_scalar(diagnostics["episode_step"])),
        "reward": _scalar(reward),
        "terminated": int(_scalar(terminated)),
        "truncated": int(_scalar(truncated)),
        "difficulty": _scalar(diagnostics["difficulty"]),
        "command_forward_m_s": _scalar(command[0]),
        "command_yaw_rad_s": _scalar(command[1]),
    }
    row.update(
        {f"reward/{name}": _scalar(value) for name, value in diagnostics["reward_terms"].items()}
    )
    if "measured_velocity" in diagnostics:
        measured = diagnostics["measured_velocity"][0]
        row["measured_forward_m_s"] = _scalar(measured[0])
        row["measured_yaw_rad_s"] = _scalar(measured[1])
        row["upright"] = _scalar(diagnostics["upright"])
    for field in ("action", "joint_position", "joint_velocity"):
        if field in diagnostics:
            row.update(
                {
                    f"{field}/{index}": _scalar(value)
                    for index, value in enumerate(diagnostics[field][0])
                }
            )
    return row


class StepLogger:
    """Write each transition, including terminal transitions, to CSV."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.writer: csv.DictWriter | None = None

    def write(self, row: dict[str, float | int]) -> None:
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()


class LoggedEnv:
    """Record every training-environment transition without replacing its managers."""

    def __init__(self, env: Any, on_transition: Callable[[tuple[Any, ...]], None]) -> None:
        self._env = env
        self._on_transition = on_transition

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, actions: torch.Tensor) -> tuple[Any, ...]:
        transition = self._env.step(actions)
        self._on_transition(transition)
        return transition


class ScriptedJoystick:
    """Cycle standstill, forward, turn, and reverse every three seconds."""

    def __init__(self, step_dt: float) -> None:
        self._step_dt = step_dt
        self._steps = 0

    def __call__(self) -> Any:
        from walter_mjlab.controllers.heuristic import JoystickInput

        phase = int(self._steps * self._step_dt / 3) % 4
        self._steps += 1
        return JoystickInput(
            forward=(0.0, 0.35, 0.35, -0.2)[phase],
            yaw=0.3 if phase == 2 else 0.0,
        )


class HeuristicActionPolicy:
    """MJLab policy adapter: gamepad or scripted input becomes a task action tensor.

    The viewer calls this once per environment step. It does not integrate a
    second simulation or write actuator forces. One environment is required
    because the heuristic owns a single target history.
    """

    def __init__(
        self,
        env: Any,
        controller: Any,
        read_input: Callable[[], Any],
        *,
        command_source: str,
        max_forward_velocity: float,
        max_yaw_velocity: float,
        on_manual_reset: Callable[[], None] | None = None,
    ) -> None:
        if env.num_envs != 1:
            raise ValueError("The heuristic joystick policy controls one environment.")
        self.env = env
        self.controller = controller
        self.read_input = read_input
        self.command_source = command_source
        self.max_forward_velocity = max_forward_velocity
        self.max_yaw_velocity = max_yaw_velocity
        self.on_manual_reset = on_manual_reset
        self.stop_requested = False
        self._awaiting_reset = True

    def reset(self) -> None:
        """Snap hip and knee targets to the measured pose after an environment reset."""
        self.controller.reset(self._joint_positions())
        self._awaiting_reset = False

    def _joint_positions(self) -> Any:
        from walter_mjlab.robots.walter import JOINT_NAMES

        robot = self.env.scene["robot"]
        joint_ids, _ = robot.find_joints(JOINT_NAMES, preserve_order=True)
        return robot.data.joint_pos[0, joint_ids].detach().cpu().numpy().copy()

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        import torch

        from walter_mjlab.controllers.heuristic import targets_to_actions

        del obs
        # episode_length_buf is zero after the initial reset and after MJLab's
        # automatic reset, before the next action is applied.
        if self._awaiting_reset or int(self.env.episode_length_buf[0]) == 0:
            self.reset()
        inputs = self.read_input()
        if inputs.quit:
            self.stop_requested = True
        if inputs.reset:
            self.env.reset()
            self.reset()
            if self.on_manual_reset is not None:
                self.on_manual_reset()
        if self.command_source == "joystick":
            command = self.env.command_manager.get_term("velocity")
            command.set_manual_command(
                inputs.forward * self.max_forward_velocity,
                inputs.yaw * self.max_yaw_velocity,
            )
        joint_positions = self._joint_positions()
        targets = self.controller.step(inputs, self.env.step_dt, joint_positions=joint_positions)
        actions = targets_to_actions(targets, joint_positions)
        return torch.as_tensor(actions, dtype=torch.float32, device=self.env.device).unsqueeze(0)


def _run_viser(env: Any, policy: HeuristicActionPolicy, steps: int) -> None:
    """Drive the training environment from MJLab's browser viewer."""
    from mjlab.viewer import ViserPlayViewer

    class JoystickViserViewer(ViserPlayViewer):
        def is_running(self) -> bool:
            return super().is_running() and not policy.stop_requested

    viewer = JoystickViserViewer(env, policy)
    port = viewer._server.get_port()
    print(f"Open the MJLab viewer at http://127.0.0.1:{port}")
    viewer.run(num_steps=None if steps == 0 else steps)


def run(args: argparse.Namespace) -> None:
    """Evaluate the heuristic through MJLab's environment step and Viser viewer."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    from walter_mjlab.controllers import (
        HeuristicControllerConfig,
        HeuristicJoystickController,
        JoystickMapping,
        PygameJoystick,
    )
    from walter_mjlab.tasks.velocity import VelocityTaskConfig, make_env_cfg
    from walter_mjlab.tasks.velocity.mdp.commands import VelocityCommand

    task = VelocityTaskConfig(**load_task_options(args.config))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    cfg = make_env_cfg(task)
    cfg.viewer.distance = 1.8
    cfg.viewer.elevation = -25.0
    cfg.viewer.azimuth = 135.0
    env = ManagerBasedRlEnv(cfg, device=args.device)
    joystick = None
    try:
        write_run_metadata(args.csv.with_suffix(".config.json"), task, args, env.sim.mj_model)
        if args.scripted:
            read_input: Callable[[], Any] = ScriptedJoystick(env.step_dt)
        else:
            mapping = JoystickMapping.from_json(args.mapping) if args.mapping else JoystickMapping()
            joystick = PygameJoystick(args.joystick_index, mapping)
            print(f"Joystick: {joystick.name}")
            read_input = joystick.poll
        env.reset()
        controller_options = (
            {} if args.controller_config is None else json.loads(args.controller_config.read_text())
        )
        controller = HeuristicJoystickController(HeuristicControllerConfig(**controller_options))
        command = env.command_manager.get_term("velocity")
        if not isinstance(command, VelocityCommand):
            raise TypeError("The velocity task must expose its manual velocity command term")
        if args.command_source == "sampled":
            command.clear_manual_command()
        episode = 0
        step = 0
        episode_reward = 0.0

        def on_manual_reset() -> None:
            nonlocal episode, episode_reward
            print(f"Manual reset: episode={episode} partial_return={episode_reward:.6f}")
            episode += 1
            episode_reward = 0.0

        policy = HeuristicActionPolicy(
            env,
            controller,
            read_input,
            command_source=args.command_source,
            max_forward_velocity=task.max_forward_velocity,
            max_yaw_velocity=task.max_yaw_velocity,
            on_manual_reset=on_manual_reset,
        )
        policy.reset()
        with args.csv.open("w", newline="") as stream, torch.inference_mode():
            logger = StepLogger(stream)

            def on_transition(transition: tuple[Any, ...]) -> None:
                nonlocal step, episode, episode_reward
                _, reward, terminated, truncated, extras = transition
                step += 1
                row = diagnostic_row(
                    step, episode, reward, terminated, truncated, extras["diagnostics"]
                )
                logger.write(row)
                episode_reward += float(row["reward"])
                done = row["terminated"] or row["truncated"]
                if step % args.report_every == 0 or done:
                    terms = " ".join(
                        f"{key[7:]}={value:.4g}"
                        for key, value in row.items()
                        if key.startswith("reward/")
                    )
                    print(
                        f"step={step} episode={episode} reward={row['reward']:.5f} "
                        f"difficulty={row['difficulty']:.3f} {terms}"
                    )
                if done:
                    print(
                        f"Episode return={episode_reward:.6f} terminated={row['terminated']} "
                        f"truncated={row['truncated']}"
                    )
                    episode += 1
                    episode_reward = 0.0

            if args.headless:
                while args.steps == 0 or step < args.steps:
                    actions = policy(torch.zeros(1))
                    if policy.stop_requested:
                        break
                    on_transition(env.step(actions))
            else:
                _run_viser(LoggedEnv(env, on_transition), policy, args.steps)
    finally:
        if joystick is not None:
            joystick.close()
        env.close()


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for standalone and installed use."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps < 0 or args.report_every < 1:
        parser.error("--steps must be nonnegative and --report-every must be positive")
    try:
        run(args)
    except KeyboardInterrupt:
        print("Joystick evaluation interrupted.")


if __name__ == "__main__":
    main()
