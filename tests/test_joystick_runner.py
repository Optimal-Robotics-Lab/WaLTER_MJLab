"""Regression tests for pre-reset evaluation logging and shared configuration."""

import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from walter_mjlab.controllers.heuristic import HeuristicJoystickController, JoystickInput
from walter_mjlab.evaluation.joystick import (
    HeuristicActionPolicy,
    StepLogger,
    build_parser,
    diagnostic_row,
    load_task_options,
)


def test_evaluation_preserves_training_configuration(tmp_path: Path) -> None:
    path = tmp_path / "task.json"
    options = {
        "num_envs": 4096,
        "curriculum": True,
        "noise": True,
        "terrain": "rough",
        "reward_weights": {"tracking": 2.0},
    }
    path.write_text(json.dumps(options))
    assert load_task_options(path) == {**options, "num_envs": 1}


def test_terminal_reward_terms_are_logged_without_rescaling() -> None:
    snapshot = {
        "episode_step": 17,
        "difficulty": 0.6,
        "command": [[0.8, -0.2]],
        "reward_terms": {"tracking": 0.03, "failure": -1.0},
    }
    row = diagnostic_row(17, 2, -0.97, True, False, snapshot)
    stream = io.StringIO()
    StepLogger(stream).write(row)
    logged = next(csv.DictReader(io.StringIO(stream.getvalue())))
    assert float(logged["reward/tracking"]) == 0.03
    assert float(logged["reward/failure"]) == -1.0
    assert logged["episode_step"] == "17"
    assert logged["terminated"] == "1"
    assert logged["truncated"] == "0"
    assert float(logged["command_yaw_rad_s"]) == -0.2


def test_heuristic_policy_emits_one_environment_action() -> None:
    class Robot:
        def __init__(self) -> None:
            self.data = SimpleNamespace(joint_pos=torch.zeros(1, 16))

        def find_joints(
            self, names: tuple[str, ...] | list[str], preserve_order: bool = True
        ) -> tuple[list[int], list[str]]:
            del preserve_order
            ordered = list(names)
            return list(range(len(ordered))), ordered

    command = SimpleNamespace(value=None)

    def set_manual_command(forward: float, yaw: float) -> None:
        command.value = (forward, yaw)

    command.set_manual_command = set_manual_command
    resets = {"count": 0}

    class Env:
        num_envs = 1
        device = "cpu"
        step_dt = 0.02
        episode_length_buf = torch.zeros(1, dtype=torch.long)
        scene = {"robot": Robot()}
        command_manager = SimpleNamespace(get_term=lambda name: command)

        def reset(self) -> None:
            resets["count"] += 1

    env = Env()
    policy = HeuristicActionPolicy(
        env,
        HeuristicJoystickController(),
        lambda: JoystickInput(forward=0.5),
        command_source="joystick",
        max_forward_velocity=3.0,
        max_yaw_velocity=1.0,
    )
    actions = policy(None)
    assert tuple(actions.shape) == (1, 16)
    torch.testing.assert_close(actions[0, 8:], torch.full((8,), 0.5))
    assert command.value == (1.5, 0.0)

    policy.read_input = lambda: JoystickInput(reset=True)
    env.episode_length_buf[:] = 3
    policy(None)
    assert resets["count"] == 1

    policy.read_input = lambda: JoystickInput(quit=True)
    policy(None)
    assert policy.stop_requested

    env.num_envs = 2
    try:
        HeuristicActionPolicy(
            env,
            HeuristicJoystickController(),
            lambda: JoystickInput(),
            command_source="joystick",
            max_forward_velocity=3.0,
            max_yaw_velocity=1.0,
        )
        raise AssertionError("A multi-environment heuristic policy should be rejected.")
    except ValueError:
        pass


def test_cli_exposes_hardware_free_cpu_smoke_mode() -> None:
    args = build_parser().parse_args(["--headless", "--scripted", "--steps", "4"])
    assert args.device == "cpu"
    assert args.steps == 4
    assert args.scripted and args.headless
    assert args.command_source == "joystick"
