"""Integration checks for the exact environment used by joystick and PPO."""

import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv

from walter_mjlab.tasks.velocity import VelocityTaskConfig, make_env_cfg
from walter_mjlab.tasks.velocity.mdp.diagnostics import update_difficulty


@pytest.fixture
def env() -> ManagerBasedRlEnv:
    environment = ManagerBasedRlEnv(
        make_env_cfg(VelocityTaskConfig(num_envs=2, noise=False, episode_length_s=0.04)),
        device="cpu",
    )
    environment.reset()
    yield environment
    environment.close()


@pytest.mark.integration
def test_observations_commands_and_terminal_reward_snapshot(env: ManagerBasedRlEnv) -> None:
    command = env.command_manager.get_term("velocity")
    command.set_manual_command(0.5, -0.2)
    action = torch.zeros((2, 16))
    obs, _, done, timeout, _ = env.step(action)
    assert obs["actor"].shape == (2, 68)
    assert obs["critic"].shape == (2, 77)
    assert torch.isfinite(obs["critic"]).all()
    assert not (done | timeout).any()
    obs, reward, _, timeout, extras = env.step(action)
    snapshot = extras["diagnostics"]
    assert timeout.all()
    assert (snapshot["episode_step"] == 2).all()
    assert (env.episode_length_buf == 0).all()
    torch.testing.assert_close(sum(snapshot["reward_terms"].values()), reward)
    torch.testing.assert_close(snapshot["command"], torch.tensor([[0.5, -0.2]]).repeat(2, 1))
    torch.testing.assert_close(command.command, snapshot["command"])
    assert torch.all(obs["actor"][:, 2:18] == 0)  # History reset, including terminal action.


@pytest.mark.integration
def test_relative_target_is_frozen_and_partial_reset_is_isolated(env: ManagerBasedRlEnv) -> None:
    term = env.action_manager.get_term("joint_targets")
    robot = env.scene["robot"]
    before = robot.data.joint_pos[:, term.joint_ids[4:8]].clone()
    action = torch.zeros((2, 16))
    action[:, 4:8] = 0.1
    term.process_actions(action)
    target = before + 0.55
    term.apply_actions()
    torch.testing.assert_close(term.position_targets[:, 4:], target)
    env.step(action)
    torch.testing.assert_close(term.position_targets[:, 4:], target)
    command = env.command_manager.get_term("velocity")
    remaining_time = command.time_left[1].clone()
    joint_state = robot.data.joint_pos[1].clone()
    other_target = term.position_targets[1].clone()
    env.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(robot.data.joint_pos[1], joint_state)
    torch.testing.assert_close(command.time_left[1], remaining_time)
    torch.testing.assert_close(term.position_targets[1], other_target)
    assert env.episode_length_buf.tolist() == [0, 1]


@pytest.mark.integration
def test_curriculum_uses_complete_episode_and_preserves_non_reset_env(
    env: ManagerBasedRlEnv,
) -> None:
    command = env.command_manager.get_term("velocity")
    command.difficulty[:] = 0.4
    command.tracking_count[:] = 10
    command.tracking_sum[:] = 10
    env.episode_length_buf[:] = env.max_episode_length
    update_difficulty(env, torch.tensor([0]))
    torch.testing.assert_close(command.difficulty, torch.tensor([0.5, 0.4]))
    env.termination_manager.terminated[0] = True
    update_difficulty(env, torch.tensor([0]))
    torch.testing.assert_close(command.difficulty, torch.tensor([0.4, 0.4]))


@pytest.mark.integration
def test_rough_terrain_resets_to_selected_fixed_difficulty() -> None:
    env = ManagerBasedRlEnv(
        make_env_cfg(
            VelocityTaskConfig(
                terrain="rough",
                curriculum=False,
                initial_difficulty=0.6,
                noise=False,
            )
        ),
        device="cpu",
    )
    try:
        env.reset()
        assert env.scene.terrain.terrain_levels.tolist() == [6]
        _, reward, _, _, extras = env.step(torch.zeros((1, 16)))
        assert torch.isfinite(reward).all()
        assert extras["diagnostics"]["difficulty"].item() == pytest.approx(0.6)
    finally:
        env.close()


def test_invalid_task_settings_and_reward_names_fail_early() -> None:
    for options in (
        {"num_envs": 0},
        {"initial_difficulty": 2},
        {"episode_length_s": float("nan")},
        {"profile": "WS"},
    ):
        with pytest.raises(ValueError):
            VelocityTaskConfig(**options)
    with pytest.raises(ValueError, match="Unknown reward"):
        make_env_cfg(VelocityTaskConfig(reward_weights={"typo": 1}))


@pytest.mark.integration
def test_reward_reference_changes_tracking_without_changing_physics(env: ManagerBasedRlEnv) -> None:
    from walter_mjlab.tasks.velocity.mdp.terms import tracking_reward

    command = env.command_manager.get_term("velocity")
    command.set_manual_command(0.0, 0.0)
    matched = tracking_reward(env)
    before = env.scene["robot"].data.joint_pos.clone()
    command.set_manual_command(0.35, 0.0)
    mismatch = tracking_reward(env)
    torch.testing.assert_close(mismatch / matched, torch.full((2,), 0.60653066))
    torch.testing.assert_close(env.scene["robot"].data.joint_pos, before)


@pytest.mark.integration
def test_scaled_and_disabled_rewards_are_reported_exactly() -> None:
    cfg = make_env_cfg(VelocityTaskConfig(noise=False, reward_weights={"torque": 0.0}))
    cfg.scale_rewards_by_dt = True
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
        env.reset()
        _, reward, _, _, extras = env.step(torch.zeros((1, 16)))
        terms = extras["diagnostics"]["reward_terms"]
        assert terms["torque"].item() == 0.0
        torch.testing.assert_close(sum(terms.values()), reward)
    finally:
        env.close()
