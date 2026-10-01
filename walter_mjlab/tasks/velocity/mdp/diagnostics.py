"""Capture terminal-step rewards before MJLab resets managers and commands."""

from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv

from walter_mjlab.tasks.velocity.mdp.commands import VelocityCommand
from walter_mjlab.tasks.velocity.mdp.terms import action_term, tracking_quality


def capture_step(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None = None) -> None:
    """Snapshot weighted contributions and accumulate curriculum evidence on-device."""
    del env_ids
    command = cast(VelocityCommand, env.command_manager.get_term("velocity"))
    command.tracking_sum += tracking_quality(env)
    command.tracking_count += 1
    # MJLab 1.6.0 exposes scalar viewer diagnostics publicly, but batched data only
    # through this buffer. Keep this pinned integration dependency in one place.
    contributions = env.reward_manager._step_reward.clone()
    if env.cfg.scale_rewards_by_dt:
        contributions *= env.step_dt
    robot = env.scene["robot"]
    joint_ids = action_term(env).joint_ids
    env.extras["diagnostics"] = {
        "reward_terms": {
            name: contributions[:, index]
            for index, name in enumerate(env.reward_manager.active_terms)
        },
        "total_reward": env.reward_buf.clone(),
        "command": command.command.clone(),
        "measured_velocity": torch.stack(
            (robot.data.root_link_lin_vel_b[:, 0], robot.data.root_link_ang_vel_b[:, 2]), dim=-1
        ).clone(),
        "upright": -robot.data.projected_gravity_b[:, 2].clone(),
        "joint_position": robot.data.joint_pos[:, joint_ids].clone(),
        "joint_velocity": robot.data.joint_vel[:, joint_ids].clone(),
        "difficulty": command.difficulty.clone(),
        "episode_step": env.episode_length_buf.clone(),
        "terminated": env.reset_terminated.clone(),
        "truncated": env.reset_time_outs.clone(),
        "action": env.action_manager.action.clamp(-1, 1).clone(),
    }


def update_difficulty(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """Advance after a full tracked episode; reduce after a fall or poor tracking."""
    command = cast(VelocityCommand, env.command_manager.get_term("velocity"))
    count = command.tracking_count[env_ids]
    score = command.tracking_sum[env_ids] / count.clamp_min(1)
    complete = env.episode_length_buf[env_ids] >= env.max_episode_length
    failed = env.termination_manager.terminated[env_ids]
    delta = torch.where(complete & ~failed & (score > 0.7), 0.1, 0.0)
    delta = torch.where((count > 0) & (failed | (score < 0.3)), -0.1, delta)
    command.difficulty[env_ids] = (command.difficulty[env_ids] + delta).clamp(0, 1)
    return {"difficulty": command.difficulty.mean()}


def set_terrain_level(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
    """Place resets at the configured stage even when progression is disabled."""
    command = cast(VelocityCommand, env.command_manager.get_term("velocity"))
    terrain = env.scene.terrain
    if terrain is not None and terrain.terrain_origins is not None:
        levels = (
            (command.difficulty[env_ids] * (terrain.terrain_origins.shape[0] - 1)).round().long()
        )
        terrain.terrain_levels[env_ids] = levels
        terrain.env_origins[env_ids] = terrain.terrain_origins[
            levels, terrain.terrain_types[env_ids]
        ]
