"""Observation, reward, and termination equations shared by every action source."""

from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import builtin_sensor

from walter_mjlab.tasks.velocity.mdp.actions import JointTargetsAction
from walter_mjlab.tasks.velocity.mdp.commands import VelocityCommand


def action_term(env: ManagerBasedRlEnv) -> JointTargetsAction:
    return cast(JointTargetsAction, env.action_manager.get_term("joint_targets"))


def actor_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return `(N, 68)`; simulated joint effort is provisional, not hardware validated."""
    robot = env.scene["robot"]
    ids = action_term(env).joint_ids
    positions = robot.data.joint_pos[:, ids]
    return torch.cat(
        [
            cast(VelocityCommand, env.command_manager.get_term("velocity")).command,
            env.action_manager.action.clamp(-1, 1),
            builtin_sensor(env, "robot/imu_acc"),
            builtin_sensor(env, "robot/imu_gyro"),
            positions[:, :4],
            positions[:, 4:8].sin(),
            positions[:, 4:8].cos(),
            robot.data.joint_vel[:, ids],
            robot.data.qfrc_actuator[:, ids],
        ],
        dim=-1,
    )


def critic_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Clean actor state plus privileged gravity and root velocities (77 features)."""
    data = env.scene["robot"].data
    return torch.cat(
        [
            actor_observation(env),
            data.projected_gravity_b,
            data.root_link_lin_vel_b,
            data.root_link_ang_vel_b,
        ],
        dim=-1,
    )


def tracking_quality(env: ManagerBasedRlEnv, sigma: float = 0.35) -> torch.Tensor:
    data = env.scene["robot"].data
    command = cast(VelocityCommand, env.command_manager.get_term("velocity")).command
    actual = torch.stack((data.root_link_lin_vel_b[:, 0], data.root_link_ang_vel_b[:, 2]), dim=-1)
    return torch.exp(-0.5 * ((actual - command) / sigma).square()).prod(dim=-1)


def tracking_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Preserve legacy multiplicative tracking, with 100 x 100 outer weight."""
    up_z = -env.scene["robot"].data.projected_gravity_b[:, 2]
    orientation = torch.exp(-0.5 * ((1 - up_z) / 0.05).square())
    return tracking_quality(env) * orientation


def torque_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.qfrc_actuator[:, action_term(env).joint_ids].square().sum(-1)


def action_rate_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    action = env.action_manager.action.clamp(-1, 1)
    previous = env.action_manager.prev_action.clamp(-1, 1)
    stationary = (
        cast(VelocityCommand, env.command_manager.get_term("velocity")).command.abs() < 0.05
    ).all(-1)
    return (action - previous).square().sum(-1) * torch.where(stationary, 5.0, 1.0)


def body_velocity_cost(env: ManagerBasedRlEnv, component: str) -> torch.Tensor:
    data = env.scene["robot"].data
    if component == "vertical":
        return data.root_link_lin_vel_b[:, 2].square()
    return data.root_link_ang_vel_b[:, {"roll": 0, "pitch": 1}[component]].square()


def stationary_joint_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    stationary = (
        cast(VelocityCommand, env.command_manager.get_term("velocity")).command.abs() < 0.05
    ).all(-1)
    return (
        env.scene["robot"].data.joint_vel[:, action_term(env).joint_ids].square().sum(-1)
        * stationary
    )


def fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
    return -env.scene["robot"].data.projected_gravity_b[:, 2] < 0.1


def flip_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    return fallen(env).float()


def wheel_collision_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Legacy front-versus-rear wheel center overlap heuristic, not contact forces."""
    action = action_term(env)
    centers = env.scene["robot"].data.geom_pos_w[:, action.wheel_geom_ids]
    left = centers[:, :2, None, :] - centers[:, None, 4:6, :]
    right = centers[:, 2:4, None, :] - centers[:, None, 6:8, :]
    overlaps = (left.norm(dim=-1) < action.wheel_diameter).sum((1, 2))
    overlaps += (right.norm(dim=-1) < action.wheel_diameter).sum((1, 2))
    return overlaps.float()
