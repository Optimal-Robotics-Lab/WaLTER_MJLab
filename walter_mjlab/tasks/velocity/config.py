"""One task configuration for PPO and heuristic environment inspection."""

import math
from dataclasses import dataclass, field
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import reset_scene_to_default, time_out
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.terrains.heightfield_terrains import HfPerlinNoiseTerrainCfg
from mjlab.utils.noise import GaussianNoiseCfg

from walter_mjlab.robots import get_robot_cfg
from walter_mjlab.tasks.velocity.mdp import terms
from walter_mjlab.tasks.velocity.mdp.actions import JointTargetsActionCfg
from walter_mjlab.tasks.velocity.mdp.commands import VelocityCommandCfg
from walter_mjlab.tasks.velocity.mdp.diagnostics import (
    capture_step,
    set_terrain_level,
    update_difficulty,
)


@dataclass
class VelocityTaskConfig:
    """Task settings; physical profiles come from the modeling submodule.

    Initial difficulty controls command amplitude and initial rough-terrain row.
    Curriculum changes difficulty at resets based on completed-episode tracking.
    Noise uses feature-specific Gaussian scales, leaving commands and actions clean;
    sensor-specific calibration and domain randomization remain future work.
    """

    num_envs: int = 1
    profile: Literal["FS", "WS"] = "FS"
    terrain: Literal["flat", "rough"] = "flat"
    seed: int = 42
    episode_length_s: float = 20.0
    curriculum: bool = True
    initial_difficulty: float = 0.0
    noise: bool = True
    max_forward_velocity: float = 3.0
    max_yaw_velocity: float = 1.0
    reward_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.num_envs, int) or self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer.")
        if self.profile not in ("FS", "WS") or self.terrain not in ("flat", "rough"):
            raise ValueError("Choose profile FS/WS and terrain flat/rough.")
        if self.profile == "WS":
            raise ValueError(
                "WS morphology is available, but requires separately tuned gains before task use."
            )
        if not math.isfinite(self.initial_difficulty) or not 0 <= self.initial_difficulty <= 1:
            raise ValueError("initial_difficulty must lie in [0, 1].")
        for name in ("episode_length_s", "max_forward_velocity", "max_yaw_velocity"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not isinstance(self.reward_weights, dict) or not all(
            isinstance(name, str) and isinstance(value, (int, float)) and math.isfinite(value)
            for name, value in self.reward_weights.items()
        ):
            raise ValueError("reward_weights must map term names to finite numbers.")


def make_env_cfg(config: VelocityTaskConfig | None = None) -> ManagerBasedRlEnvCfg:
    """Compose the same managers regardless of whether actions come from RL or a joystick."""
    task = config or VelocityTaskConfig()
    terrain = TerrainEntityCfg(terrain_type="plane", env_spacing=4.0)
    if task.terrain == "rough":
        terrain = TerrainEntityCfg(
            terrain_type="generator",
            max_init_terrain_level=round(task.initial_difficulty * 10),
            terrain_generator=TerrainGeneratorCfg(
                seed=task.seed,
                curriculum=True,
                size=(8.0, 8.0),
                num_rows=11,
                num_cols=1,
                sub_terrains={
                    "perlin": HfPerlinNoiseTerrainCfg(height_range=(0.0, 0.3), scale=3.0)
                },
            ),
        )
    rewards = {
        "tracking": RewardTermCfg(func=terms.tracking_reward, weight=10000.0),
        "torque": RewardTermCfg(func=terms.torque_cost, weight=-0.0005),
        "action_rate": RewardTermCfg(func=terms.action_rate_cost, weight=-1000.0),
        "vertical_velocity": RewardTermCfg(
            func=terms.body_velocity_cost, weight=-5.0, params={"component": "vertical"}
        ),
        "roll_velocity": RewardTermCfg(
            func=terms.body_velocity_cost, weight=-0.5, params={"component": "roll"}
        ),
        "pitch_velocity": RewardTermCfg(
            func=terms.body_velocity_cost, weight=-0.5, params={"component": "pitch"}
        ),
        "stationary_joints": RewardTermCfg(func=terms.stationary_joint_cost, weight=-10.0),
        "flipped": RewardTermCfg(func=terms.flip_cost, weight=-100000.0),
        "wheel_collision": RewardTermCfg(func=terms.wheel_collision_cost, weight=-10000.0),
    }
    for name, weight in task.reward_weights.items():
        if name not in rewards:
            raise ValueError(f"Unknown reward {name!r}; choose from {list(rewards)}.")
        rewards[name].weight = weight
    return ManagerBasedRlEnvCfg(
        seed=task.seed,
        decimation=5,
        scene=SceneCfg(
            num_envs=task.num_envs, entities={"robot": get_robot_cfg(task.profile)}, terrain=terrain
        ),
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.004)),
        episode_length_s=task.episode_length_s,
        scale_rewards_by_dt=False,
        actions={"joint_targets": JointTargetsActionCfg(entity_name="robot")},
        observations={
            "actor": ObservationGroupCfg(
                terms={
                    "state": ObservationTermCfg(
                        func=terms.actor_observation,
                        noise=GaussianNoiseCfg(
                            mean=0.0, std=(0.0,) * 18 + (0.01,) * 6 + (0.001,) * 12 + (0.1,) * 32
                        ),
                    )
                },
                concatenate_terms=True,
                enable_corruption=task.noise,
            ),
            "critic": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=terms.critic_observation)},
                concatenate_terms=True,
                enable_corruption=False,
            ),
        },
        commands={
            "velocity": VelocityCommandCfg(
                resampling_time_range=(1.0, 10.0),
                initial_difficulty=task.initial_difficulty,
                max_forward_velocity=task.max_forward_velocity,
                max_yaw_velocity=task.max_yaw_velocity,
            )
        },
        rewards=rewards,
        terminations={
            "fallen": TerminationTermCfg(func=terms.fallen),
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
        },
        events={
            "terrain_level": EventTermCfg(func=set_terrain_level, mode="reset"),
            "reset_scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
            "diagnostics": EventTermCfg(func=capture_step, mode="step"),
        },
        curriculum={"difficulty": CurriculumTermCfg(func=update_difficulty)}
        if task.curriculum
        else {},
    )
