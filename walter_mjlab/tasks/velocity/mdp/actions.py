"""Canonical normalized actions with policy-rate knee target sampling."""

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from walter_mjlab.robots import JOINT_NAMES


@dataclass(kw_only=True)
class JointTargetsActionCfg(ActionTermCfg):
    """Control hips absolutely, knees relatively, and wheels by velocity."""

    hip_scale: float = 1.25
    knee_scale: float = 5.5
    wheel_scale: float = 60.0

    def build(self, env: ManagerBasedRlEnv) -> "JointTargetsAction":
        return JointTargetsAction(self, env)


class JointTargetsAction(ActionTerm):
    """Apply `(N, 16)` actions in named hip/knee/wheel order, all in SI units."""

    def __init__(self, cfg: JointTargetsActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        ids, names = self._entity.find_joints(JOINT_NAMES, preserve_order=True)
        if tuple(names) != JOINT_NAMES:
            raise ValueError(f"Joint contract mismatch: {names}")
        self.joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
        wheel_names = tuple(
            f"{leg}_{end}_wheel_geom"
            for leg in ("head_left", "head_right", "torso_left", "torso_right")
            for end in ("front", "rear")
        )
        geom_ids, _ = self._entity.find_geoms(wheel_names, preserve_order=True)
        self.wheel_geom_ids = torch.tensor(geom_ids, device=self.device, dtype=torch.long)
        self.wheel_diameter = 2 * float(self._entity.spec.geom(wheel_names[0]).size[0])
        self._raw_actions = torch.zeros((self.num_envs, 16), device=self.device)
        self.position_targets = torch.zeros((self.num_envs, 8), device=self.device)
        self.velocity_targets = torch.zeros((self.num_envs, 8), device=self.device)

    @property
    def action_dim(self) -> int:
        return 16

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        if actions.shape != self._raw_actions.shape:
            raise ValueError(
                f"Expected actions {tuple(self._raw_actions.shape)}, got {actions.shape}"
            )
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        self.position_targets[:, :4] = self.cfg.hip_scale * self._raw_actions[:, :4]
        # Freeze the measured-relative target once; never integrate it each substep.
        self.position_targets[:, 4:] = (
            self._entity.data.joint_pos[:, self.joint_ids[4:8]]
            + self.cfg.knee_scale * self._raw_actions[:, 4:8]
        )
        self.velocity_targets[:] = self.cfg.wheel_scale * self._raw_actions[:, 8:]

    def apply_actions(self) -> None:
        self._entity.set_joint_position_target(self.position_targets, joint_ids=self.joint_ids[:8])
        self._entity.set_joint_velocity_target(self.velocity_targets, joint_ids=self.joint_ids[8:])

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0
        self.position_targets[ids] = 0
        self.velocity_targets[ids] = 0
