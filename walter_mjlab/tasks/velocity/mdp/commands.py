"""Sampled training commands with an explicit persistent manual override."""

import math
from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
    max_forward_velocity: float = 3.0
    max_yaw_velocity: float = 1.0
    initial_difficulty: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> "VelocityCommand":
        return VelocityCommand(self, env)


class VelocityCommand(CommandTerm):
    """Forward m/s and yaw rad/s; manual targets do not bypass task machinery."""

    def __init__(self, cfg: VelocityCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        self._command = torch.zeros((self.num_envs, 2), device=self.device)
        self._manual: torch.Tensor | None = None
        self.difficulty = torch.full((self.num_envs,), cfg.initial_difficulty, device=self.device)
        self.tracking_sum = torch.zeros(self.num_envs, device=self.device)
        self.tracking_count = torch.zeros_like(self.tracking_sum)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def set_manual_command(self, forward_m_s: float, yaw_rad_s: float) -> None:
        """Set a physical reward reference, retained across automatic resets."""
        if not math.isfinite(forward_m_s) or not math.isfinite(yaw_rad_s):
            raise ValueError("Manual command must be finite.")
        if (
            abs(forward_m_s) > self.cfg.max_forward_velocity
            or abs(yaw_rad_s) > self.cfg.max_yaw_velocity
        ):
            raise ValueError("Manual command exceeds configured task command limits.")
        self._manual = torch.tensor([forward_m_s, yaw_rad_s], device=self.device)
        self._command[:] = self._manual

    def clear_manual_command(self) -> None:
        self._manual = None
        self._resample(torch.arange(self.num_envs, device=self.device))

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        scale = 0.25 + 0.75 * self.difficulty[env_ids]
        limits = torch.tensor(
            [self.cfg.max_forward_velocity, self.cfg.max_yaw_velocity], device=self.device
        )
        self._command[env_ids] = (
            (2 * torch.rand((count, 2), device=self.device) - 1) * limits * scale[:, None]
        )
        zero_components = torch.rand((count, 2), device=self.device) < 0.1
        self._command[env_ids] *= ~zero_components
        zero_all = torch.rand(count, device=self.device) < 0.2
        self._command[env_ids[zero_all]] = 0
        if self._manual is not None:
            self._command[env_ids] = self._manual

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        if self._manual is not None:
            self._command[slice(None) if env_ids is None else env_ids] = self._manual

    def _update_metrics(self) -> None:
        # Accumulation occurs before reset in the diagnostic event, not after reset here.
        pass

    def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
        result = super().reset(env_ids)
        self.tracking_sum[env_ids] = 0
        self.tracking_count[env_ids] = 0
        return result
