"""Smooth fixed-range omnidirectional velocity commands for TONY5 Aero."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.manager_based.tony5.tony5_velocity_curriculum import (
  Tony5RadialVelocityCommand,
  Tony5RadialVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class Tony5OmniVelocityCommandCfg(Tony5RadialVelocityCommandCfg):
  """Configuration for smooth fixed-range omnidirectional commands."""

  command_transition_time_s: float = 1.0
  enable_command_smoothing: bool = True

  def build(self, env: ManagerBasedRlEnv) -> Tony5OmniVelocityCommand:
    return Tony5OmniVelocityCommand(self, env)


class Tony5OmniVelocityCommand(Tony5RadialVelocityCommand):
  """Ramp sampled commands instead of applying new targets instantaneously."""

  def __init__(self, cfg: Tony5OmniVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._omni_cfg = cfg
    if self._omni_cfg.enable_command_smoothing and cfg.command_transition_time_s <= 0.0:
      raise ValueError("command_transition_time_s must be positive.")
    self._target_command = torch.zeros_like(self.vel_command_b)
    self._transition_start = torch.zeros_like(self.vel_command_b)
    self._transition_elapsed = torch.full(
      (self.num_envs,),
      cfg.command_transition_time_s,
      device=self.device,
    )
    self._smoothing_dt: float | torch.Tensor = self._env.step_dt

  @property
  def target_command(self) -> torch.Tensor:
    """Return the latest sampled command before transition smoothing."""
    return self._target_command

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    self.vel_command_b[env_ids] = 0.0
    self.vel_command_w[env_ids] = 0.0
    self._target_command[env_ids] = 0.0
    self._transition_start[env_ids] = 0.0
    self._transition_elapsed[env_ids] = self._omni_cfg.command_transition_time_s
    return super().reset(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    transition_start = self.vel_command_b[env_ids].clone()
    super()._resample_command(env_ids)
    target_command = self.vel_command_b[env_ids].clone()
    self._target_command[env_ids] = target_command
    self._transition_start[env_ids] = transition_start
    self._transition_elapsed[env_ids] = 0.0
    if self._omni_cfg.enable_command_smoothing:
      self.vel_command_b[env_ids] = transition_start
    else:
      self._transition_start[env_ids] = target_command
      self._transition_elapsed[env_ids] = self._omni_cfg.command_transition_time_s
      self.vel_command_b[env_ids] = target_command
    self._sync_world_command(env_ids)

  def compute(
    self,
    dt: float | torch.Tensor,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    self._smoothing_dt = dt
    super().compute(dt, env_ids)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    super()._update_command(env_ids)
    if env_ids is None:
      active_ids = torch.arange(self.num_envs, device=self.device)
    else:
      active_ids = env_ids
    if len(active_ids) == 0:
      return
    if not self._omni_cfg.enable_command_smoothing:
      self.vel_command_b[active_ids] = self._target_command[active_ids]
      self._sync_world_command(active_ids)
      return

    if isinstance(self._smoothing_dt, torch.Tensor):
      step_dt = self._smoothing_dt.expand(self.num_envs)[active_ids]
    else:
      step_dt = torch.full(
        (len(active_ids),),
        self._smoothing_dt,
        device=self.device,
      )
    elapsed = self._transition_elapsed[active_ids] + step_dt
    self._transition_elapsed[active_ids] = elapsed
    linear_alpha = torch.clamp(
      elapsed / self._omni_cfg.command_transition_time_s,
      min=0.0,
      max=1.0,
    )
    alpha = linear_alpha * linear_alpha * (3.0 - 2.0 * linear_alpha)
    self.vel_command_b[active_ids] = torch.lerp(
      self._transition_start[active_ids],
      self._target_command[active_ids],
      alpha.unsqueeze(-1),
    )
    self._sync_world_command(active_ids)

  def _sync_world_command(self, env_ids: torch.Tensor) -> None:
    """Keep the explicit world-frame command synchronized during smoothing."""
    if self.cfg.linear_velocity_frame != "world" or len(env_ids) == 0:
      return
    self.vel_command_w[env_ids, :2] = self.vel_command_b[env_ids, :2]
    self.vel_command_w[env_ids, 2] = self.vel_command_b[env_ids, 3]

  def _apply_keyboard_command(self, env_ids: torch.Tensor | None = None) -> None:
    """Route manual commands through the configured velocity frame.

    V3 configures horizontal linear velocity in the world frame. The
    keyboard/gamepad values are therefore already ``[vx_w, vy_w]`` and must
    not be rotated by the vehicle heading. Body-frame TONY5 tasks continue
    to use the inherited body-frame behavior.
    """
    if not self._keyboard_enabled:
      return
    if env_ids is None:
      active_ids = torch.arange(self.num_envs, device=self.device)
    else:
      active_ids = env_ids
    desired = self._keyboard_command.unsqueeze(0).expand(len(active_ids), -1)
    target = desired
    if not torch.equal(self._target_command[active_ids], target):
      self._transition_start[active_ids] = self.vel_command_b[active_ids]
      self._target_command[active_ids] = target
      self._transition_elapsed[active_ids] = 0.0
    if not self._omni_cfg.enable_command_smoothing:
      self.vel_command_b[active_ids] = target
    self.is_standing_env[active_ids] = False
    self.is_heading_env[active_ids] = False
    self.is_world_env[active_ids] = False
    self._sync_world_command(active_ids)


__all__ = [
  "Tony5OmniVelocityCommand",
  "Tony5OmniVelocityCommandCfg",
]
