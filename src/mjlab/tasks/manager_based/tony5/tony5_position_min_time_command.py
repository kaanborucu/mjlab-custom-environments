"""Episode-level target-pose commands for the TONY5 minimum-time task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class Tony5PositionMinTimeCommandCfg(CommandTermCfg):
  """Configuration for one fixed target pose per episode.

  The command is stored as a local target ``[x, y, z, yaw]``.  ``x`` and ``y``
  are sampled in polar coordinates so every target is between the configured
  horizontal distance limits.
  """

  asset_name: str = "robot"
  horizontal_distance_range: tuple[float, float] = (5.0, 30.0)
  vertical_displacement_range: tuple[float, float] = (-3.0, 3.0)
  target_yaw_range: tuple[float, float] = (-torch.pi, torch.pi)
  reference_height: float = 1.5
  minimum_target_height: float = 0.5

  def build(self, env: ManagerBasedRlEnv) -> Tony5PositionMinTimeCommand:
    return Tony5PositionMinTimeCommand(self, env)


class Tony5PositionMinTimeCommand(CommandTerm):
  """Sample a target pose once on reset and keep it until the next reset."""

  cfg: Tony5PositionMinTimeCommandCfg

  def __init__(self, cfg: Tony5PositionMinTimeCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._asset: Entity = env.scene[cfg.asset_name]
    self._command = torch.zeros((self.num_envs, 4), device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _update_metrics(self) -> None:
    pass

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids

  def compute(
    self,
    dt: float | torch.Tensor,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    """Keep the reset-sampled target unchanged for the whole episode."""
    del dt
    self._update_metrics()
    self._update_command(env_ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    distance_min, distance_max = self.cfg.horizontal_distance_range
    radius = torch.empty(len(env_ids), device=self.device).uniform_(
      distance_min, distance_max
    )
    angle = torch.empty(len(env_ids), device=self.device).uniform_(-torch.pi, torch.pi)
    self._command[env_ids, 0] = radius * torch.cos(angle)
    self._command[env_ids, 1] = radius * torch.sin(angle)

    displacement_min, displacement_max = self.cfg.vertical_displacement_range
    target_z = torch.empty(len(env_ids), device=self.device).uniform_(
      self.cfg.reference_height + displacement_min,
      self.cfg.reference_height + displacement_max,
    )
    origins_z = self._env.scene.env_origins[env_ids, 2]
    target_z = torch.maximum(
      target_z + origins_z,
      torch.full_like(target_z, self.cfg.minimum_target_height),
    )
    self._command[env_ids, 2] = target_z - origins_z
    self._command[env_ids, 3] = torch.empty(len(env_ids), device=self.device).uniform_(
      *self.cfg.target_yaw_range
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    target_positions = (
      (self._command[:, :3] + self._env.scene.env_origins).detach().cpu().numpy()
    )
    target_yaws = self._command[:, 3].detach().cpu().numpy()
    for batch in env_indices:
      target_position = target_positions[batch]
      target_yaw = target_yaws[batch]
      visualizer.add_sphere(
        center=target_position,
        radius=0.08,
        color=(1.0, 0.35, 0.0, 0.9),
        label=f"tony5_position_min_time_target_{batch}",
      )
      arrow_start = target_position.copy()
      arrow_start[2] += 0.08
      arrow_end = arrow_start + 0.5 * np.array(
        [np.cos(target_yaw), np.sin(target_yaw), 0.0]
      )
      visualizer.add_arrow(
        start=arrow_start,
        end=arrow_end,
        color=(1.0, 0.0, 0.0, 1.0),
        width=0.025,
        label=f"tony5_position_min_time_yaw_{batch}",
      )


__all__ = [
  "Tony5PositionMinTimeCommand",
  "Tony5PositionMinTimeCommandCfg",
]
