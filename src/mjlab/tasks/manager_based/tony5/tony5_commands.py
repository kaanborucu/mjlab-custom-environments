"""Position+yaw command generation for TONY5 V0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class Tony5PositionYawCommandCfg(CommandTermCfg):
  """Configuration for a four-component position and yaw command."""

  asset_name: str = "robot"
  x_range: tuple[float, float] = (-1.0, 1.0)
  y_range: tuple[float, float] = (-1.0, 1.0)
  z_range: tuple[float, float] = (0.8, 2.0)
  yaw_range: tuple[float, float] = (-torch.pi, torch.pi)
  fixed_command: tuple[float, float, float, float] = (0.0, 0.0, 1.5, 0.0)
  fixed: bool = False
  reset_xy_range: tuple[float, float] = (-0.20, 0.20)
  reset_z_range: tuple[float, float] = (-0.15, 0.15)
  reset_roll_range: tuple[float, float] = (-0.1745329252, 0.1745329252)
  reset_pitch_range: tuple[float, float] = (-0.1745329252, 0.1745329252)
  reset_yaw_offset_range: tuple[float, float] = (-torch.pi, torch.pi)
  reset_lin_vel_range: tuple[float, float] = (-0.2, 0.2)
  reset_ang_vel_range: tuple[float, float] = (-0.5, 0.5)

  def build(self, env: ManagerBasedRlEnv) -> Tony5PositionYawCommand:
    return Tony5PositionYawCommand(self, env)


class Tony5PositionYawCommand(CommandTerm):
  """Generate targets and initialize each reset near its sampled target."""

  cfg: Tony5PositionYawCommandCfg

  def __init__(self, cfg: Tony5PositionYawCommandCfg, env: ManagerBasedRlEnv):
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

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    """Draw the target position and target-yaw direction in the viewer."""
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
        radius=0.04,
        color=(1.0, 0.35, 0.0, 0.9),
        label=f"tony5_target_position_{batch}",
      )

      arrow_start = target_position.copy()
      arrow_start[2] += 0.04
      arrow_end = arrow_start + 0.35 * np.array(
        [np.cos(target_yaw), np.sin(target_yaw), 0.0]
      )
      visualizer.add_arrow(
        start=arrow_start,
        end=arrow_end,
        color=(1.0, 0.0, 0.0, 1.0),
        width=0.02,
        label=f"tony5_target_yaw_{batch}",
      )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    if env_ids is None or isinstance(env_ids, slice):
      raise TypeError("TONY5 command reset requires concrete environment IDs")
    extras = super().reset(env_ids)
    self._reset_vehicle_near_command(env_ids)
    return extras

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.cfg.fixed:
      fixed = torch.tensor(
        self.cfg.fixed_command, device=self.device, dtype=torch.float
      )
      self._command[env_ids] = fixed
    else:
      self._command[env_ids, 0] = self._sample(self.cfg.x_range, len(env_ids))
      self._command[env_ids, 1] = self._sample(self.cfg.y_range, len(env_ids))
      self._command[env_ids, 2] = self._sample(self.cfg.z_range, len(env_ids))
      self._command[env_ids, 3] = self._sample(self.cfg.yaw_range, len(env_ids))

  def _reset_vehicle_near_command(self, env_ids: torch.Tensor) -> None:
    command = self._command[env_ids]
    origins = self._env.scene.env_origins[env_ids]
    position = command[:, :3] + origins
    position[:, :2] += self._sample(self.cfg.reset_xy_range, (len(env_ids), 2))
    position[:, 2] += self._sample(self.cfg.reset_z_range, len(env_ids))

    roll = self._sample(self.cfg.reset_roll_range, len(env_ids))
    pitch = self._sample(self.cfg.reset_pitch_range, len(env_ids))
    yaw = command[:, 3] + self._sample(self.cfg.reset_yaw_offset_range, len(env_ids))
    orientation = quat_from_euler_xyz(roll, pitch, yaw)

    linear_velocity = self._sample(self.cfg.reset_lin_vel_range, (len(env_ids), 3))
    angular_velocity = self._sample(self.cfg.reset_ang_vel_range, (len(env_ids), 3))

    self._asset.write_root_link_pose_to_sim(
      torch.cat((position, orientation), dim=-1), env_ids=env_ids
    )
    self._asset.write_root_link_velocity_to_sim(
      torch.cat((linear_velocity, angular_velocity), dim=-1),
      env_ids=env_ids,
    )

  def _sample(
    self,
    value_range: tuple[float, float],
    shape: int | tuple[int, ...],
  ) -> torch.Tensor:
    if isinstance(shape, int):
      shape = (shape,)
    low, high = value_range
    return torch.rand(shape, device=self.device) * (high - low) + low


__all__ = ["Tony5PositionYawCommand", "Tony5PositionYawCommandCfg"]
