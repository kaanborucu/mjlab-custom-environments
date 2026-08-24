"""World-frame observation terms for the TONY5 Aero Omni V3 task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def world_velocity_command(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Return the V3 command as ``[vx_w, vy_w, yaw_rate, vz_w]``."""
  command = env.command_manager.get_command(command_name)
  if command is None:
    raise ValueError(f"Command '{command_name}' not found.")
  command_term = env.command_manager.get_term(command_name)
  world_linear = getattr(command_term, "vel_command_w", None)
  if not isinstance(world_linear, torch.Tensor) or world_linear.shape[-1] != 3:
    raise TypeError("V3 world commands require a 3-vector vel_command_w buffer.")
  return torch.cat(
    (
      world_linear[:, :2],
      command[:, 2:3],
      world_linear[:, 2:3],
    ),
    dim=-1,
  )


def world_linear_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return root linear velocity in world coordinates."""
  return env.scene[asset_cfg.name].data.root_link_lin_vel_w


def heading_sin_cos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return yaw heading as ``[sin(yaw), cos(yaw)]``."""
  heading = env.scene[asset_cfg.name].data.heading_w
  return torch.stack((torch.sin(heading), torch.cos(heading)), dim=-1)


__all__ = [
  "heading_sin_cos",
  "world_linear_velocity",
  "world_velocity_command",
]
