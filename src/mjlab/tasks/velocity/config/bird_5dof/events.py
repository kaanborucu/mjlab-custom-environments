"""Reset events shared by the bird velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_heading_forward_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  speed_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Set positive horizontal speed along each bird's sampled heading."""
  env_ids = resolve_env_ids(env, env_ids)
  asset: Entity = env.scene[asset_cfg.name]
  root_velocity = asset.data.root_link_vel_w[env_ids].clone()
  heading = asset.data.heading_w[env_ids]
  speed = torch.empty(len(env_ids), device=env.device).uniform_(*speed_range)
  root_velocity[:, 0] = speed * torch.cos(heading)
  root_velocity[:, 1] = speed * torch.sin(heading)
  asset.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
