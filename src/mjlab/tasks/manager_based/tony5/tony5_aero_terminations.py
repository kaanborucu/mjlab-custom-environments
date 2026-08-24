"""V1-only TONY5 downward-velocity termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manager_based.tony5.tony5_aero_safety import (
  numerical_safety_failure,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

TONY5_AERO_DOWNWARD_VELOCITY_LIMIT = -5.0
TONY5_AERO_DOWNWARD_REWARD_LIMIT = -3.0


def rapid_descent_termination(
  env: ManagerBasedRlEnv,
  minimum_vertical_velocity: float = TONY5_AERO_DOWNWARD_VELOCITY_LIMIT,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate V1 when measured world-frame downward speed is excessive."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_w[:, 2] < minimum_vertical_velocity


__all__ = [
  "TONY5_AERO_DOWNWARD_VELOCITY_LIMIT",
  "TONY5_AERO_DOWNWARD_REWARD_LIMIT",
  "numerical_safety_failure",
  "rapid_descent_termination",
]
