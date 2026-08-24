"""V1-only speed-adaptive reward terms for high-speed TONY5 flight."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manager_based.tony5 import tony5_mdp
from mjlab.tasks.manager_based.tony5.tony5_aero_terminations import (
  TONY5_AERO_DOWNWARD_REWARD_LIMIT,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _velocity_command(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return command


def commanded_horizontal_speed(
  env: ManagerBasedRlEnv,
  command_name: str = "velocity",
) -> torch.Tensor:
  """Return the commanded horizontal speed for each environment."""
  command = _velocity_command(env, command_name)
  return torch.linalg.vector_norm(command[:, :2], dim=-1)


def upright_reward_scale(horizontal_speed: torch.Tensor) -> torch.Tensor:
  """Fade uprightness weight to zero for commands at or above 5 m/s."""
  return torch.clamp(1.0 - horizontal_speed / 5.0, min=0.0, max=1.0)


def roll_pitch_rate_weight(horizontal_speed: torch.Tensor) -> torch.Tensor:
  """Retain a 0.2 minimum penalty on roll and pitch rates at high speed."""
  return torch.clamp(1.0 - horizontal_speed / 10.0, min=0.2, max=1.0)


def speed_scaled_uprightness_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "velocity",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Apply the existing uprightness reward with a speed-dependent scale."""
  scale = upright_reward_scale(commanded_horizontal_speed(env, command_name))
  return scale * tony5_mdp.uprightness_reward(env, asset_cfg)


def high_speed_roll_uprightness_reward(
  env: ManagerBasedRlEnv,
  command_name: str = "velocity",
  minimum_horizontal_speed: float = 8.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward roll uprightness only for commands above the speed threshold."""
  speed = commanded_horizontal_speed(env, command_name)
  asset = env.scene[asset_cfg.name]
  projected_gravity_b = asset.data.projected_gravity_b
  roll = torch.atan2(-projected_gravity_b[:, 1], -projected_gravity_b[:, 2])
  roll_uprightness = 0.5 * (torch.cos(roll) + 1.0)
  return torch.where(speed > minimum_horizontal_speed, roll_uprightness, 0.0)


def body_angular_acceleration_l2(
  env: ManagerBasedRlEnv,
  angular_acceleration_sensor_name: str = "robot/imu_angacc",
) -> torch.Tensor:
  """Return MuJoCo frame angular acceleration squared."""
  angular_acceleration = envs_mdp.builtin_sensor(env, angular_acceleration_sensor_name)
  return torch.sum(torch.square(angular_acceleration), dim=-1)


def downward_velocity_l2(
  env: ManagerBasedRlEnv,
  minimum_vertical_velocity: float = TONY5_AERO_DOWNWARD_REWARD_LIMIT,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return excess measured downward speed squared for soft penalization."""
  asset = env.scene[asset_cfg.name]
  vertical_velocity = asset.data.root_link_lin_vel_w[:, 2]
  excess_downward_speed = torch.clamp(
    minimum_vertical_velocity - vertical_velocity,
    min=0.0,
  )
  return torch.square(excess_downward_speed)


def track_angular_velocity_speed_scaled(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track yaw rate while relaxing roll/pitch-rate suppression with speed."""
  asset = env.scene[asset_cfg.name]
  command = _velocity_command(env, command_name)
  actual = asset.data.root_link_ang_vel_b
  yaw_error = torch.square(command[:, 2] - actual[:, 2])
  pq_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  pq_weight = roll_pitch_rate_weight(torch.linalg.vector_norm(command[:, :2], dim=-1))
  angular_error = yaw_error + pq_weight * pq_error
  return torch.exp(-angular_error / std**2)


__all__ = [
  "body_angular_acceleration_l2",
  "commanded_horizontal_speed",
  "downward_velocity_l2",
  "high_speed_roll_uprightness_reward",
  "roll_pitch_rate_weight",
  "speed_scaled_uprightness_reward",
  "track_angular_velocity_speed_scaled",
  "upright_reward_scale",
]
