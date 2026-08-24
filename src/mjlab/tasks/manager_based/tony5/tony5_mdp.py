"""TONY5 V0 observations, rewards, terminations, and reset helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manager_based.tony5.tony5_actions import Tony5RotorSpeedAction
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  OMEGA_HOVER,
  OMEGA_MAX,
  ROTOR_JOINTS,
)
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply_inverse,
  wrap_to_pi,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _asset(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


def _command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' does not exist."
  return command


def target_position_w(
  env: ManagerBasedRlEnv,
  command_name: str = "position_yaw",
) -> torch.Tensor:
  return _command(env, command_name)[:, :3] + env.scene.env_origins


def position_error_w(
  env: ManagerBasedRlEnv,
  command_name: str = "position_yaw",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return (
    target_position_w(env, command_name) - _asset(env, asset_cfg).data.root_link_pos_w
  )


def position_error_b(
  env: ManagerBasedRlEnv,
  position_scale: float = 2.0,
  command_name: str = "position_yaw",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  error_w = position_error_w(env, command_name, asset_cfg)
  return quat_apply_inverse(asset.data.root_link_quat_w, error_w) / position_scale


def body_linear_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return _asset(env, asset_cfg).data.root_link_lin_vel_b


def body_angular_velocity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return _asset(env, asset_cfg).data.root_link_ang_vel_b


def root_height(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return the vehicle root height relative to its environment origin."""
  asset = _asset(env, asset_cfg)
  return (asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def projected_gravity(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return _asset(env, asset_cfg).data.projected_gravity_b


def yaw_error(
  env: ManagerBasedRlEnv,
  command_name: str = "position_yaw",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  _, _, actual_yaw = euler_xyz_from_quat(asset.data.root_link_quat_w)
  return wrap_to_pi(_command(env, command_name)[:, 3] - actual_yaw)


def yaw_error_sin_cos(
  env: ManagerBasedRlEnv,
  command_name: str = "position_yaw",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  error = yaw_error(env, command_name, asset_cfg)
  return torch.stack((torch.sin(error), torch.cos(error)), dim=-1)


def rotor_speed_observation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  speeds = _asset(env, asset_cfg).data.joint_vel[:, asset_cfg.joint_ids]
  return speeds.abs() / OMEGA_MAX


def previous_action(
  env: ManagerBasedRlEnv,
  action_name: str = "rotor_speed",
) -> torch.Tensor:
  term = cast(Tony5RotorSpeedAction, env.action_manager.get_term(action_name))
  return term.previous_action


def position_tracking_exp(
  env: ManagerBasedRlEnv,
  sigma: float = 0.4,
  command_name: str = "position_yaw",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  error = position_error_w(env, command_name, asset_cfg)
  return torch.exp(-torch.sum(torch.square(error), dim=-1) / sigma**2)


def yaw_tracking_exp(
  env: ManagerBasedRlEnv,
  sigma: float = 0.5,
  command_name: str = "position_yaw",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return torch.exp(-torch.square(yaw_error(env, command_name, asset_cfg)) / sigma**2)


def body_up_alignment(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  return (-asset.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0)


def uprightness_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return 0.5 * (body_up_alignment(env, asset_cfg) + 1.0)


def linear_velocity_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return torch.sum(torch.square(body_linear_velocity(env, asset_cfg)), dim=-1)


def angular_velocity_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return torch.sum(torch.square(body_angular_velocity(env, asset_cfg)), dim=-1)


def action_rate_l2(
  env: ManagerBasedRlEnv,
  action_name: str = "rotor_speed",
) -> torch.Tensor:
  term = cast(Tony5RotorSpeedAction, env.action_manager.get_term(action_name))
  return torch.sum(torch.square(term.raw_action - term.previous_action), dim=-1)


def position_error_termination(
  env: ManagerBasedRlEnv,
  limit: float = 4.0,
  command_name: str = "position_yaw",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return (
    torch.linalg.vector_norm(position_error_w(env, command_name, asset_cfg), dim=-1)
    > limit
  )


def root_height_termination(
  env: ManagerBasedRlEnv,
  minimum_height: float = 0.10,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _asset(env, asset_cfg)
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return height < minimum_height


def attitude_termination(
  env: ManagerBasedRlEnv,
  minimum_up_alignment: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  return body_up_alignment(env, asset_cfg) < minimum_up_alignment


def reset_rotor_speeds(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  asset = _asset(env, asset_cfg)
  joint_ids, joint_names = asset.find_joints(ROTOR_JOINTS, preserve_order=True)
  if tuple(joint_names) != ROTOR_JOINTS:
    raise ValueError(f"Unexpected TONY5 reset joint order: {joint_names}")
  ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
  speeds = torch.rand((len(env_ids), 4), device=env.device)
  speeds = (0.95 + 0.10 * speeds) * OMEGA_HOVER
  asset.write_joint_velocity_to_sim(speeds, joint_ids=ids, env_ids=env_ids)


__all__ = [
  "action_rate_l2",
  "angular_velocity_l2",
  "attitude_termination",
  "body_angular_velocity",
  "body_linear_velocity",
  "body_up_alignment",
  "linear_velocity_l2",
  "position_error_b",
  "position_error_termination",
  "position_error_w",
  "position_tracking_exp",
  "previous_action",
  "projected_gravity",
  "reset_rotor_speeds",
  "root_height",
  "root_height_termination",
  "rotor_speed_observation",
  "target_position_w",
  "uprightness_reward",
  "yaw_error",
  "yaw_error_sin_cos",
  "yaw_tracking_exp",
]
