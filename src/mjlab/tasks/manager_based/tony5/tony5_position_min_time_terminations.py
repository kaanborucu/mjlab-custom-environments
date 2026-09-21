"""Success and safety conditions for the TONY5 minimum-time task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manager_based.tony5.tony5_aero_safety import (
  numerical_safety_failure_with_limits,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_terminations import (
  rapid_descent_termination,
)
from mjlab.tasks.manager_based.tony5.tony5_mdp import (
  target_position_w,
  yaw_error,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

TONY5_POSITION_MIN_TIME_ROOT_SPEED_LIMIT = 100.0
TONY5_POSITION_MIN_TIME_BODY_ANGULAR_SPEED_LIMIT = 20.0
TONY5_POSITION_MIN_TIME_QACC_LIMIT = 300_000.0

TONY5_SUCCESS_POSITION_TOLERANCE = 0.28125
TONY5_SUCCESS_YAW_TOLERANCE = 5.0 * torch.pi / 180.0
TONY5_SUCCESS_LINEAR_SPEED_TOLERANCE = 0.5
TONY5_SUCCESS_ANGULAR_SPEED_TOLERANCE = 0.75


def success_mask(
  position_error: torch.Tensor,
  yaw_error_abs: torch.Tensor,
  linear_speed: torch.Tensor,
  angular_speed: torch.Tensor,
  position_tolerance: float = TONY5_SUCCESS_POSITION_TOLERANCE,
  yaw_tolerance: float = TONY5_SUCCESS_YAW_TOLERANCE,
  linear_speed_tolerance: float = TONY5_SUCCESS_LINEAR_SPEED_TOLERANCE,
  angular_speed_tolerance: float = TONY5_SUCCESS_ANGULAR_SPEED_TOLERANCE,
) -> torch.Tensor:
  """Return the exact all-conditions success mask."""
  return (
    (position_error < position_tolerance)
    & (yaw_error_abs < yaw_tolerance)
    & (linear_speed < linear_speed_tolerance)
    & (angular_speed < angular_speed_tolerance)
  )


def success_condition(
  env: ManagerBasedRlEnv,
  position_tolerance: float = TONY5_SUCCESS_POSITION_TOLERANCE,
  yaw_tolerance: float = TONY5_SUCCESS_YAW_TOLERANCE,
  linear_speed_tolerance: float = TONY5_SUCCESS_LINEAR_SPEED_TOLERANCE,
  angular_speed_tolerance: float = TONY5_SUCCESS_ANGULAR_SPEED_TOLERANCE,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  command_name: str = "target_pose",
) -> torch.Tensor:
  """Terminate successfully inside the target pose and speed tolerances."""
  asset: Entity = env.scene[asset_cfg.name]
  position_error = torch.linalg.vector_norm(
    target_position_w(env, command_name) - asset.data.root_link_pos_w,
    dim=-1,
  )
  yaw_error_abs = yaw_error(env, command_name, asset_cfg).abs()
  linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1)
  angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_b, dim=-1)
  return success_mask(
    position_error,
    yaw_error_abs,
    linear_speed,
    angular_speed,
    position_tolerance,
    yaw_tolerance,
    linear_speed_tolerance,
    angular_speed_tolerance,
  )


def rapid_descent(
  env: ManagerBasedRlEnv,
  minimum_vertical_velocity: float = -5.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate when the vehicle is descending at an unsafe rate."""
  return rapid_descent_termination(env, minimum_vertical_velocity, asset_cfg)


def safety_failure(
  env: ManagerBasedRlEnv,
  minimum_vertical_velocity: float = -5.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return the dedicated crash condition used by V5 diagnostics/rewards."""
  return rapid_descent(env, minimum_vertical_velocity, asset_cfg) | (
    numerical_safety_failure_with_limits(
      env,
      root_speed_limit=TONY5_POSITION_MIN_TIME_ROOT_SPEED_LIMIT,
      qacc_limit=TONY5_POSITION_MIN_TIME_QACC_LIMIT,
    )
  )


__all__ = [
  "TONY5_POSITION_MIN_TIME_BODY_ANGULAR_SPEED_LIMIT",
  "TONY5_POSITION_MIN_TIME_QACC_LIMIT",
  "TONY5_POSITION_MIN_TIME_ROOT_SPEED_LIMIT",
  "TONY5_SUCCESS_ANGULAR_SPEED_TOLERANCE",
  "TONY5_SUCCESS_LINEAR_SPEED_TOLERANCE",
  "TONY5_SUCCESS_POSITION_TOLERANCE",
  "TONY5_SUCCESS_YAW_TOLERANCE",
  "rapid_descent",
  "safety_failure",
  "success_condition",
  "success_mask",
]
