"""Rewards for minimum-time target-pose flight."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manager_based.tony5 import tony5_mdp as mdp
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_rewards import (
  rotor_torque_l2,
)
from mjlab.tasks.manager_based.tony5.tony5_position_min_time_terminations import (
  success_condition,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS = {
  "distance_progress": 10.0,
  "time_penalty": -0.14,
  "settling": -0.00001,
  "action_rate": -0.02,
  "motor_torque": -1.0e-7,
  "success": 300.0,
  "crash": -100.0,
}

TONY5_POSITION_MIN_TIME_SETTLING_DISTANCE_SCALE = 0.75
TONY5_POSITION_MIN_TIME_SETTLING_SPEED_SCALE = 0.75
TONY5_POSITION_MIN_TIME_YAW_PROGRESS_SCALE = 0.75


def _distance_to_target(
  env: ManagerBasedRlEnv,
  command_name: str = "target_pose",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  error = mdp.position_error_w(env, command_name, asset_cfg)
  return torch.linalg.vector_norm(error, dim=-1)


class Tony5DistanceProgressReward(ManagerTermBase):
  """Return positive progress from the previous policy step."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    super().__init__(env)
    self._previous = torch.zeros(self.num_envs, device=self.device)
    self._previous_yaw_error = torch.zeros(self.num_envs, device=self.device)
    self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str = "target_pose",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    current = _distance_to_target(env, command_name, asset_cfg)
    current_yaw_error = mdp.yaw_error(env, command_name, asset_cfg).abs()
    progress = (
      self._previous - current
    ) + TONY5_POSITION_MIN_TIME_YAW_PROGRESS_SCALE * (
      self._previous_yaw_error - current_yaw_error
    )
    progress = torch.where(self._initialized, progress, torch.zeros_like(progress))
    self._previous[:] = current
    self._previous_yaw_error[:] = current_yaw_error
    self._initialized[:] = True
    return progress

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._previous[env_ids] = 0.0
    self._previous_yaw_error[env_ids] = 0.0
    self._initialized[env_ids] = False


def time_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return one per policy step; the reward weight supplies the penalty."""
  return torch.ones(env.num_envs, device=env.device)


def near_target_settling_cost(
  env: ManagerBasedRlEnv,
  command_name: str = "target_pose",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distance_scale: float = TONY5_POSITION_MIN_TIME_SETTLING_DISTANCE_SCALE,
  speed_scale: float = TONY5_POSITION_MIN_TIME_SETTLING_SPEED_SCALE,
) -> torch.Tensor:
  """Return a smooth, bounded cost for motion near the target pose.

  The Gaussian distance gate removes the cost far from the target.  The
  exponential speed term is zero at rest and approaches one smoothly as the
  combined linear/angular speed grows, avoiding a hard near-target switch.
  """
  asset = env.scene[asset_cfg.name]
  position_error = mdp.position_error_w(env, command_name, asset_cfg)
  position_error_squared = torch.sum(torch.square(position_error), dim=-1)
  near_target_gate = torch.exp(-0.5 * position_error_squared / distance_scale**2)
  linear_speed_squared = torch.sum(torch.square(asset.data.root_link_lin_vel_w), dim=-1)
  angular_speed_squared = torch.sum(
    torch.square(asset.data.root_link_ang_vel_b), dim=-1
  )
  normalized_speed_squared = (
    linear_speed_squared + angular_speed_squared
  ) / speed_scale**2
  bounded_speed = 1.0 - torch.exp(-0.5 * normalized_speed_squared)
  return near_target_gate * bounded_speed


def success_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Read the dedicated success termination, never the generic done flag."""
  if "success" not in env.termination_manager.active_terms:
    return torch.zeros(env.num_envs, device=env.device)
  return env.termination_manager.get_term("success").float()


def crash_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize safety failures, excluding both success and timeout."""
  active_terms = env.termination_manager.active_terms
  if (
    "rapid_descent" not in active_terms
    or "numerical_safety_failure" not in active_terms
  ):
    return torch.zeros(env.num_envs, device=env.device)
  crash = env.termination_manager.get_term(
    "rapid_descent"
  ) | env.termination_manager.get_term("numerical_safety_failure")
  if "success" in active_terms:
    success = env.termination_manager.get_term("success")
  else:
    success = success_condition(env)
  return (crash & ~success).float()


def tony5_position_min_time_rewards() -> dict[str, RewardTermCfg]:
  """Return the complete V5 reward table in policy-step units."""
  return {
    "distance_progress": RewardTermCfg(
      func=Tony5DistanceProgressReward,
      weight=TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS["distance_progress"],
    ),
    "time_penalty": RewardTermCfg(
      func=time_penalty,
      weight=TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS["time_penalty"],
    ),
    "settling": RewardTermCfg(
      func=near_target_settling_cost,
      weight=TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS["settling"],
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS["action_rate"],
    ),
    "motor_torque": RewardTermCfg(
      func=rotor_torque_l2,
      weight=TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS["motor_torque"],
    ),
    "success": RewardTermCfg(
      func=success_reward,
      weight=TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS["success"],
    ),
    "crash": RewardTermCfg(
      func=crash_penalty,
      weight=TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS["crash"],
    ),
  }


__all__ = [
  "Tony5DistanceProgressReward",
  "TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS",
  "TONY5_POSITION_MIN_TIME_SETTLING_DISTANCE_SCALE",
  "TONY5_POSITION_MIN_TIME_SETTLING_SPEED_SCALE",
  "TONY5_POSITION_MIN_TIME_YAW_PROGRESS_SCALE",
  "crash_penalty",
  "near_target_settling_cost",
  "success_reward",
  "time_penalty",
  "tony5_position_min_time_rewards",
]
