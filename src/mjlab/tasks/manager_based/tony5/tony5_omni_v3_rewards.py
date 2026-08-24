"""V3-only rewards for world-frame linear velocity commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manager_based.tony5 import tony5_mdp as mdp
from mjlab.tasks.manager_based.tony5.tony5_aero_rewards import (
  body_angular_acceleration_l2,
  downward_velocity_l2,
  high_speed_roll_uprightness_reward,
  track_angular_velocity_speed_scaled,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_observations import (
  world_velocity_command,
)
from mjlab.tasks.manager_based.tony5.tony5_velocity_curriculum import (
  velocity_reward_sigma,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# Edit these V3-only values when tuning this task. V0, Aero-v1, and Omni-v0
# reward tables are defined separately and are not affected by these values.
TONY5_OMNI_V3_REWARD_WEIGHTS = {
  "linear_velocity_tracking": 5.0,
  "angular_velocity_tracking": 3.0,
  "action_rate": -0.04,
  "crash": -100.0,
  "body_angular_acceleration": -0.0000001 * 0,
  "downward_velocity": -0.1 * 0,
  "uprightness": 1.0 * 0,
  "motor_torque": -1.0e-6,
}

TONY5_OMNI_V3_ANGULAR_TRACKING_STD = 2.0
TONY5_OMNI_V3_UPRIGHT_MINIMUM_HORIZONTAL_SPEED = 8.0

# Backward-compatible named alias for the currently requested torque weight.
TONY5_OMNI_V3_MOTOR_TORQUE_PENALTY_WEIGHT = TONY5_OMNI_V3_REWARD_WEIGHTS["motor_torque"]


def rotor_torque_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return the squared sum of the four actual rotor actuator torques."""
  asset = env.scene[asset_cfg.name]
  # ``_joint_ids`` are entity-local joint indices, while
  # ``joint_v_adr`` contains generalized-velocity addresses.  Indexing the
  # latter with the former can select invalid addresses on CUDA.  The native
  # actuator-force view is already restricted to this asset's four motor
  # actuators and is the correct actuation-space torque signal for this term.
  rotor_torque = asset.data.actuator_force
  return torch.sum(torch.square(rotor_torque), dim=-1)


def track_linear_velocity_world(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track commanded x/y/z velocity in world coordinates."""
  asset = env.scene[asset_cfg.name]
  command = world_velocity_command(env, command_name)
  actual_w = asset.data.root_link_lin_vel_w
  xy_error = torch.sum(torch.square(command[:, :2] - actual_w[:, :2]), dim=1)
  z_error = torch.square(command[:, 3] - actual_w[:, 2])
  linear_velocity_error = xy_error + z_error
  commanded_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
  sigma_v = velocity_reward_sigma(commanded_speed)
  return torch.exp(-linear_velocity_error / torch.square(sigma_v))


def tony5_omni_v3_rewards() -> dict[str, RewardTermCfg]:
  """Return the complete independently editable V3 reward table."""
  return {
    "linear_velocity_tracking": RewardTermCfg(
      func=track_linear_velocity_world,
      weight=TONY5_OMNI_V3_REWARD_WEIGHTS["linear_velocity_tracking"],
      params={"command_name": "velocity", "asset_cfg": _DEFAULT_ASSET_CFG},
    ),
    "angular_velocity_tracking": RewardTermCfg(
      func=track_angular_velocity_speed_scaled,
      weight=TONY5_OMNI_V3_REWARD_WEIGHTS["angular_velocity_tracking"],
      params={
        "std": TONY5_OMNI_V3_ANGULAR_TRACKING_STD,
        "command_name": "velocity",
        "asset_cfg": _DEFAULT_ASSET_CFG,
      },
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=TONY5_OMNI_V3_REWARD_WEIGHTS["action_rate"],
    ),
    "crash": RewardTermCfg(
      func=envs_mdp.is_terminated,
      weight=TONY5_OMNI_V3_REWARD_WEIGHTS["crash"],
    ),
    "body_angular_acceleration": RewardTermCfg(
      func=body_angular_acceleration_l2,
      weight=TONY5_OMNI_V3_REWARD_WEIGHTS["body_angular_acceleration"],
      log=False,
    ),
    "downward_velocity": RewardTermCfg(
      func=downward_velocity_l2,
      weight=TONY5_OMNI_V3_REWARD_WEIGHTS["downward_velocity"],
      log=False,
    ),
    "uprightness": RewardTermCfg(
      func=high_speed_roll_uprightness_reward,
      weight=TONY5_OMNI_V3_REWARD_WEIGHTS["uprightness"],
      params={
        "command_name": "velocity",
        "minimum_horizontal_speed": (TONY5_OMNI_V3_UPRIGHT_MINIMUM_HORIZONTAL_SPEED),
        "asset_cfg": _DEFAULT_ASSET_CFG,
      },
      log=False,
    ),
    "motor_torque": RewardTermCfg(
      func=rotor_torque_l2,
      weight=TONY5_OMNI_V3_REWARD_WEIGHTS["motor_torque"],
      params={"asset_cfg": _DEFAULT_ASSET_CFG},
    ),
  }


__all__ = [
  "TONY5_OMNI_V3_ANGULAR_TRACKING_STD",
  "TONY5_OMNI_V3_MOTOR_TORQUE_PENALTY_WEIGHT",
  "TONY5_OMNI_V3_REWARD_WEIGHTS",
  "TONY5_OMNI_V3_UPRIGHT_MINIMUM_HORIZONTAL_SPEED",
  "rotor_torque_l2",
  "tony5_omni_v3_rewards",
  "track_linear_velocity_world",
]
