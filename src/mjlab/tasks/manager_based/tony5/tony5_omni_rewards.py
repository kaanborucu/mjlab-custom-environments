"""Editable reward configuration for the fixed omnidirectional TONY5 task."""

from __future__ import annotations

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
from mjlab.tasks.manager_based.tony5.tony5_velocity_curriculum import (
  track_linear_velocity_adaptive,
)

OMNI_ROBOT_CFG = SceneEntityCfg("robot")

# Edit these Omni-only values when tuning this task. V0 and Aero-v1 have their
# own reward tables and are not affected by changes here.
TONY5_OMNI_REWARD_WEIGHTS = {
  "linear_velocity_tracking": 5.0,
  "angular_velocity_tracking": 3.0,
  "action_rate": -0.04,
  "crash": -100.0,
  "body_angular_acceleration": -0.0000001 * 0,
  "downward_velocity": -0.1 * 0,
  "uprightness": 1.0 * 0,
}

TONY5_OMNI_ANGULAR_TRACKING_STD = 2.0


def tony5_omni_rewards() -> dict[str, RewardTermCfg]:
  """Return the independently editable Omni reward terms."""
  return {
    "linear_velocity_tracking": RewardTermCfg(
      func=track_linear_velocity_adaptive,
      weight=TONY5_OMNI_REWARD_WEIGHTS["linear_velocity_tracking"],
      params={"command_name": "velocity"},
    ),
    "angular_velocity_tracking": RewardTermCfg(
      func=track_angular_velocity_speed_scaled,
      weight=TONY5_OMNI_REWARD_WEIGHTS["angular_velocity_tracking"],
      params={
        "std": TONY5_OMNI_ANGULAR_TRACKING_STD,
        "command_name": "velocity",
        "asset_cfg": OMNI_ROBOT_CFG,
      },
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=TONY5_OMNI_REWARD_WEIGHTS["action_rate"],
    ),
    "crash": RewardTermCfg(
      func=envs_mdp.is_terminated,
      weight=TONY5_OMNI_REWARD_WEIGHTS["crash"],
    ),
    "body_angular_acceleration": RewardTermCfg(
      func=body_angular_acceleration_l2,
      weight=TONY5_OMNI_REWARD_WEIGHTS["body_angular_acceleration"],
      log=False,
    ),
    "downward_velocity": RewardTermCfg(
      func=downward_velocity_l2,
      weight=TONY5_OMNI_REWARD_WEIGHTS["downward_velocity"],
    ),
    "uprightness": RewardTermCfg(
      func=high_speed_roll_uprightness_reward,
      weight=TONY5_OMNI_REWARD_WEIGHTS["uprightness"],
      params={"command_name": "velocity", "minimum_horizontal_speed": 8.0},
      log=False,
    ),
  }


__all__ = [
  "TONY5_OMNI_ANGULAR_TRACKING_STD",
  "TONY5_OMNI_REWARD_WEIGHTS",
  "tony5_omni_rewards",
]
