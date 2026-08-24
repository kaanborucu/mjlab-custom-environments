"""TONY5 V0 body-velocity tracking environment configuration."""

from __future__ import annotations

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.manager_based.tony5 import tony5_mdp as mdp
from mjlab.tasks.manager_based.tony5.tony5_actions import Tony5RotorSpeedActionCfg
from mjlab.tasks.manager_based.tony5.tony5_env_cfg import (
  ROBOT_CFG,
  ROTOR_CFG,
  tony5_position_yaw_env_cfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

ROBOT_VELOCITY_CFG = SceneEntityCfg("robot")

VELOCITY_REWARD_WEIGHTS = {
  "linear_velocity_tracking": 1.5,
  "angular_velocity_tracking": 0.35,
  "uprightness": 0.1,
  "action_rate": -0.05,
  "crash": -5.0,
}

LINEAR_VELOCITY_TRACKING_STD = 0.8
ANGULAR_VELOCITY_TRACKING_STD = 0.9


def _disable_ground_collision(spec: mujoco.MjSpec) -> None:
  """Keep the ground visible while making it non-colliding for TONY5."""
  ground = spec.geom("terrain")
  ground.contype = 0
  ground.conaffinity = 0


def _velocity_observation_terms() -> dict[str, ObservationTermCfg]:
  return {
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "velocity"},
    ),
    "linear_velocity_b": ObservationTermCfg(
      func=mdp.body_linear_velocity,
      params={"asset_cfg": ROBOT_VELOCITY_CFG},
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      params={"asset_cfg": ROBOT_VELOCITY_CFG},
    ),
    "angular_velocity_b": ObservationTermCfg(
      func=mdp.body_angular_velocity,
      params={"asset_cfg": ROBOT_VELOCITY_CFG},
    ),
    "root_height": ObservationTermCfg(
      func=mdp.root_height,
      params={"asset_cfg": ROBOT_VELOCITY_CFG},
    ),
    "rotor_speed": ObservationTermCfg(
      func=mdp.rotor_speed_observation,
      params={"asset_cfg": ROTOR_CFG},
    ),
    "previous_action": ObservationTermCfg(func=mdp.previous_action),
  }


def _velocity_observations(play: bool) -> dict[str, ObservationGroupCfg]:
  terms = _velocity_observation_terms()
  return {
    "actor": ObservationGroupCfg(
      terms=terms,
      enable_corruption=not play,
      history_length=5,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=_velocity_observation_terms(),
      enable_corruption=False,
      history_length=5,
      flatten_history_dim=True,
    ),
  }


def tony5_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the TONY5 body-velocity tracking environment."""
  cfg = tony5_position_yaw_env_cfg()
  cfg.scene.spec_fn = _disable_ground_collision
  cfg.observations = _velocity_observations(play)
  cfg.actions = {
    "rotor_speed": Tony5RotorSpeedActionCfg(entity_name="robot"),
  }
  cfg.commands = {
    "velocity": UniformVelocityCommandCfg(
      entity_name="robot",
      linear_velocity_frame="body",
      yaw_velocity_frame="body",
      resampling_time_range=(3.0, 8.0),
      # Explicit zero-velocity commands train stable hover behavior.
      rel_standing_envs=0.2,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-2.0, 2.0),
        lin_vel_y=(-2.0, 2.0),
        lin_vel_z=(-0.75, 0.75),
        ang_vel_z=(-1.5, 1.5),
      ),
    ),
  }
  cfg.rewards = {
    "linear_velocity_tracking": RewardTermCfg(
      func=velocity_mdp.track_linear_velocity,
      weight=VELOCITY_REWARD_WEIGHTS["linear_velocity_tracking"],
      params={
        "std": LINEAR_VELOCITY_TRACKING_STD,
        "command_name": "velocity",
        "asset_cfg": ROBOT_CFG,
      },
    ),
    "angular_velocity_tracking": RewardTermCfg(
      func=velocity_mdp.track_angular_velocity,
      weight=VELOCITY_REWARD_WEIGHTS["angular_velocity_tracking"],
      params={
        "std": ANGULAR_VELOCITY_TRACKING_STD,
        "command_name": "velocity",
        "asset_cfg": ROBOT_CFG,
      },
    ),
    "uprightness": RewardTermCfg(
      func=mdp.uprightness_reward,
      weight=VELOCITY_REWARD_WEIGHTS["uprightness"],
      params={"asset_cfg": ROBOT_CFG},
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=VELOCITY_REWARD_WEIGHTS["action_rate"],
    ),
    "crash": RewardTermCfg(
      func=envs_mdp.is_terminated,
      weight=VELOCITY_REWARD_WEIGHTS["crash"],
    ),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "attitude": TerminationTermCfg(
      func=mdp.attitude_termination,
      params={"minimum_up_alignment": 0.2, "asset_cfg": ROBOT_CFG},
    ),
  }
  if play:
    cfg.episode_length_s = 1e9
  return cfg


def tony5_velocity_play_env_cfg() -> ManagerBasedRlEnvCfg:
  """Return the velocity-tracking viewer configuration."""
  cfg = tony5_velocity_env_cfg(play=True)
  cfg.scene.num_envs = 1
  return cfg


__all__ = [
  "ROBOT_VELOCITY_CFG",
  "VELOCITY_REWARD_WEIGHTS",
  "tony5_velocity_env_cfg",
  "tony5_velocity_play_env_cfg",
]
