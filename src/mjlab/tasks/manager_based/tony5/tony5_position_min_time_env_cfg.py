"""TONY5 V5 minimum-time target-pose environment configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.manager_based.tony5 import tony5_mdp as mdp
from mjlab.tasks.manager_based.tony5.tony5_aero_safety import (
  numerical_safety_failure_with_limits,
)
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  BATTERY_VOLTAGE,
  OMEGA_MAX,
  ROTOR_JOINTS,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_actions import (
  Tony5OmniV3RotorSpeedAction,
  Tony5OmniV3RotorSpeedActionCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_battery import (
  electrical_power,
  loaded_voltage,
  total_current,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_env_cfg import (
  tony5_velocity_aero_omni_v3_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_observations import (
  heading_sin_cos,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_wind import (
  Tony5OmniV3WindCfg,
  gust_x,
  gust_y,
  gust_z,
  wind_x,
  wind_y,
  wind_z,
)
from mjlab.tasks.manager_based.tony5.tony5_position_min_time_command import (
  Tony5PositionMinTimeCommandCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_position_min_time_metrics import (
  tony5_position_min_time_metrics,
)
from mjlab.tasks.manager_based.tony5.tony5_position_min_time_rewards import (
  tony5_position_min_time_rewards,
)
from mjlab.tasks.manager_based.tony5.tony5_position_min_time_terminations import (
  TONY5_POSITION_MIN_TIME_BODY_ANGULAR_SPEED_LIMIT,
  TONY5_POSITION_MIN_TIME_QACC_LIMIT,
  TONY5_POSITION_MIN_TIME_ROOT_SPEED_LIMIT,
  rapid_descent,
  success_condition,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ROBOT_CFG = SceneEntityCfg("robot")
ROTOR_CFG = SceneEntityCfg(
  "robot",
  joint_names=ROTOR_JOINTS,
  preserve_order=True,
)


def _observation_terms() -> dict[str, ObservationTermCfg]:
  return {
    "target_position_error_b": ObservationTermCfg(
      func=mdp.position_error_b,
      params={
        "command_name": "target_pose",
        "position_scale": 1.0,
        "asset_cfg": ROBOT_CFG,
      },
    ),
    "target_yaw_error_sin_cos": ObservationTermCfg(
      func=mdp.yaw_error_sin_cos,
      params={"command_name": "target_pose", "asset_cfg": ROBOT_CFG},
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      params={"asset_cfg": ROBOT_CFG},
    ),
    "angular_velocity_b": ObservationTermCfg(
      func=mdp.body_angular_velocity,
      params={"asset_cfg": ROBOT_CFG},
    ),
    "linear_velocity_b": ObservationTermCfg(
      func=mdp.body_linear_velocity,
      params={"asset_cfg": ROBOT_CFG},
    ),
    "heading_sin_cos": ObservationTermCfg(
      func=heading_sin_cos,
      params={"asset_cfg": ROBOT_CFG},
    ),
    "root_height": ObservationTermCfg(
      func=mdp.root_height,
      params={"asset_cfg": ROBOT_CFG},
    ),
    "rotor_speed": ObservationTermCfg(
      func=mdp.rotor_speed_observation,
      params={"asset_cfg": ROTOR_CFG},
    ),
    "previous_action": ObservationTermCfg(func=mdp.previous_action),
  }


def _observations(play: bool) -> dict[str, ObservationGroupCfg]:
  return {
    "actor": ObservationGroupCfg(
      terms=_observation_terms(),
      enable_corruption=not play,
      history_length=5,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=_observation_terms(),
      enable_corruption=False,
      history_length=5,
      flatten_history_dim=True,
    ),
  }


def _target_pose(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return the complete commanded target pose as privileged state."""
  command = env.command_manager.get_command("target_pose")
  if command is None:
    raise ValueError("TONY5 teacher requires the target_pose command.")
  return command


def _root_position_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  asset = env.scene[ROBOT_CFG.name]
  return asset.data.root_link_pos_w - env.scene.env_origins


def _root_quaternion_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene[ROBOT_CFG.name].data.root_link_quat_w


def _root_linear_velocity_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene[ROBOT_CFG.name].data.root_link_lin_vel_b


def _root_angular_velocity_w(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene[ROBOT_CFG.name].data.root_link_ang_vel_w


def _rotor_position(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene[ROTOR_CFG.name].data.joint_pos[:, ROTOR_CFG.joint_ids]


def _rotor_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene[ROTOR_CFG.name].data.joint_vel[:, ROTOR_CFG.joint_ids] / OMEGA_MAX


def _rotor_speed_targets(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = cast(
    Tony5OmniV3RotorSpeedAction,
    env.action_manager.get_term("rotor_speed"),
  )
  return action.speed_targets / OMEGA_MAX


def _last_voltage(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = cast(
    Tony5OmniV3RotorSpeedAction,
    env.action_manager.get_term("rotor_speed"),
  )
  return action.last_voltage / BATTERY_VOLTAGE


def _wind_world(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.stack((wind_x(env), wind_y(env), wind_z(env)), dim=-1)


def _gust_world(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.stack((gust_x(env), gust_y(env), gust_z(env)), dim=-1)


def _privileged_observation_terms() -> dict[str, ObservationTermCfg]:
  """Return one-frame teacher observations with simulator privileged state."""
  return {
    **_observation_terms(),
    "target_pose": ObservationTermCfg(func=_target_pose),
    "root_position_w": ObservationTermCfg(func=_root_position_w),
    "root_quaternion_w": ObservationTermCfg(func=_root_quaternion_w),
    "root_linear_velocity_b": ObservationTermCfg(func=_root_linear_velocity_b),
    "root_angular_velocity_w": ObservationTermCfg(func=_root_angular_velocity_w),
    "rotor_position": ObservationTermCfg(func=_rotor_position),
    "rotor_velocity": ObservationTermCfg(func=_rotor_velocity),
    "rotor_speed_targets": ObservationTermCfg(func=_rotor_speed_targets),
    "last_voltage": ObservationTermCfg(func=_last_voltage),
    "wind_world": ObservationTermCfg(func=_wind_world),
    "gust_world": ObservationTermCfg(func=_gust_world),
    "loaded_voltage": ObservationTermCfg(func=loaded_voltage),
    "total_current": ObservationTermCfg(func=total_current),
    "electrical_power": ObservationTermCfg(func=electrical_power),
  }


def _privileged_observations(play: bool) -> dict[str, ObservationGroupCfg]:
  terms = _privileged_observation_terms()
  return {
    "actor": ObservationGroupCfg(
      terms=terms,
      enable_corruption=False,
      history_length=1,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms=_privileged_observation_terms(),
      enable_corruption=False,
      history_length=1,
      flatten_history_dim=True,
    ),
  }


def tony5_position_min_time_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build V5 from a fresh V3-derived config without mutating V3."""
  cfg = tony5_velocity_aero_omni_v3_env_cfg(play=play)
  cfg.episode_length_s = 10.0
  cfg.scale_rewards_by_dt = False
  cfg.observations = _observations(play)
  cfg.commands = {
    "target_pose": Tony5PositionMinTimeCommandCfg(
      # Required by CommandTermCfg; V5 overrides compute() to never resample
      # before the environment reset.
      resampling_time_range=(1.0, 1.0),
      horizontal_distance_range=(1.0, 15.0),
      vertical_displacement_range=(-3.0, 3.0),
      minimum_target_height=0.5,
      debug_vis=True,
    ),
  }
  cfg.rewards = tony5_position_min_time_rewards()
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "success": TerminationTermCfg(func=success_condition),
    "rapid_descent": TerminationTermCfg(func=rapid_descent),
    "numerical_safety_failure": TerminationTermCfg(
      func=numerical_safety_failure_with_limits,
      params={
        "root_speed_limit": TONY5_POSITION_MIN_TIME_ROOT_SPEED_LIMIT,
        "body_angular_speed_limit": (TONY5_POSITION_MIN_TIME_BODY_ANGULAR_SPEED_LIMIT),
        "qacc_limit": TONY5_POSITION_MIN_TIME_QACC_LIMIT,
      },
    ),
  }
  cfg.metrics.update(tony5_position_min_time_metrics())

  action_cfg = cfg.actions["rotor_speed"]
  if not isinstance(action_cfg, Tony5OmniV3RotorSpeedActionCfg):
    raise TypeError("TONY5 V5 requires the V3 rotor-speed action config.")
  # Match V3's randomized background wind and correlated gust process.
  action_cfg.wind = Tony5OmniV3WindCfg(
    enable_wind=True,
    wind_x=2.0,
    randomize_background_wind=True,
    background_horizontal_speed_range=(0.0, 10.0),
    background_vertical_speed_range=(-1.0, 1.0),
    enable_gusts=True,
    gust_sigma=1.0,
    gust_tau=1.0,
    max_gust_speed=5.0,
  )
  return cfg


def tony5_position_min_time_play_env_cfg() -> ManagerBasedRlEnvCfg:
  """Return the single-environment V5 viewer configuration."""
  cfg = tony5_position_min_time_env_cfg(play=True)
  cfg.scene.num_envs = 1
  return cfg


def tony5_position_min_time_teacher_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the privileged one-frame V5 teacher environment."""
  cfg = tony5_position_min_time_env_cfg(play=play)
  cfg.observations = _privileged_observations(play)
  return cfg


def tony5_position_min_time_teacher_play_env_cfg() -> ManagerBasedRlEnvCfg:
  """Return the single-environment privileged V5 teacher viewer config."""
  cfg = tony5_position_min_time_teacher_env_cfg(play=True)
  cfg.scene.num_envs = 1
  return cfg


__all__ = [
  "tony5_position_min_time_env_cfg",
  "tony5_position_min_time_play_env_cfg",
  "tony5_position_min_time_teacher_env_cfg",
  "tony5_position_min_time_teacher_play_env_cfg",
]
