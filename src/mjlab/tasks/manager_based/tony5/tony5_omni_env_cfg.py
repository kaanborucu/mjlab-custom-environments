"""Fixed-speed omnidirectional TONY5 Aero velocity task configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.manager_based.tony5.tony5_aero_env_cfg import (
  tony5_velocity_aero_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_command import (
  Tony5OmniVelocityCommandCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_rewards import tony5_omni_rewards

TONY5_OMNI_MAX_HORIZONTAL_SPEED = 10.0
TONY5_OMNI_COMMAND_TRANSITION_TIME_S = 1.0 / 3.0

_RADIAL_METRICS = (
  "curriculum_stage",
  "current_vmax",
  "horizontal_velocity_rmse",
  "high_speed_steady_rmse",
  "mean_commanded_horizontal_speed",
  "mean_achieved_horizontal_speed",
)


def tony5_velocity_aero_omni_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the fixed 10 m/s omnidirectional TONY5 Aero task."""
  cfg = tony5_velocity_aero_env_cfg(play=play)
  cfg.rewards = tony5_omni_rewards()
  cfg.commands = {
    "velocity": Tony5OmniVelocityCommandCfg(
      entity_name="robot",
      stage_speeds=(TONY5_OMNI_MAX_HORIZONTAL_SPEED,),
      curriculum_switch_iterations=(),
      previous_speed_fraction=0.0,
      current_speed_fraction=1.0,
      high_speed_fraction=0.0,
      linear_velocity_frame="body",
      yaw_velocity_frame="body",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.2,
      play_high_speed_min=0.0,
      play_high_speed_max=TONY5_OMNI_MAX_HORIZONTAL_SPEED,
      debug_vis=True,
      ranges=Tony5OmniVelocityCommandCfg.Ranges(
        lin_vel_x=(-TONY5_OMNI_MAX_HORIZONTAL_SPEED, TONY5_OMNI_MAX_HORIZONTAL_SPEED),
        lin_vel_y=(-TONY5_OMNI_MAX_HORIZONTAL_SPEED, TONY5_OMNI_MAX_HORIZONTAL_SPEED),
        lin_vel_z=(-3.0, 3.0),
        ang_vel_z=(-1.5, 1.5),
      ),
      command_transition_time_s=TONY5_OMNI_COMMAND_TRANSITION_TIME_S,
    ),
  }
  for metric_name in _RADIAL_METRICS:
    cfg.metrics.pop(metric_name, None)
  return cfg


def tony5_velocity_aero_omni_play_env_cfg() -> ManagerBasedRlEnvCfg:
  """Return the one-environment fixed omnidirectional viewer configuration."""
  cfg = tony5_velocity_aero_omni_env_cfg(play=True)
  cfg.scene.num_envs = 1
  return cfg


__all__ = [
  "TONY5_OMNI_MAX_HORIZONTAL_SPEED",
  "TONY5_OMNI_COMMAND_TRANSITION_TIME_S",
  "tony5_velocity_aero_omni_env_cfg",
  "tony5_velocity_aero_omni_play_env_cfg",
]
