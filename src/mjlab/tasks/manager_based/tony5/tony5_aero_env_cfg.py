"""TONY5 V1 velocity task with additional aerodynamic physics."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.manager_based.tony5.tony5_aero_actions import (
  Tony5AeroRotorSpeedActionCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_physics import (
  configure_tony5_aero_scene,
  get_tony5_aero_robot_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_rewards import (
  body_angular_acceleration_l2,
  downward_velocity_l2,
  high_speed_roll_uprightness_reward,
  track_angular_velocity_speed_scaled,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_safety import (
  max_abs_qacc,
  max_body_angular_speed,
  max_root_speed,
  max_rotor_speed,
  numerical_safety_failure_rate,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_terminations import (
  TONY5_AERO_DOWNWARD_VELOCITY_LIMIT,
  numerical_safety_failure,
  rapid_descent_termination,
)
from mjlab.tasks.manager_based.tony5.tony5_velocity_curriculum import (
  Tony5RadialVelocityCommandCfg,
  tony5_radial_velocity_metrics,
  track_linear_velocity_adaptive,
)
from mjlab.tasks.manager_based.tony5.tony5_velocity_env_cfg import (
  tony5_velocity_env_cfg,
)

# Edit these V1-only coefficients to tune the active reward terms. V0 keeps its
# separate reward table in tony5_velocity_env_cfg.py.
TONY5_AERO_REWARD_WEIGHTS = {
  "linear_velocity_tracking": 5.0,
  "angular_velocity_tracking": 2.0,
  "action_rate": -0.01,
  "crash": -10.0,
  "body_angular_acceleration": -0.0000001 * 0,
  "downward_velocity": -0.1 * 1,
  "uprightness": 1.0 * 0,
}


def tony5_velocity_aero_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build V1 by preserving V0 task behavior and replacing only physics."""
  cfg = tony5_velocity_env_cfg(play=play)
  cfg.sim.nan_guard.enabled = True
  cfg.scene.entities["robot"] = get_tony5_aero_robot_cfg()
  cfg.scene.spec_fn = configure_tony5_aero_scene
  cfg.actions = {
    "rotor_speed": Tony5AeroRotorSpeedActionCfg(entity_name="robot"),
  }
  cfg.terminations.pop("attitude")
  cfg.terminations["rapid_descent"] = TerminationTermCfg(
    func=rapid_descent_termination,
    params={
      "minimum_vertical_velocity": TONY5_AERO_DOWNWARD_VELOCITY_LIMIT,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.terminations["numerical_safety_failure"] = TerminationTermCfg(
    func=numerical_safety_failure,
  )
  cfg.terminations["time_out"].log = False
  cfg.commands = {
    "velocity": Tony5RadialVelocityCommandCfg(
      entity_name="robot",
      stage_speeds=(2.0, 6.0, 10.0, 27.78),
      curriculum_switch_iterations=(50, 100, 200),
      curriculum_steps_per_iteration=24,
      linear_velocity_frame="body",
      yaw_velocity_frame="body",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.2,
      debug_vis=True,
      ranges=Tony5RadialVelocityCommandCfg.Ranges(
        lin_vel_x=(-27.78, 27.78),
        lin_vel_y=(-27.78, 27.78),
        lin_vel_z=(-3.0, 3.0),
        ang_vel_z=(-1.5, 1.5),
      ),
    ),
  }
  linear_velocity_reward = cfg.rewards["linear_velocity_tracking"]
  assert isinstance(linear_velocity_reward, RewardTermCfg)
  cfg.rewards["linear_velocity_tracking"] = RewardTermCfg(
    func=track_linear_velocity_adaptive,
    weight=TONY5_AERO_REWARD_WEIGHTS["linear_velocity_tracking"],
    params={"command_name": "velocity"},
  )
  cfg.rewards.pop("uprightness")
  cfg.rewards["body_angular_acceleration"] = RewardTermCfg(
    func=body_angular_acceleration_l2,
    weight=TONY5_AERO_REWARD_WEIGHTS["body_angular_acceleration"],
    log=False,
  )
  cfg.rewards["downward_velocity"] = RewardTermCfg(
    func=downward_velocity_l2,
    weight=TONY5_AERO_REWARD_WEIGHTS["downward_velocity"],
  )
  cfg.rewards["uprightness"] = RewardTermCfg(
    func=high_speed_roll_uprightness_reward,
    weight=TONY5_AERO_REWARD_WEIGHTS["uprightness"],
    params={"command_name": "velocity", "minimum_horizontal_speed": 8.0},
    log=False,
  )
  angular_velocity_reward = cfg.rewards["angular_velocity_tracking"]
  assert isinstance(angular_velocity_reward, RewardTermCfg)
  cfg.rewards["angular_velocity_tracking"] = RewardTermCfg(
    func=track_angular_velocity_speed_scaled,
    weight=TONY5_AERO_REWARD_WEIGHTS["angular_velocity_tracking"],
    params=angular_velocity_reward.params,
  )
  cfg.rewards["action_rate"].weight = TONY5_AERO_REWARD_WEIGHTS["action_rate"]
  cfg.rewards["crash"].weight = TONY5_AERO_REWARD_WEIGHTS["crash"]
  cfg.metrics = tony5_radial_velocity_metrics()
  cfg.metrics.update(
    {
      "numerical_safety_failure_rate": MetricsTermCfg(
        func=numerical_safety_failure_rate,
      ),
      "max_root_speed": MetricsTermCfg(func=max_root_speed, reduce="last"),
      "max_body_angular_speed": MetricsTermCfg(
        func=max_body_angular_speed,
        reduce="last",
      ),
      "max_rotor_speed": MetricsTermCfg(func=max_rotor_speed, reduce="last"),
      "max_abs_qacc": MetricsTermCfg(func=max_abs_qacc, reduce="last"),
    }
  )
  return cfg


def tony5_velocity_aero_play_env_cfg() -> ManagerBasedRlEnvCfg:
  """Return the one-environment V1 viewer configuration."""
  cfg = tony5_velocity_aero_env_cfg(play=True)
  cfg.scene.num_envs = 1
  return cfg


__all__ = [
  "TONY5_AERO_REWARD_WEIGHTS",
  "tony5_velocity_aero_env_cfg",
  "tony5_velocity_aero_play_env_cfg",
]
