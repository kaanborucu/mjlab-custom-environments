"""TONY5 Omni V3 task with global shared wind and correlated gusts."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manager_based.tony5.tony5_omni_command import (
  Tony5OmniVelocityCommandCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_env_cfg import (
  tony5_velocity_aero_omni_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_actions import (
  Tony5OmniV3RotorSpeedActionCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_battery import (
  Tony5OmniV3BatteryCfg,
  voltage_sag,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_observations import (
  heading_sin_cos,
  world_linear_velocity,
  world_velocity_command,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_physics import (
  Tony5OmniV3PhysicsCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_prop import (
  Tony5OmniV3PropCfg,
  static_cq0,
  static_ct0,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_rewards import (
  tony5_omni_v3_rewards,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_wind import (
  Tony5OmniV3WindCfg,
  wind_speed,
)


def tony5_velocity_aero_omni_v3_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build V3 from a fresh copy of the existing Omni V0 configuration."""
  cfg = tony5_velocity_aero_omni_env_cfg(play=play)
  velocity_cfg = cfg.commands["velocity"]
  if not isinstance(velocity_cfg, Tony5OmniVelocityCommandCfg):
    raise TypeError("TONY5 Omni V3 requires the Omni velocity command config.")
  velocity_cfg.linear_velocity_frame = "world"
  velocity_cfg.yaw_velocity_frame = "body"
  velocity_cfg.resampling_time_range = (1.0, 10.0)
  velocity_cfg.enable_command_smoothing = False
  velocity_cfg.command_transition_time_s = 0.0
  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]
    group.terms = {
      name: term for name, term in group.terms.items() if name != "linear_velocity_b"
    }
    group.terms["command"] = ObservationTermCfg(
      func=world_velocity_command,
      params={"command_name": "velocity"},
    )
    group.terms["linear_velocity_w"] = ObservationTermCfg(
      func=world_linear_velocity,
      params={"asset_cfg": SceneEntityCfg("robot")},
    )
    group.terms["heading_sin_cos"] = ObservationTermCfg(
      func=heading_sin_cos,
      params={"asset_cfg": SceneEntityCfg("robot")},
    )
  cfg.rewards = tony5_omni_v3_rewards()
  cfg.actions = {
    "rotor_speed": Tony5OmniV3RotorSpeedActionCfg(
      entity_name="robot",
      physics=Tony5OmniV3PhysicsCfg(
        prop=Tony5OmniV3PropCfg(
          enable_advanced_prop_model=True,
          ct_curve_j=(0.0,),
          ct_curve_values=(static_ct0(),),
          cq_curve_j=(0.0,),
          cq_curve_values=(static_cq0(),),
        ),
        enable_blade_flapping=True,
      ),
      battery=Tony5OmniV3BatteryCfg(enable_battery_sag=True),
      wind=Tony5OmniV3WindCfg(
        enable_wind=True,
        wind_x=2.0,
        randomize_background_wind=True,
        background_horizontal_speed_range=(0.0, 10.0),
        background_vertical_speed_range=(-1.0, 1.0),
        enable_gusts=True,
        gust_sigma=1.0,
        gust_tau=1.0,
        max_gust_speed=5.0,
      ),
    ),
  }
  # Keep routine training logs focused on learning and numerical validity.
  # Detailed prop, flapping, electrical, and per-axis wind diagnostics remain
  # available in the diagnostic code and are not part of the PPO signal.
  cfg.metrics["wind_speed"] = MetricsTermCfg(func=wind_speed)
  cfg.metrics["voltage_sag"] = MetricsTermCfg(func=voltage_sag)
  return cfg


def tony5_velocity_aero_omni_v3_play_env_cfg() -> ManagerBasedRlEnvCfg:
  """Return the one-environment V3 viewer configuration."""
  cfg = tony5_velocity_aero_omni_v3_env_cfg(play=True)
  cfg.rewards = {name: term for name, term in cfg.rewards.items() if term.weight != 0.0}
  action_cfg = cfg.actions["rotor_speed"]
  if not isinstance(action_cfg, Tony5OmniV3RotorSpeedActionCfg):
    raise TypeError("TONY5 Omni V3 play requires the V3 rotor-speed action config.")
  # No measured CT(J)/CQ(J) data are available yet. This one-point curve is
  # derived exactly from kT/kQ, so play mode can exercise the advanced path
  # without inventing an inflow-dependent aerodynamic curve.
  action_cfg.physics = Tony5OmniV3PhysicsCfg(
    prop=Tony5OmniV3PropCfg(
      enable_advanced_prop_model=True,
      ct_curve_j=(0.0,),
      ct_curve_values=(static_ct0(),),
      cq_curve_j=(0.0,),
      cq_curve_values=(static_cq0(),),
    ),
    enable_blade_flapping=True,
  )
  action_cfg.battery = Tony5OmniV3BatteryCfg(enable_battery_sag=True)
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
  cfg.scene.num_envs = 1
  return cfg


__all__ = [
  "tony5_velocity_aero_omni_v3_env_cfg",
  "tony5_velocity_aero_omni_v3_play_env_cfg",
]
