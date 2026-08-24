"""Manager-based TONY5 V0 position+yaw environment configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manager_based.tony5 import tony5_mdp as mdp
from mjlab.tasks.manager_based.tony5.tony5_actions import Tony5RotorSpeedActionCfg
from mjlab.tasks.manager_based.tony5.tony5_commands import (
  Tony5PositionYawCommandCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  DECIMATION,
  PHYSICS_DT,
  ROTOR_JOINTS,
  get_tony5_robot_cfg,
)
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

ROTOR_CFG = SceneEntityCfg(
  "robot",
  joint_names=ROTOR_JOINTS,
  preserve_order=True,
)
ROBOT_CFG = SceneEntityCfg("robot")

REWARD_WEIGHTS = {
  "position_tracking": 4.0,
  "yaw_tracking": 0.5,
  "uprightness": 0.001,
  "linear_velocity": -0.05,
  "angular_velocity": -0.05,
  "action_rate": -0.01,
  "crash": -5.0,
}


def _observation_terms() -> dict[str, ObservationTermCfg]:
  return {
    "position_error_b": ObservationTermCfg(
      func=mdp.position_error_b,
      params={"position_scale": 2.0, "asset_cfg": ROBOT_CFG},
    ),
    "linear_velocity_b": ObservationTermCfg(
      func=mdp.body_linear_velocity,
      params={"asset_cfg": ROBOT_CFG},
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      params={"asset_cfg": ROBOT_CFG},
    ),
    "yaw_error_sin_cos": ObservationTermCfg(func=mdp.yaw_error_sin_cos),
    "angular_velocity_b": ObservationTermCfg(
      func=mdp.body_angular_velocity,
      params={"asset_cfg": ROBOT_CFG},
    ),
    "rotor_speed": ObservationTermCfg(
      func=mdp.rotor_speed_observation,
      params={"asset_cfg": ROTOR_CFG},
    ),
    "previous_action": ObservationTermCfg(func=mdp.previous_action),
  }


def tony5_position_yaw_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the TONY5 V0 training or fixed-command play configuration."""
  observations = {
    "actor": ObservationGroupCfg(
      terms=_observation_terms(),
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=_observation_terms(),
      enable_corruption=False,
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=8192,
      env_spacing=4.0,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": get_tony5_robot_cfg()},
    ),
    observations=observations,
    actions={
      "rotor_speed": Tony5RotorSpeedActionCfg(entity_name="robot"),
    },
    commands={
      "position_yaw": Tony5PositionYawCommandCfg(
        resampling_time_range=(4.0, 8.0),
        fixed=play,
        debug_vis=True,
      ),
    },
    events={
      "reset_scene_to_default": EventTermCfg(
        func=envs_mdp.reset_scene_to_default,
        mode="reset",
      ),
      "reset_rotor_speeds": EventTermCfg(
        func=mdp.reset_rotor_speeds,
        mode="reset",
        params={"asset_cfg": ROTOR_CFG},
      ),
    },
    rewards={
      "position_tracking": RewardTermCfg(
        func=mdp.position_tracking_exp,
        weight=REWARD_WEIGHTS["position_tracking"],
        params={"sigma": 0.4, "asset_cfg": ROBOT_CFG},
      ),
      "yaw_tracking": RewardTermCfg(
        func=mdp.yaw_tracking_exp,
        weight=REWARD_WEIGHTS["yaw_tracking"],
        params={"sigma": 0.5, "asset_cfg": ROBOT_CFG},
      ),
      "uprightness": RewardTermCfg(
        func=mdp.uprightness_reward,
        weight=REWARD_WEIGHTS["uprightness"],
        params={"asset_cfg": ROBOT_CFG},
      ),
      "linear_velocity": RewardTermCfg(
        func=mdp.linear_velocity_l2,
        weight=REWARD_WEIGHTS["linear_velocity"],
        params={"asset_cfg": ROBOT_CFG},
      ),
      "angular_velocity": RewardTermCfg(
        func=mdp.angular_velocity_l2,
        weight=REWARD_WEIGHTS["angular_velocity"],
        params={"asset_cfg": ROBOT_CFG},
      ),
      "action_rate": RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=REWARD_WEIGHTS["action_rate"],
      ),
      "crash": RewardTermCfg(
        func=envs_mdp.is_terminated,
        weight=REWARD_WEIGHTS["crash"],
      ),
    },
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
      "ground": TerminationTermCfg(
        func=mdp.root_height_termination,
        params={"minimum_height": 0.10, "asset_cfg": ROBOT_CFG},
      ),
      "position_error": TerminationTermCfg(
        func=mdp.position_error_termination,
        params={"limit": 4.0, "asset_cfg": ROBOT_CFG},
      ),
      "attitude": TerminationTermCfg(
        func=mdp.attitude_termination,
        params={"minimum_up_alignment": 0.2, "asset_cfg": ROBOT_CFG},
      ),
    },
    sim=SimulationCfg(
      mujoco=MujocoCfg(
        timestep=PHYSICS_DT,
        integrator="implicitfast",
        gravity=(0.0, 0.0, -9.80665),
      ),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="quad_base",
      distance=4.0,
      elevation=-20.0,
      azimuth=90.0,
    ),
    decimation=DECIMATION,
    episode_length_s=12.0,
  )


def tony5_position_yaw_play_env_cfg() -> ManagerBasedRlEnvCfg:
  """Return the play configuration with periodically resampled commands."""
  cfg = tony5_position_yaw_env_cfg(play=True)
  command_cfg = cfg.commands["position_yaw"]
  assert isinstance(command_cfg, Tony5PositionYawCommandCfg)
  command_cfg.fixed = False
  cfg.episode_length_s = 1e9
  cfg.observations["actor"].enable_corruption = False
  return cfg


__all__ = [
  "REWARD_WEIGHTS",
  "ROBOT_CFG",
  "ROTOR_CFG",
  "tony5_position_yaw_env_cfg",
  "tony5_position_yaw_play_env_cfg",
]
