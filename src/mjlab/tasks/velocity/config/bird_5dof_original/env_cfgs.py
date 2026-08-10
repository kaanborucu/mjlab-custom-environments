"""Frozen original forward-velocity task for the five-DoF bird."""

import math

import mujoco

from mjlab.asset_zoo.robots import (
  BIRD_5DOF_ORIGINAL_ACTION_SCALE,
  get_bird_5dof_original_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
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
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import ViewerConfig

ORIGINAL_REWARD_WEIGHTS = {
  "track_linear_velocity": 2.0,
  "track_angular_velocity": 0.5,
  "upright": 0.5,
  "alive": 0.2,
  "action_rate": -0.01,
  "joint_torques": -1.0e-4,
}

ORIGINAL_REWARD_STDS = {
  "track_linear_velocity": 0.5,
  "track_angular_velocity": math.sqrt(0.5),
  "upright": math.sqrt(0.2),
}


def _configure_original_atmosphere(spec: mujoco.MjSpec) -> None:
  spec.option.density = 1.225
  spec.option.viscosity = 1.81e-5
  spec.option.wind[:] = (0.0, 0.0, 0.0)


def bird_5dof_original_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Reproduce the first validated forward-only bird environment."""
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=dict(actor_terms),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      debug_vis=play,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(2.5, 4.0),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
      ),
    )
  }

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=ORIGINAL_REWARD_WEIGHTS["track_linear_velocity"],
      params={
        "command_name": "twist",
        "std": ORIGINAL_REWARD_STDS["track_linear_velocity"],
      },
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=ORIGINAL_REWARD_WEIGHTS["track_angular_velocity"],
      params={
        "command_name": "twist",
        "std": ORIGINAL_REWARD_STDS["track_angular_velocity"],
      },
    ),
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=ORIGINAL_REWARD_WEIGHTS["upright"],
      params={
        "std": ORIGINAL_REWARD_STDS["upright"],
        "asset_cfg": SceneEntityCfg("robot", body_names=("bird",)),
      },
    ),
    "alive": RewardTermCfg(
      func=mdp.is_alive,
      weight=ORIGINAL_REWARD_WEIGHTS["alive"],
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=ORIGINAL_REWARD_WEIGHTS["action_rate"],
    ),
    "joint_torques": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=ORIGINAL_REWARD_WEIGHTS["joint_torques"],
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "low_altitude": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.25},
    ),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(80.0)},
    ),
    "non_finite_state": TerminationTermCfg(func=mdp.nan_detection),
  }

  events = {
    "reset_scene_to_default": EventTermCfg(
      func=mdp.reset_scene_to_default,
      mode="reset",
    )
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=1 if play else 4096,
      extent=10.0,
      entities={"robot": get_bird_5dof_original_robot_cfg()},
      spec_fn=_configure_original_atmosphere,
    ),
    observations=observations,
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=BIRD_5DOF_ORIGINAL_ACTION_SCALE,
        use_default_offset=True,
      )
    },
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="bird",
      distance=1.5,
      elevation=-15.0,
      azimuth=140.0,
    ),
    sim=SimulationCfg(
      nconmax=4,
      njmax=32,
      mujoco=MujocoCfg(
        timestep=0.002,
        integrator="implicitfast",
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=10,
    episode_length_s=1.0e9 if play else 10.0,
  )
