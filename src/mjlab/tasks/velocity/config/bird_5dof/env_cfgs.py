"""Velocity-tracking environment for the five-DoF flapping-wing bird."""

import math

import mujoco

from mjlab.asset_zoo.robots import (
  BIRD_5DOF_ACTION_SCALE,
  get_bird_5dof_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
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

from .events import reset_heading_forward_velocity
from .flight_profile import (
  DIRECTION_MIN_SPEED,
  DIRECTION_REWARD_STD,
  DIRECTIONAL_FLIGHT_REWARD_WEIGHTS,
  LOW_SPEED_WORLD_UP_MAX_SPEED,
  ROLL_FULL_SPEED,
  ROLL_MIN_HORIZONTAL,
  ROLL_REWARD_STD,
  ROLL_ZERO_SPEED,
  TILT_LIMIT_DEG,
)

# Reward tuning.
#
# Change values here instead of editing the reward definitions below. Positive
# weights are rewards; negative weights are penalties.
REWARD_WEIGHTS = {
  "track_linear_velocity": 1.0,
  **DIRECTIONAL_FLIGHT_REWARD_WEIGHTS,
  "alive": 0.01,
  "action_rate": -0.01,
  "joint_torques": -1.0e-6,
  "joint_velocity": -0.50e-7,
  "joint_acceleration": -0.50e-7,
}

# Smaller standard deviations make tracking/stability rewards stricter.
REWARD_STDS = {
  "track_linear_velocity": 5.0,
}

# Curriculum thresholds count environment steps. The PPO runner collects 24
# environment steps per iteration.
_PPO_STEPS_PER_ITERATION = 24
VELOCITY_CURRICULUM_STAGES = [
  {
    "step": 0,
    "name": "proven_forward_3d",
    "lin_vel_x": (0.0, 10.0),
    "lin_vel_y": (-4.0, 4.0),
    "lin_vel_z": (-6.0, 6.0),
    "ang_vel_z": (0.0, 0.0),
    "rel_standing_envs": 0.0,
    "rel_forward_envs": 0.0,
    "resampling_time_range": (2.0, 4.0),
    "reset_pose_range": {
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-math.pi, math.pi),
    },
    "reset_velocity_range": {
      "x": (-0.1, 0.1),
      "y": (-0.1, 0.1),
      "z": (-0.1, 0.1),
      "roll": (-0.2, 0.2),
      "pitch": (-0.2, 0.2),
      "yaw": (-0.2, 0.2),
    },
    "joint_position_ranges": {
      "reset_flap_joints": (-0.2, 0.2),
      "reset_feather_joints": (-0.2, 0.2),
      "reset_tail_joint": (-0.1, 0.1),
    },
  },
  {
    "step": 200 * _PPO_STEPS_PER_ITERATION,
    "name": "hover_and_stall_recovery",
    "lin_vel_x": (0.0, 10.0),
    "lin_vel_y": (-4.0, 4.0),
    "lin_vel_z": (-6.0, 6.0),
    "ang_vel_z": (0.0, 0.0),
    "rel_standing_envs": 0.15,
    "rel_forward_envs": 0.0,
    "resampling_time_range": (2.0, 4.0),
    "reset_pose_range": {
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
    },
    "reset_velocity_range": {
      "x": (-0.1, 0.1),
      "y": (-0.1, 0.1),
      "z": (-0.1, 0.1),
      "roll": (-0.2, 0.2),
      "pitch": (-0.2, 0.2),
      "yaw": (-0.2, 0.2),
    },
    "joint_position_ranges": {
      "reset_flap_joints": (-0.2, 0.2),
      "reset_feather_joints": (-0.2, 0.2),
      "reset_tail_joint": (-0.1, 0.1),
    },
  },
  {
    "step": 400 * _PPO_STEPS_PER_ITERATION,
    "name": "full_world_directions",
    "lin_vel_x": (-10.0, 10.0),
    "lin_vel_y": (-6.0, 6.0),
    "lin_vel_z": (-6.0, 6.0),
    "ang_vel_z": (0.0, 0.0),
    "rel_standing_envs": 0.1,
    "rel_forward_envs": 0.0,
    "resampling_time_range": (2.0, 4.0),
    "reset_pose_range": {
      "roll": (-0.15, 0.15),
      "pitch": (-0.15, 0.15),
    },
    "reset_velocity_range": {
      "x": (-0.35, 0.35),
      "y": (-0.35, 0.35),
      "z": (-0.35, 0.35),
      "roll": (-0.35, 0.35),
      "pitch": (-0.35, 0.35),
      "yaw": (-0.35, 0.35),
    },
    "joint_position_ranges": {
      "reset_flap_joints": (-0.4, 0.4),
      "reset_feather_joints": (-0.4, 0.4),
      "reset_tail_joint": (-0.2, 0.2),
    },
  },
  {
    "step": 600 * _PPO_STEPS_PER_ITERATION,
    "name": "rapid_redirection",
    "lin_vel_x": (-10.0, 10.0),
    "lin_vel_y": (-6.0, 6.0),
    "lin_vel_z": (-6.0, 6.0),
    "ang_vel_z": (0.0, 0.0),
    "rel_standing_envs": 0.1,
    "rel_forward_envs": 0.0,
    "resampling_time_range": (1.5, 3.0),
    "reset_pose_range": {
      "roll": (-0.25, 0.25),
      "pitch": (-0.25, 0.25),
    },
    "reset_velocity_range": {
      "x": (-0.75, 0.75),
      "y": (-0.75, 0.75),
      "z": (-0.75, 0.75),
      "roll": (-0.75, 0.75),
      "pitch": (-0.75, 0.75),
      "yaw": (-0.75, 0.75),
    },
    "joint_position_ranges": {
      "reset_flap_joints": (-0.6, 0.6),
      "reset_feather_joints": (-0.6, 0.6),
      "reset_tail_joint": (-0.3, 0.3),
    },
  },
  {
    "step": 800 * _PPO_STEPS_PER_ITERATION,
    "name": "moderate_disturbances",
    "lin_vel_x": (-10.0, 10.0),
    "lin_vel_y": (-6.0, 6.0),
    "lin_vel_z": (-6.0, 6.0),
    "ang_vel_z": (0.0, 0.0),
    "rel_standing_envs": 0.1,
    "rel_forward_envs": 0.0,
    "resampling_time_range": (1.5, 3.0),
    "reset_pose_range": {
      "roll": (-0.35, 0.35),
      "pitch": (-0.35, 0.35),
    },
    "reset_velocity_range": {
      "x": (-1.0, 1.0),
      "y": (-1.0, 1.0),
      "z": (-1.0, 1.0),
      "roll": (-1.0, 1.0),
      "pitch": (-1.0, 1.0),
      "yaw": (-1.0, 1.0),
    },
    "joint_position_ranges": {
      "reset_flap_joints": (-0.8, 0.8),
      "reset_feather_joints": (-0.8, 0.8),
      "reset_tail_joint": (-0.4, 0.4),
    },
  },
  {
    "step": 1000 * _PPO_STEPS_PER_ITERATION,
    "name": "robustness",
    "lin_vel_x": (-10.0, 10.0),
    "lin_vel_y": (-6.0, 6.0),
    "lin_vel_z": (-6.0, 6.0),
    "ang_vel_z": (0.0, 0.0),
    "rel_standing_envs": 0.1,
    "rel_forward_envs": 0.0,
    "resampling_time_range": (1.5, 3.0),
    "reset_pose_range": {
      "roll": (-0.5, 0.5),
      "pitch": (-0.5, 0.5),
    },
    "reset_velocity_range": {
      "x": (-1.5, 1.5),
      "y": (-1.5, 1.5),
      "z": (-1.5, 1.5),
      "roll": (-1.5, 1.5),
      "pitch": (-1.5, 1.5),
      "yaw": (-1.5, 1.5),
    },
    "joint_position_ranges": {
      "reset_flap_joints": (-1.0, 1.0),
      "reset_feather_joints": (-1.0, 1.0),
      "reset_tail_joint": (-0.5, 0.5),
    },
  },
]


def _configure_atmosphere(spec: mujoco.MjSpec) -> None:
  """Restore fluid settings after the robot MJCF is attached to the scene."""
  spec.option.density = 1.225
  spec.option.viscosity = 1.81e-5
  spec.option.wind[:] = (0.0, 0.0, 0.0)


def bird_5dof_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the omnidirectional 3D velocity-tracking task."""
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel_heading),
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
      resampling_time_range=(1.5, 3.0) if play else (2.0, 4.0),
      rel_standing_envs=0.1 if play else 0.0,
      rel_world_envs=1.0,
      rel_forward_envs=0.0,
      linear_velocity_frame="heading",
      linear_velocity_sampling="ellipsoid",
      yaw_velocity_frame="world",
      yaw_command_speed_threshold=None,
      debug_vis=play,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-10.0, 10.0) if play else (0.0, 10.0),
        lin_vel_y=(-6.0, 6.0) if play else (-4.0, 4.0),
        lin_vel_z=(-6.0, 6.0),
        ang_vel_z=(0.0, 0.0),
      ),
    )
  }

  events: dict[str, EventTermCfg] = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "roll": (-0.1, 0.1),
          "pitch": (-0.1, 0.1),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {
          "x": (-0.1, 0.1),
          "y": (-0.1, 0.1),
          "z": (-0.1, 0.1),
          "roll": (-0.2, 0.2),
          "pitch": (-0.2, 0.2),
          "yaw": (-0.2, 0.2),
        },
      },
    ),
    "reset_flap_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.2, 0.2),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=("left_flap", "right_flap"),
        ),
      },
    ),
    "reset_feather_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.2, 0.2),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=("left_feather", "right_feather"),
        ),
      },
    ),
    "reset_tail_joint": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.1, 0.1),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=("tail_pitch",)),
      },
    ),
    "reset_forward_velocity": EventTermCfg(
      func=reset_heading_forward_velocity,
      mode="reset",
      params={"speed_range": (0.75, 1.5)},
    ),
  }

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity_heading,
      weight=REWARD_WEIGHTS["track_linear_velocity"],
      params={
        "command_name": "twist",
        "std": REWARD_STDS["track_linear_velocity"],
      },
    ),
    "velocity_direction_alignment": RewardTermCfg(
      func=mdp.track_commanded_velocity_direction,
      weight=REWARD_WEIGHTS["velocity_direction_alignment"],
      params={
        "command_name": "twist",
        "std": DIRECTION_REWARD_STD,
        "min_speed": DIRECTION_MIN_SPEED,
      },
    ),
    "low_speed_world_up": RewardTermCfg(
      func=mdp.low_speed_world_up_alignment,
      weight=REWARD_WEIGHTS["low_speed_world_up"],
      params={
        "command_name": "twist",
        "max_speed": LOW_SPEED_WORLD_UP_MAX_SPEED,
      },
    ),
    "level_roll": RewardTermCfg(
      func=mdp.level_roll_alignment,
      weight=REWARD_WEIGHTS["level_roll"],
      params={
        "std": ROLL_REWARD_STD,
        "min_horizontal_forward": ROLL_MIN_HORIZONTAL,
        "command_name": "twist",
        "full_speed": ROLL_FULL_SPEED,
        "zero_speed": ROLL_ZERO_SPEED,
      },
    ),
    "roll_pitch_angular_velocity": RewardTermCfg(
      func=mdp.base_roll_pitch_angular_velocity_l2,
      weight=REWARD_WEIGHTS["roll_pitch_angular_velocity"],
    ),
    "alive": RewardTermCfg(
      func=mdp.is_alive,
      weight=REWARD_WEIGHTS["alive"],
    ),
    "action_rate": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=REWARD_WEIGHTS["action_rate"],
    ),
    "joint_torques": RewardTermCfg(
      func=mdp.joint_torques_l2,
      weight=REWARD_WEIGHTS["joint_torques"],
    ),
    "joint_velocity": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=REWARD_WEIGHTS["joint_velocity"],
    ),
    "joint_acceleration": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=REWARD_WEIGHTS["joint_acceleration"],
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "non_finite_state": TerminationTermCfg(func=mdp.nan_detection),
    "excessive_tilt": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(TILT_LIMIT_DEG)},
    ),
  }

  curriculum = {
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": VELOCITY_CURRICULUM_STAGES,
      },
    )
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=1 if play else 8192,
      extent=10.0,
      entities={"robot": get_bird_5dof_robot_cfg()},
      spec_fn=_configure_atmosphere,
    ),
    observations=observations,
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=BIRD_5DOF_ACTION_SCALE,
        use_default_offset=True,
      )
    },
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={} if play else curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="bird",
      distance=1.5,
      elevation=-15.0,
      azimuth=140.0,
      enable_reflections=False,
      enable_shadows=False,
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
    episode_length_s=1.0e9 if play else 20.0,
  )
