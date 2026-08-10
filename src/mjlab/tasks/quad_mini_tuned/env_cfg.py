"""Flat-terrain Quad Mini Tuned environment configuration."""

from __future__ import annotations

import math

from mjlab.asset_zoo.robots.quad_mini_tuned.quad_constants import (
  FEET_GEOMS,
  FEET_SITES,
  JOINT_NAMES,
  get_quad_mini_tuned_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.quad_mini_tuned import mdp
from mjlab.tasks.quad_mini_tuned.mdp import QuadMiniVelocityCommandCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import NoiseModelWithAdditiveBiasCfg, UniformNoiseCfg
from mjlab.viewer import ViewerConfig

ROBOT_CFG = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
FEET_CFG = SceneEntityCfg("robot", site_names=FEET_SITES, preserve_order=True)
PRIVILEGED_CFG = SceneEntityCfg(
  "robot",
  joint_names=JOINT_NAMES,
  site_names=FEET_SITES,
  preserve_order=True,
)
ACTUATOR_CFG = SceneEntityCfg(
  "robot", actuator_names=list(JOINT_NAMES), preserve_order=True
)
FOOT_GEOM_CFG = SceneEntityCfg("robot", geom_names=FEET_GEOMS, preserve_order=True)

REWARD_WEIGHTS = {
  "tracking_lin_vel": 5.0,
  "tracking_ang_vel": 2.5,
  "lin_vel_z": -1.0,
  "ang_vel_xy": -0.15,
  "orientation": -5.0,
  "base_height": -1.0e-8,
  "dof_pos_limits": -1.0,
  "pose": 0.5,
  "termination": -0.001,
  "stand_still": -0.001,
  "torques": -0.0001,
  "action_rate": -0.01,
  "energy": -0.001,
  "dof_acc": -1.0e-7,
  "dof_vel": -0.00002,
  "feet_clearance": -2.0,
  "feet_height": -0.1,
  "feet_slip": -0.1,
  "impact_feet_vel": -0.001,
  "feet_air_time": 0.20,
  "diagonal_trot": 0.00001,
  "all_feet_sync": -0.09,
  "three_feet_support": -0.00001,
}


def _bias_noise(scale: float) -> NoiseModelWithAdditiveBiasCfg:
  return NoiseModelWithAdditiveBiasCfg(
    noise_cfg=UniformNoiseCfg(n_min=-scale, n_max=scale),
    bias_noise_cfg=UniformNoiseCfg(n_min=-scale, n_max=scale),
  )


def _contact_sensor() -> ContactSensorCfg:
  return ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FEET_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


def quad_mini_tuned_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Quad Mini Tuned flat-terrain task configuration."""
  actor_terms = {
    "accelerometer": ObservationTermCfg(
      func=mdp.imu_acceleration,
      params={
        "accelerometer_sensor_name": "robot/imu_linacc",
        "gravity_sensor_name": "robot/upvector",
      },
      noise=_bias_noise(0.2),
      delay_min_lag=1,
      delay_max_lag=1,
    ),
    "gyro": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_angvel"},
      noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
      delay_min_lag=1,
      delay_max_lag=1,
    ),
    "gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity_from_sensor,
      params={"sensor_name": "robot/upvector"},
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
      delay_min_lag=1,
      delay_max_lag=1,
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"biased": True, "asset_cfg": ROBOT_CFG},
      noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": ROBOT_CFG},
      noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
    ),
    "torque": ObservationTermCfg(
      func=mdp.torque_observation,
      params={"asset_cfg": ROBOT_CFG},
      noise=_bias_noise(0.5),
      delay_min_lag=1,
      delay_max_lag=1,
    ),
    "last_action": ObservationTermCfg(func=envs_mdp.last_action),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }

  scene = SceneCfg(
    num_envs=8192,
    env_spacing=2.0,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_quad_mini_tuned_robot_cfg()},
    sensors=(_contact_sensor(),),
  )

  events = {
    "reset_scene_to_default": EventTermCfg(
      mode="reset", func=envs_mdp.reset_scene_to_default
    ),
    "reset_root_state": EventTermCfg(
      mode="reset",
      func=envs_mdp.reset_root_state_uniform,
      params={
        "pose_range": {"z": (-0.005, 0.005)},
        "velocity_range": {"z": (-0.05, 0.05)},
        "asset_cfg": ROBOT_CFG,
      },
    ),
    "reset_joint_state": EventTermCfg(
      mode="reset",
      func=envs_mdp.reset_joints_by_offset,
      params={
        "position_range": (-0.005, 0.005),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": ROBOT_CFG,
      },
    ),
    "floor_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("terrain", geom_names=("terrain",)),
        "operation": "abs",
        "ranges": (0.4, 1.5),
        "shared_random": True,
      },
    ),
    "joint_friction": EventTermCfg(
      mode="startup",
      func=dr.joint_friction,
      params={"asset_cfg": ROBOT_CFG, "operation": "scale", "ranges": (0.85, 1.15)},
    ),
    "joint_armature": EventTermCfg(
      mode="startup",
      func=dr.joint_armature,
      params={"asset_cfg": ROBOT_CFG, "operation": "scale", "ranges": (0.95, 1.10)},
    ),
    "joint_damping": EventTermCfg(
      mode="startup",
      func=dr.joint_damping,
      params={"asset_cfg": ROBOT_CFG, "operation": "scale", "ranges": (0.7, 1.5)},
    ),
    "torso_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
        "operation": "add",
        "ranges": {0: (-0.10, 0.10), 1: (-0.10, 0.10), 2: (-0.10, 0.10)},
      },
    ),
    "link_mass": EventTermCfg(
      mode="startup",
      func=mdp.quad_body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "ranges": (0.90, 1.05),
      },
    ),
    "torso_payload": EventTermCfg(
      mode="startup",
      func=mdp.quad_body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
        "operation": "add",
        "ranges": (-1.0, 3.0),
      },
    ),
    "body_inertia": EventTermCfg(
      mode="startup",
      func=mdp.quad_body_inertia_scale,
      params={"asset_cfg": SceneEntityCfg("robot"), "ranges": (0.8, 1.2)},
    ),
    "default_joint_pose": EventTermCfg(
      mode="startup",
      func=mdp.quad_default_joint_pose,
      params={"asset_cfg": ROBOT_CFG, "ranges": (-0.03, 0.03)},
    ),
    "pd_gains": EventTermCfg(
      mode="startup",
      func=dr.pd_gains,
      params={
        "asset_cfg": ACTUATOR_CFG,
        "operation": "abs",
        "kp_range": (20.0, 40.0),
        "kd_range": (0.15, 0.50),
      },
    ),
    "effort_limits": EventTermCfg(
      mode="startup",
      func=dr.effort_limits,
      params={
        "asset_cfg": ACTUATOR_CFG,
        "operation": "scale",
        "effort_limit_range": (0.75, 1.15),
      },
    ),
    "foot_radius": EventTermCfg(
      mode="startup",
      func=dr.geom_size,
      params={
        "asset_cfg": FOOT_GEOM_CFG,
        "operation": "scale",
        "axes": [0],
        "ranges": (0.85, 1.15),
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={"asset_cfg": ROBOT_CFG, "bias_range": (-0.03, 0.03)},
    ),
  }

  rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=mdp.quad_tracking_linear_velocity,
      weight=REWARD_WEIGHTS["tracking_lin_vel"],
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=mdp.quad_tracking_angular_velocity,
      weight=REWARD_WEIGHTS["tracking_ang_vel"],
      params={"command_name": "twist", "sigma": 0.25},
    ),
    "lin_vel_z": RewardTermCfg(
      func=mdp.quad_lin_vel_z, weight=REWARD_WEIGHTS["lin_vel_z"]
    ),
    "ang_vel_xy": RewardTermCfg(
      func=mdp.quad_ang_vel_xy, weight=REWARD_WEIGHTS["ang_vel_xy"]
    ),
    "orientation": RewardTermCfg(
      func=mdp.quad_orientation, weight=REWARD_WEIGHTS["orientation"]
    ),
    "base_height": RewardTermCfg(
      func=mdp.quad_base_height,
      weight=REWARD_WEIGHTS["base_height"],
      params={"target_height": 0.25},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=REWARD_WEIGHTS["dof_pos_limits"],
      params={"asset_cfg": ROBOT_CFG},
    ),
    "pose": RewardTermCfg(func=mdp.quad_pose, weight=REWARD_WEIGHTS["pose"]),
    "termination": RewardTermCfg(
      func=envs_mdp.is_terminated, weight=REWARD_WEIGHTS["termination"]
    ),
    "stand_still": RewardTermCfg(
      func=mdp.quad_stand_still,
      weight=REWARD_WEIGHTS["stand_still"],
      params={"command_name": "twist"},
    ),
    "torques": RewardTermCfg(
      func=mdp.quad_torques_cost,
      weight=REWARD_WEIGHTS["torques"],
      params={"asset_cfg": ROBOT_CFG},
    ),
    "action_rate": RewardTermCfg(
      func=envs_mdp.action_rate_l2, weight=REWARD_WEIGHTS["action_rate"]
    ),
    "energy": RewardTermCfg(
      func=mdp.quad_energy_cost,
      weight=REWARD_WEIGHTS["energy"],
      params={"asset_cfg": ROBOT_CFG},
    ),
    "dof_acc": RewardTermCfg(func=mdp.quad_dof_acc, weight=REWARD_WEIGHTS["dof_acc"]),
    "dof_vel": RewardTermCfg(func=mdp.quad_dof_vel, weight=REWARD_WEIGHTS["dof_vel"]),
    "feet_clearance": RewardTermCfg(
      func=mdp.quad_feet_clearance,
      weight=REWARD_WEIGHTS["feet_clearance"],
      params={
        "target_height": 0.25,
        "command_name": "twist",
        "asset_cfg": FEET_CFG,
      },
    ),
    "feet_height": RewardTermCfg(
      func=mdp.QuadFeetHeight,
      weight=REWARD_WEIGHTS["feet_height"],
      params={
        "sensor_name": "feet_ground_contact",
        "target_height": 0.25,
        "command_name": "twist",
        "asset_cfg": FEET_CFG,
      },
    ),
    "feet_slip": RewardTermCfg(
      func=velocity_mdp.feet_slip,
      weight=REWARD_WEIGHTS["feet_slip"],
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.01,
        "asset_cfg": FEET_CFG,
      },
    ),
    "impact_feet_vel": RewardTermCfg(
      func=mdp.quad_impact_feet_velocity,
      weight=REWARD_WEIGHTS["impact_feet_vel"],
      params={"sensor_name": "feet_ground_contact", "asset_cfg": FEET_CFG},
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.quad_feet_air_time,
      weight=REWARD_WEIGHTS["feet_air_time"],
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    ),
    "diagonal_trot": RewardTermCfg(
      func=mdp.quad_diagonal_trot,
      weight=REWARD_WEIGHTS["diagonal_trot"],
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    ),
    "all_feet_sync": RewardTermCfg(
      func=mdp.quad_all_feet_sync,
      weight=REWARD_WEIGHTS["all_feet_sync"],
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    ),
    "three_feet_support": RewardTermCfg(
      func=mdp.quad_three_feet_support,
      weight=REWARD_WEIGHTS["three_feet_support"],
      params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    ),
  }

  cfg = ManagerBasedRlEnvCfg(
    decimation=5,
    scene=scene,
    observations={
      "actor": ObservationGroupCfg(
        terms=actor_terms, concatenate_terms=True, enable_corruption=True
      ),
      "critic": ObservationGroupCfg(
        terms={
          "privileged_state": ObservationTermCfg(
            func=mdp.quad_privileged_observation,
            params={
              "command_name": "twist",
              "contact_sensor_name": "feet_ground_contact",
              "asset_cfg": PRIVILEGED_CFG,
            },
          )
        },
        concatenate_terms=True,
        enable_corruption=False,
      ),
    },
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=JOINT_NAMES,
        scale=0.6,
        use_default_offset=True,
      )
    },
    commands={
      "twist": QuadMiniVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        ranges=QuadMiniVelocityCommandCfg.Ranges(
          lin_vel_x=(-1.5, 1.5),
          lin_vel_y=(-1.0, 1.0),
          ang_vel_z=(-1.5, 1.5),
        ),
        command_amplitudes=(1.5, 1.0, 1.5),
        zero_probabilities=(0.1, 0.75, 0.5),
        debug_vis=True,
      )
    },
    events=events,
    rewards=rewards,
    terminations={
      "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
      "fell_over": TerminationTermCfg(
        func=envs_mdp.bad_orientation,
        params={"limit_angle": math.pi / 2.0, "asset_cfg": ROBOT_CFG},
      ),
    },
    sim=SimulationCfg(
      nconmax=8 * 8192,
      njmax=128,
      contact_sensor_maxmatch=64,
      mujoco=MujocoCfg(
        timestep=0.004,
        cone="elliptic",
        impratio=10.0,
        ccd_iterations=50,
      ),
    ),
    viewer=ViewerConfig(body_name="trunk", distance=1.5, elevation=-10.0),
    episode_length_s=20.0,
    scale_rewards_by_dt=True,
  )

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 1.0e9
    cfg.observations["actor"].enable_corruption = False
    cfg.commands["twist"].debug_vis = True

  return cfg
