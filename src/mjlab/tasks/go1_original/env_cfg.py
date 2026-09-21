"""Teacher and student observation variants of the original flat Go1 task."""

from __future__ import annotations

import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.go1.env_cfgs import (
  unitree_go1_flat_env_cfg,
  unitree_go1_rough_env_cfg,
)

# Edit these tables to change both the custom teacher and student tasks for a
# terrain. Rough mode adds the collision penalties removed from flat mode.
TORQUE_REWARD_WEIGHT = -2.0e-3
FLAT_REWARD_WEIGHTS: dict[str, float] = {
  "track_linear_velocity": 4.0,
  "track_angular_velocity": 3.0,
  "upright": 1.0,
  "pose": 1.0,
  "body_ang_vel": 0.0,
  "angular_momentum": 0.0,
  "dof_pos_limits": -1.0,
  "action_rate_l2": -0.05,
  "air_time": 0.0,
  "foot_clearance": -2.0,
  "foot_swing_height": -0.25,
  "foot_slip": -0.1,
  "soft_landing": -1.0e-5,
  "joint_torques": TORQUE_REWARD_WEIGHT,
}
ROUGH_REWARD_WEIGHTS: dict[str, float] = {
  **FLAT_REWARD_WEIGHTS,
  "self_collisions": -0.1,
  "shank_collision": -0.1,
  "trunk_head_collision": -0.1,
}

# Reset-time domain randomization. Edit these constants to change the custom
# teacher and student tasks without changing the original Go1 environments.
PD_KP_RANGE = (25.0, 45.0)
PD_KD_RANGE = (1.0, 2.5)
BASE_MASS_SCALE_RANGE = (0.85, 1.15)
THIGH_MASS_SCALE_RANGE = (0.90, 1.10)
CALF_MASS_SCALE_RANGE = (0.90, 1.10)
LINK_INERTIA_SCALE_RANGE = (0.85, 1.15)
BASE_COM_OFFSET_RANGES = {
  0: (-0.008, 0.008),
  1: (-0.008, 0.008),
  2: (-0.005, 0.005),
}
LEG_COM_OFFSET_RANGES = {
  0: (-0.003, 0.003),
  1: (-0.003, 0.003),
  2: (-0.003, 0.003),
}
GROUND_FRICTION_RANGE = (0.5, 1.5)
JOINT_DAMPING_SCALE_RANGE = (0.6, 1.4)
JOINT_FRICTION_SCALE_RANGE = (0.5, 1.5)
EFFORT_SCALE_RANGE = (0.85, 1.15)
INITIAL_JOINT_POSITION_RANGE = (-0.08, 0.08)
INITIAL_JOINT_VELOCITY_RANGE = (-0.5, 0.5)
INITIAL_ROLL_PITCH_RANGE = (-math.radians(5.0), math.radians(5.0))
INITIAL_BASE_VELOCITY_RANGE = (-0.25, 0.25)
INITIAL_YAW_RATE_RANGE = (-0.25, 0.25)
PUSH_HORIZONTAL_VELOCITY_RANGE = (-0.8, 0.8)
PUSH_YAW_RATE_RANGE = (-0.5, 0.5)
STUDENT_HISTORY_LENGTH = 5
STUDENT_HISTORY_TERMS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)

GO1_BODY_NAMES = (
  "trunk",
  "FR_hip",
  "FL_hip",
  "RR_hip",
  "RL_hip",
  "FR_thigh",
  "FL_thigh",
  "RR_thigh",
  "RL_thigh",
  "FR_calf",
  "FL_calf",
  "RR_calf",
  "RL_calf",
)
GO1_THIGH_BODY_NAMES = ("FR_thigh", "FL_thigh", "RR_thigh", "RL_thigh")
GO1_CALF_BODY_NAMES = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")


def _apply_reward_weights(
  cfg: ManagerBasedRlEnvCfg,
  weights: dict[str, float],
) -> None:
  """Apply an editable reward-weight table to a task configuration."""
  missing = set(weights) - set(cfg.rewards)
  if missing:
    raise KeyError(f"Reward weights reference missing terms: {sorted(missing)}")
  for name, weight in weights.items():
    cfg.rewards[name].weight = weight


def _add_torque_reward(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add the editable squared actuator-torque penalty term."""
  cfg.rewards["joint_torques"] = RewardTermCfg(
    func=mdp.joint_torques_l2,
    weight=TORQUE_REWARD_WEIGHT,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )


def _single_frame(group):
  """Return an observation group with an explicit one-frame history."""
  group = deepcopy(group)
  group.history_length = 1
  group.flatten_history_dim = True
  return group


def _student_observation(group):
  """Add temporal context without repeating commands or the terrain scan."""
  group = deepcopy(group)
  group.history_length = None
  group.flatten_history_dim = True
  for name, term in group.terms.items():
    term.history_length = STUDENT_HISTORY_LENGTH if name in STUDENT_HISTORY_TERMS else 0
    term.flatten_history_dim = True
  return group


def _configure_reset_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Configure initial-state and occasional push randomization."""
  cfg.events["reset_base"].params["pose_range"].update(
    {
      "roll": INITIAL_ROLL_PITCH_RANGE,
      "pitch": INITIAL_ROLL_PITCH_RANGE,
      "yaw": (-math.pi, math.pi),
    }
  )
  cfg.events["reset_base"].params["velocity_range"] = {
    "x": INITIAL_BASE_VELOCITY_RANGE,
    "y": INITIAL_BASE_VELOCITY_RANGE,
    "yaw": INITIAL_YAW_RATE_RANGE,
  }

  cfg.events["reset_robot_joints"].params["position_range"] = (
    INITIAL_JOINT_POSITION_RANGE
  )
  cfg.events["reset_robot_joints"].params["velocity_range"] = (
    INITIAL_JOINT_VELOCITY_RANGE
  )

  push_cfg = cfg.events.get("push_robot")
  if push_cfg is not None:
    push_cfg.params["velocity_range"] = {
      "x": PUSH_HORIZONTAL_VELOCITY_RANGE,
      "y": PUSH_HORIZONTAL_VELOCITY_RANGE,
      "yaw": PUSH_YAW_RATE_RANGE,
    }


def _configure_physics_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Configure reset-time physical-parameter randomization."""
  # Replace the original startup COM perturbation with the requested reset-time
  # base and leg COM ranges below.
  cfg.events.pop("base_com", None)

  cfg.events["randomize_link_inertia"] = EventTermCfg(
    mode="reset",
    func=dr.body_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=GO1_BODY_NAMES),
      "operation": "scale",
      "ranges": LINK_INERTIA_SCALE_RANGE,
    },
  )
  cfg.events["randomize_base_mass"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
      "operation": "scale",
      "ranges": BASE_MASS_SCALE_RANGE,
      "shared_random": True,
      "warn": False,
    },
  )
  cfg.events["randomize_thigh_mass"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=GO1_THIGH_BODY_NAMES),
      "operation": "scale",
      "ranges": THIGH_MASS_SCALE_RANGE,
      "shared_random": True,
      "warn": False,
    },
  )
  cfg.events["randomize_calf_mass"] = EventTermCfg(
    mode="reset",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=GO1_CALF_BODY_NAMES),
      "operation": "scale",
      "ranges": CALF_MASS_SCALE_RANGE,
      "shared_random": True,
      "warn": False,
    },
  )
  cfg.events["randomize_base_com"] = EventTermCfg(
    mode="reset",
    func=dr.body_com_offset,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
      "operation": "add",
      "ranges": BASE_COM_OFFSET_RANGES,
    },
  )
  cfg.events["randomize_leg_com"] = EventTermCfg(
    mode="reset",
    func=dr.body_com_offset,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=GO1_THIGH_BODY_NAMES + GO1_CALF_BODY_NAMES
      ),
      "operation": "add",
      "ranges": LEG_COM_OFFSET_RANGES,
    },
  )

  cfg.events["randomize_joint_damping"] = EventTermCfg(
    mode="reset",
    func=dr.joint_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "scale",
      "ranges": JOINT_DAMPING_SCALE_RANGE,
    },
  )
  cfg.events["randomize_joint_friction"] = EventTermCfg(
    mode="reset",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "scale",
      "ranges": JOINT_FRICTION_SCALE_RANGE,
    },
  )
  cfg.events["randomize_effort_limits"] = EventTermCfg(
    mode="reset",
    func=dr.effort_limits,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "effort_limit_range": EFFORT_SCALE_RANGE,
    },
  )

  for name in ("foot_friction_slide", "foot_friction_spin", "foot_friction_roll"):
    foot_friction_cfg = cfg.events.get(name)
    if foot_friction_cfg is not None:
      foot_friction_cfg.mode = "reset"
  foot_friction_cfg = cfg.events.get("foot_friction_slide")
  if foot_friction_cfg is not None:
    foot_friction_cfg.params["ranges"] = GROUND_FRICTION_RANGE


def _configure_pd_gain_randomization(cfg: ManagerBasedRlEnvCfg) -> None:
  """Add shared reset-time PD gain randomization to a custom Go1 task."""
  cfg.events["randomize_pd_gains"] = EventTermCfg(
    mode="reset",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "abs",
      "kp_range": PD_KP_RANGE,
      "kd_range": PD_KD_RANGE,
      "shared_random": True,
    },
  )


def go1_original_teacher_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the original flat Go1 task with privileged teacher observations."""
  cfg = unitree_go1_flat_env_cfg(play=play)
  _add_torque_reward(cfg)
  _apply_reward_weights(cfg, FLAT_REWARD_WEIGHTS)
  _configure_reset_randomization(cfg)
  _configure_physics_randomization(cfg)
  _configure_pd_gain_randomization(cfg)
  privileged = _single_frame(cfg.observations["critic"])
  cfg.observations["actor"] = deepcopy(privileged)
  cfg.observations["critic"] = privileged
  return cfg


def go1_original_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the flat Go1 task with deployable temporal observations."""
  cfg = unitree_go1_flat_env_cfg(play=play)
  _add_torque_reward(cfg)
  _apply_reward_weights(cfg, FLAT_REWARD_WEIGHTS)
  _configure_reset_randomization(cfg)
  _configure_physics_randomization(cfg)
  _configure_pd_gain_randomization(cfg)
  cfg.observations["actor"] = _student_observation(cfg.observations["actor"])
  cfg.observations["critic"] = _single_frame(cfg.observations["critic"])
  return cfg


def _rough_go1_env_cfg(play: bool) -> ManagerBasedRlEnvCfg:
  """Create the rough Go1 config with memory-safe vectorized defaults."""
  cfg = unitree_go1_rough_env_cfg(play=play)
  # Reserve capacity for the rough map so vectorized runs can initialize
  # reliably. The lower CCD and contact-match limits keep a 4096-environment
  # run within the available GPU memory while retaining rough-terrain contact
  # handling.
  cfg.sim.nconmax = 128
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 128
  if not play:
    cfg.scene.num_envs = 4096
  _add_torque_reward(cfg)
  _apply_reward_weights(cfg, ROUGH_REWARD_WEIGHTS)
  _configure_reset_randomization(cfg)
  _configure_physics_randomization(cfg)
  _configure_pd_gain_randomization(cfg)
  return cfg


def go1_original_rough_teacher_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the original rough Go1 task with privileged teacher observations."""
  cfg = _rough_go1_env_cfg(play=play)
  privileged = _single_frame(cfg.observations["critic"])
  cfg.observations["actor"] = deepcopy(privileged)
  cfg.observations["critic"] = privileged
  return cfg


def go1_original_rough_student_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the rough Go1 task with deployable temporal observations."""
  cfg = _rough_go1_env_cfg(play=play)
  cfg.observations["actor"] = _student_observation(cfg.observations["actor"])
  cfg.observations["critic"] = _single_frame(cfg.observations["critic"])
  return cfg


__all__ = [
  "FLAT_REWARD_WEIGHTS",
  "ROUGH_REWARD_WEIGHTS",
  "STUDENT_HISTORY_LENGTH",
  "STUDENT_HISTORY_TERMS",
  "TORQUE_REWARD_WEIGHT",
  "go1_original_rough_student_env_cfg",
  "go1_original_rough_teacher_env_cfg",
  "go1_original_student_env_cfg",
  "go1_original_teacher_env_cfg",
]
