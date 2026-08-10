"""MJLab environment configuration for planar velocity-command tracking."""

import math
from typing import Literal

from mjlab.asset_zoo.robots.crawler_3dof.robot_cfg import (
  ACTION_LIMIT,
  ACTION_SCALE,
  CRAWLER_CFG,
  DECIMATION,
  JOINT_LIMIT,
  JOINT_NAMES,
  MAX_PLANAR_SPEED,
  PHYSICS_TIMESTEP,
  SEGMENT_HALF_LENGTH_M,
  SEGMENT_RADIUS_M,
  SERVO_NO_LOAD_SPEED_RAD_S_7V4,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.crawler import mdp
from mjlab.tasks.crawler.actions import LowPassJointPositionActionCfg
from mjlab.tasks.crawler.constants import PPO_ROLLOUT_STEPS
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import (
  HfRandomUniformTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)
from mjlab.utils.noise import (
  GaussianNoiseCfg,
  NoiseCfg,
  NoiseModelCfg,
  NoiseModelWithAdditiveBiasCfg,
  UniformNoiseCfg,
)
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

ROBOT_CFG = SceneEntityCfg(
  "robot",
  joint_names=JOINT_NAMES,
  preserve_order=True,
)
BODY_CFG = SceneEntityCfg(
  "robot",
  body_names=("segment_0", "segment_1", "segment_2", "segment_3"),
  preserve_order=True,
)
COLLISION_GEOM_CFG = SceneEntityCfg(
  "robot",
  geom_names=(
    "segment_0_collision",
    "segment_1_collision",
    "segment_2_collision",
    "segment_3_collision",
  ),
  preserve_order=True,
)
IMU_SITE_CFG = SceneEntityCfg("robot", site_names=("imu_site",))

# Central reward tuning. Negative values are penalties. The curriculum stage
# tables below override perpendicular drift and tracking width over time.
REWARD_WEIGHTS = {
  "track_planar_velocity": 5.0,
  "commanded_no_motion": -1.0,
  "stop_tracking": 1.0,
  "perpendicular_drift": -0.50,
  "extreme_body_impact": -0.0001,
  "action_rate": -0.01,
  "action_smoothness": -0.01,
  "torque_saturation": -0.001,
  "prolonged_saturation": -0.001,
  "mechanical_power": 0.0,
  "joint_velocity": -2.0e-6,
  "joint_acceleration": -2.0e-6,
  "joint_limits": -0.20,
}
TRACKING_WINDOW_S = 0.40
STOP_TRACKING_STD = 0.08
STOP_COMMAND_THRESHOLD = 0.02
COMMANDED_MOTION_THRESHOLD = 0.05
PERPENDICULAR_VELOCITY_REFERENCE = 0.20
MAX_NORMALIZED_DRIFT_PENALTY = 4.0
IMPACT_FREE_SPEED = 0.50
IMPACT_NORMALIZATION_SPEED = 1.0
SATURATION_THRESHOLD = 0.90
SATURATION_WINDOW_S = 0.50
SATURATION_ALLOWED_FRACTION = 0.20
POWER_REFERENCE_JOINT_SPEED = SERVO_NO_LOAD_SPEED_RAD_S_7V4
POWER_FREE_BUDGET = 0.40
ACTUATOR_RESPONSE_TIME_S = 0.010
ACTION_TARGET_NEW_WEIGHT = 1.0 - math.exp(
  -(PHYSICS_TIMESTEP * DECIMATION) / ACTUATOR_RESPONSE_TIME_S
)
IMU_DELAY_CONTROL_STEPS = 1
IMU_BIAS_RANGES = {
  "imu_lin_acc": (-0.10, 0.10),
  "imu_ang_vel": (-0.02, 0.02),
}
IMU_NOISE_STD = {
  "imu_lin_acc": 0.03,
  "imu_ang_vel": 0.003,
}

# Full robustness randomization. This is intentionally not used in foundation
# training because applying it from iteration zero previously prevented learning.
FULL_DOMAIN_RANDOMIZATION = {
  "sliding_friction": (0.6, 1.4),
  "torsional_friction_scale": (0.5, 2.0),
  "rolling_friction_scale": (0.5, 3.0),
  "mass_inertia_scale": (0.90, 1.10),
  "com_offset_x": (-0.002, 0.002),
  "com_offset_y": (-0.002, 0.002),
  "com_offset_z": (-0.001, 0.001),
  "joint_damping_scale": (0.75, 1.25),
  "joint_armature_scale": (0.90, 1.10),
  "motor_kp_scale": (0.85, 1.15),
  "motor_kd_scale": (0.75, 1.25),
  "motor_effort_scale": (0.80, 1.05),
  "encoder_bias": (-0.02, 0.02),
  "imu_mount_angle": (-math.radians(1.0), math.radians(1.0)),
}

MILD_DOMAIN_RANDOMIZATION = {
  "sliding_friction": (0.85, 1.25),
  "torsional_friction_scale": (0.75, 1.50),
  "rolling_friction_scale": (0.75, 1.50),
  "mass_inertia_scale": (0.97, 1.03),
  "com_offset_x": (-0.0005, 0.0005),
  "com_offset_y": (-0.0005, 0.0005),
  "com_offset_z": (-0.00025, 0.00025),
  "joint_damping_scale": (0.90, 1.10),
  "joint_armature_scale": (0.95, 1.05),
  "motor_kp_scale": (0.95, 1.05),
  "motor_kd_scale": (0.90, 1.10),
  "motor_effort_scale": (0.90, 1.00),
  "encoder_bias": (-0.005, 0.005),
  "imu_mount_angle": (-math.radians(0.25), math.radians(0.25)),
}

FULL_OBSERVATION_NOISE = {
  "joint_pos": (-0.01, 0.01),
  "joint_vel": (-0.25, 0.25),
  "imu_lin_acc": (-0.20, 0.20),
  "imu_ang_vel": (-0.10, 0.10),
  "gravity_vector": (-0.025, 0.025),
}

MILD_OBSERVATION_NOISE = {
  "joint_pos": (-0.005, 0.005),
  "joint_vel": (-0.125, 0.125),
  "imu_lin_acc": (-0.10, 0.10),
  "imu_ang_vel": (-0.05, 0.05),
  "gravity_vector": (-0.0125, 0.0125),
}

# Backward-compatible aliases for the complete sim-to-real ranges.
DOMAIN_RANDOMIZATION = FULL_DOMAIN_RANDOMIZATION
OBSERVATION_NOISE = FULL_OBSERVATION_NOISE

# Central training reset ranges.
RESET_JOINT_POSITION_RANGE = (-ACTION_LIMIT, ACTION_LIMIT)
RESET_EASY_ROLL_PITCH_RANGE = (-math.radians(10.0), math.radians(10.0))
RESET_ROOT_XY_RANGE = (-0.25, 0.25)
RESET_ROOT_HEIGHT_RANGE = (0.02, 0.10)
RESET_PLANAR_VELOCITY_RANGE = (-0.25, 0.25)
RESET_VERTICAL_VELOCITY_RANGE = (-0.25, 0.40)
RESET_ROLL_PITCH_RATE_RANGE = (-0.50, 0.50)
RESET_YAW_RATE_RANGE = (-0.50, 0.50)
RESET_GROUND_CLEARANCE = 0.002
ROUGH_TERRAIN_RELIEF = 0.040

ROUGH_TERRAIN_GENERATOR = TerrainGeneratorCfg(
  seed=42,
  curriculum=False,
  size=(50.0, 50.0),
  border_width=0.0,
  num_rows=1,
  num_cols=1,
  sub_terrains={
    "random_rough": HfRandomUniformTerrainCfg(
      proportion=1.0,
      noise_range=(-0.020, 0.020),
      noise_step=0.002,
      horizontal_scale=0.20,
      vertical_scale=0.002,
      downsampled_scale=0.40,
      border_width=0.40,
    )
  },
  add_lights=True,
)
FOUNDATION_CURRICULUM_STAGES: list[mdp.CrawlerCurriculumStage] = [
  {
    "step": 0,
    "max_command_speed": 0.30,
    "hard_reset_probability": 0.05,
    "easy_joint_position_range": (-0.35, 0.35),
    "tracking_std": 0.60,
    "perpendicular_drift_weight": -0.02,
    "rel_standing_envs": 0.0,
    "mechanical_power_weight": 0.0,
  },
  {
    "step": 100 * PPO_ROLLOUT_STEPS,
    "max_command_speed": 0.50,
    "hard_reset_probability": 0.15,
    "easy_joint_position_range": (-0.60, 0.60),
    "tracking_std": 0.50,
    "perpendicular_drift_weight": -0.05,
    "rel_standing_envs": 0.0,
    "mechanical_power_weight": 0.0,
  },
  {
    "step": 250 * PPO_ROLLOUT_STEPS,
    "max_command_speed": 0.75,
    "hard_reset_probability": 0.30,
    "easy_joint_position_range": (-1.00, 1.00),
    "tracking_std": 0.40,
    "perpendicular_drift_weight": -0.08,
    "rel_standing_envs": 0.20,
    "mechanical_power_weight": 0.0,
  },
  {
    "step": 450 * PPO_ROLLOUT_STEPS,
    "max_command_speed": 1.00,
    "hard_reset_probability": 0.50,
    "easy_joint_position_range": (-1.20, 1.20),
    "tracking_std": 0.30,
    "perpendicular_drift_weight": -0.10,
    "rel_standing_envs": 0.20,
    "mechanical_power_weight": 0.0,
  },
]

ROBUST_CURRICULUM_STAGES: list[mdp.CrawlerCurriculumStage] = [
  {
    "step": 600 * PPO_ROLLOUT_STEPS,
    "max_command_speed": 1.50,
    "hard_reset_probability": 0.75,
    "easy_joint_position_range": (-ACTION_LIMIT, ACTION_LIMIT),
    "tracking_std": 0.25,
    "perpendicular_drift_weight": -0.10,
    "rel_standing_envs": 0.20,
    "mechanical_power_weight": -0.00000002,
  },
  {
    "step": 800 * PPO_ROLLOUT_STEPS,
    "max_command_speed": 1.50,
    "hard_reset_probability": 1.00,
    "easy_joint_position_range": (-ACTION_LIMIT, ACTION_LIMIT),
    "tracking_std": 0.25,
    "perpendicular_drift_weight": -0.10,
    "rel_standing_envs": 0.20,
    "mechanical_power_weight": -0.00000002,
  },
]

# Continuous flat training uses the foundation and robustness command/reset
# progression, keeps the terrain flat, and uses full randomization from startup.
FLAT_CURRICULUM_STAGES: list[mdp.CrawlerCurriculumStage] = (
  FOUNDATION_CURRICULUM_STAGES + ROBUST_CURRICULUM_STAGES
)
FLAT_CURRICULUM_MAX_ITERATIONS = 1200


def _actor_terms(
  *,
  corrupted: bool,
  noise_ranges: dict[str, tuple[float, float]],
  include_imu: bool = True,
  include_gravity: bool = True,
  realistic_imu: bool = False,
) -> dict[str, ObservationTermCfg]:
  """Build hardware-compatible actor observation terms."""

  def noise(name: str) -> NoiseCfg | NoiseModelCfg | None:
    if not corrupted:
      return None
    lower, upper = noise_ranges[name]
    return UniformNoiseCfg(n_min=lower, n_max=upper)

  def imu_noise(name: str) -> NoiseCfg | NoiseModelCfg | None:
    if not realistic_imu:
      return noise(name)
    bias_lower, bias_upper = IMU_BIAS_RANGES[name]
    return NoiseModelWithAdditiveBiasCfg(
      noise_cfg=GaussianNoiseCfg(std=IMU_NOISE_STD[name]),
      bias_noise_cfg=UniformNoiseCfg(n_min=bias_lower, n_max=bias_upper),
      sample_bias_per_component=True,
    )

  imu_delay = IMU_DELAY_CONTROL_STEPS if realistic_imu else 0

  terms = {
    "joint_pos": ObservationTermCfg(
      func=base_mdp.joint_pos_rel,
      params={"asset_cfg": ROBOT_CFG, "biased": corrupted},
      noise=noise("joint_pos"),
      clip=(-JOINT_LIMIT, JOINT_LIMIT),
      scale=1.0,
    ),
    "joint_vel": ObservationTermCfg(
      func=base_mdp.joint_vel_rel,
      params={"asset_cfg": ROBOT_CFG},
      noise=noise("joint_vel"),
      clip=(-20.0, 20.0),
      scale=0.1,
    ),
    "imu_lin_acc": ObservationTermCfg(
      func=base_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_acc"},
      noise=imu_noise("imu_lin_acc"),
      clip=(-50.0, 50.0),
      scale=0.1,
      delay_min_lag=imu_delay,
      delay_max_lag=imu_delay,
    ),
    "imu_ang_vel": ObservationTermCfg(
      func=base_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=imu_noise("imu_ang_vel"),
      clip=(-20.0, 20.0),
      scale=0.25,
      delay_min_lag=imu_delay,
      delay_max_lag=imu_delay,
    ),
    "gravity_vector": ObservationTermCfg(
      func=base_mdp.projected_gravity_from_sensor,
      params={"sensor_name": "robot/imu_upvector"},
      noise=noise("gravity_vector"),
      clip=(-1.0, 1.0),
    ),
    "last_action": ObservationTermCfg(
      func=base_mdp.last_action,
      clip=(-1.0, 1.0),
      scale=1.0,
    ),
    "velocity_command": ObservationTermCfg(
      func=mdp.velocity_command,
      params={"command_name": "planar_velocity"},
      clip=(-MAX_PLANAR_SPEED, MAX_PLANAR_SPEED),
    ),
  }
  if not include_imu:
    del terms["imu_lin_acc"]
    del terms["imu_ang_vel"]
  if not include_imu or not include_gravity:
    del terms["gravity_vector"]
  return terms


def crawler_env_cfg(
  play: bool = False,
  profile: Literal["foundation", "flat", "robust"] = "foundation",
  include_imu_in_actor: bool = True,
  include_gravity_in_actor: bool = True,
  actor_history_length: int | None = None,
  realistic_imu: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the crawler training or deterministic play configuration.

  The critic retains privileged IMU observations when they are omitted from the
  actor, matching the original critic input while allowing an IMU-free policy.
  Actor history, when requested, is flattened into the policy observation.
  """
  if profile == "foundation":
    stages = FOUNDATION_CURRICULUM_STAGES
    domain_randomization = MILD_DOMAIN_RANDOMIZATION
    observation_noise = MILD_OBSERVATION_NOISE
  elif profile == "flat":
    stages = FLAT_CURRICULUM_STAGES
    domain_randomization = FULL_DOMAIN_RANDOMIZATION
    observation_noise = FULL_OBSERVATION_NOISE
  else:
    stages = ROBUST_CURRICULUM_STAGES
    domain_randomization = FULL_DOMAIN_RANDOMIZATION
    observation_noise = FULL_OBSERVATION_NOISE
  rough_terrain = profile == "robust"

  # Playback evaluates the final difficulty of its selected phase.
  initial_stage = stages[-1] if play else stages[0]
  actor_terms = _actor_terms(
    corrupted=not play,
    noise_ranges=observation_noise,
    include_imu=include_imu_in_actor,
    include_gravity=include_gravity_in_actor,
    realistic_imu=realistic_imu,
  )
  critic_terms = {
    **_actor_terms(corrupted=False, noise_ranges=observation_noise),
    "base_lin_vel": ObservationTermCfg(
      func=base_mdp.base_lin_vel,
      clip=(-5.0, 5.0),
    ),
    "root_height": ObservationTermCfg(
      func=mdp.root_height,
      clip=(-0.5, 1.0),
    ),
  }
  # Rough heightfield contacts can occasionally make a freshly sensed IMU sample
  # non-finite after terminations have already been evaluated. Sanitize that one
  # observation so the existing non_finite_state termination can reset only the
  # affected environment on the next control step. Keep foundation training strict.
  observation_nan_policy = "sanitize" if rough_terrain else "error"
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=not play,
      history_length=actor_history_length,
      nan_policy=observation_nan_policy,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      nan_policy=observation_nan_policy,
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": LowPassJointPositionActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      scale=ACTION_SCALE,
      new_target_weight=ACTION_TARGET_NEW_WEIGHT,
      use_default_offset=True,
      preserve_order=True,
      clip={name: (-ACTION_LIMIT, ACTION_LIMIT) for name in JOINT_NAMES},
    ),
  }

  commands: dict[str, CommandTermCfg] = {
    "planar_velocity": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      linear_velocity_frame="heading",
      linear_velocity_sampling="ellipsoid",
      rel_standing_envs=initial_stage["rel_standing_envs"],
      debug_vis=False,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(
          -initial_stage["max_command_speed"],
          initial_stage["max_command_speed"],
        ),
        lin_vel_y=(
          -initial_stage["max_command_speed"],
          initial_stage["max_command_speed"],
        ),
        ang_vel_z=(0.0, 0.0),
      ),
    ),
  }

  body_ground_contact = ContactSensorCfg(
    name="body_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern="segment_.*",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    track_air_time=True,
  )

  events = {
    "reset_scene_to_default": EventTermCfg(
      func=base_mdp.reset_scene_to_default,
      mode="reset",
    ),
    "reset_crawler_state": EventTermCfg(
      func=mdp.reset_crawler_state_curriculum,
      mode="reset",
      params={
        "hard_reset_probability": initial_stage["hard_reset_probability"],
        "easy_joint_position_range": initial_stage["easy_joint_position_range"],
        "hard_joint_position_range": RESET_JOINT_POSITION_RANGE,
        "easy_roll_pitch_range": RESET_EASY_ROLL_PITCH_RANGE,
        "xy_range": RESET_ROOT_XY_RANGE,
        "height_range": RESET_ROOT_HEIGHT_RANGE,
        "planar_velocity_range": RESET_PLANAR_VELOCITY_RANGE,
        "vertical_velocity_range": RESET_VERTICAL_VELOCITY_RANGE,
        "roll_pitch_rate_range": RESET_ROLL_PITCH_RATE_RANGE,
        "yaw_rate_range": RESET_YAW_RATE_RANGE,
        "asset_cfg": ROBOT_CFG,
      },
    ),
    "lift_spawn_above_ground": EventTermCfg(
      func=mdp.lift_spawn_above_ground,
      mode="reset",
      params={
        "clearance": RESET_GROUND_CLEARANCE
        + (ROUGH_TERRAIN_RELIEF if rough_terrain else 0.0),
        "cylinder_radius": SEGMENT_RADIUS_M,
        "cylinder_half_length": SEGMENT_HALF_LENGTH_M,
        "asset_cfg": COLLISION_GEOM_CFG,
      },
    ),
  }

  if not play:
    mass_scale = domain_randomization["mass_inertia_scale"]
    events.update(
      {
        "randomize_sliding_friction": EventTermCfg(
          mode="startup",
          func=dr.geom_friction,
          params={
            "asset_cfg": COLLISION_GEOM_CFG,
            "operation": "abs",
            "axes": [0],
            "ranges": domain_randomization["sliding_friction"],
            "shared_random": True,
          },
        ),
        "randomize_torsional_friction": EventTermCfg(
          mode="startup",
          func=dr.geom_friction,
          params={
            "asset_cfg": COLLISION_GEOM_CFG,
            "operation": "scale",
            "axes": [1],
            "ranges": domain_randomization["torsional_friction_scale"],
            "shared_random": True,
          },
        ),
        "randomize_rolling_friction": EventTermCfg(
          mode="startup",
          func=dr.geom_friction,
          params={
            "asset_cfg": COLLISION_GEOM_CFG,
            "operation": "scale",
            "axes": [2],
            "ranges": domain_randomization["rolling_friction_scale"],
            "shared_random": True,
          },
        ),
        "randomize_inertial_properties": EventTermCfg(
          mode="startup",
          func=dr.pseudo_inertia,
          params={
            "asset_cfg": BODY_CFG,
            "alpha_range": (
              0.5 * math.log(mass_scale[0]),
              0.5 * math.log(mass_scale[1]),
            ),
            "t1_range": domain_randomization["com_offset_x"],
            "t2_range": domain_randomization["com_offset_y"],
            "t3_range": domain_randomization["com_offset_z"],
          },
        ),
        "randomize_joint_damping": EventTermCfg(
          mode="startup",
          func=dr.joint_damping,
          params={
            "asset_cfg": ROBOT_CFG,
            "operation": "scale",
            "ranges": domain_randomization["joint_damping_scale"],
          },
        ),
        "randomize_joint_armature": EventTermCfg(
          mode="startup",
          func=dr.joint_armature,
          params={
            "asset_cfg": ROBOT_CFG,
            "operation": "scale",
            "ranges": domain_randomization["joint_armature_scale"],
          },
        ),
        "randomize_motor_gains": EventTermCfg(
          mode="startup",
          func=dr.pd_gains,
          params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
            "kp_range": domain_randomization["motor_kp_scale"],
            "kd_range": domain_randomization["motor_kd_scale"],
          },
        ),
        "randomize_motor_effort": EventTermCfg(
          mode="startup",
          func=dr.effort_limits,
          params={
            "asset_cfg": SceneEntityCfg("robot"),
            "operation": "scale",
            "effort_limit_range": domain_randomization["motor_effort_scale"],
          },
        ),
        "randomize_imu_mount": EventTermCfg(
          mode="startup",
          func=dr.site_quat,
          params={
            "asset_cfg": IMU_SITE_CFG,
            "roll_range": domain_randomization["imu_mount_angle"],
            "pitch_range": domain_randomization["imu_mount_angle"],
            "yaw_range": domain_randomization["imu_mount_angle"],
          },
        ),
        "randomize_encoder_bias": EventTermCfg(
          mode="reset",
          func=dr.encoder_bias,
          params={
            "asset_cfg": ROBOT_CFG,
            "bias_range": domain_randomization["encoder_bias"],
          },
        ),
      }
    )
  rewards = {
    "track_planar_velocity": RewardTermCfg(
      func=mdp.track_com_progress,
      weight=REWARD_WEIGHTS["track_planar_velocity"],
      params={
        "command_name": "planar_velocity",
        "sensor_name": "robot/whole_body_com",
        "std": initial_stage["tracking_std"],
        "window_s": TRACKING_WINDOW_S,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "commanded_no_motion": RewardTermCfg(
      func=mdp.commanded_no_motion_penalty,
      weight=REWARD_WEIGHTS["commanded_no_motion"],
      params={
        "tracking_term_name": "track_planar_velocity",
        "command_name": "planar_velocity",
        "command_threshold": COMMANDED_MOTION_THRESHOLD,
      },
    ),
    "perpendicular_drift": RewardTermCfg(
      func=mdp.com_perpendicular_drift_l2,
      weight=initial_stage["perpendicular_drift_weight"],
      params={
        "tracking_term_name": "track_planar_velocity",
        "command_name": "planar_velocity",
        "velocity_reference": PERPENDICULAR_VELOCITY_REFERENCE,
        "max_normalized_penalty": MAX_NORMALIZED_DRIFT_PENALTY,
      },
    ),
    "stop_tracking": RewardTermCfg(
      func=mdp.stop_com_velocity_tracking,
      weight=REWARD_WEIGHTS["stop_tracking"],
      params={
        "tracking_term_name": "track_planar_velocity",
        "command_name": "planar_velocity",
        "std": STOP_TRACKING_STD,
        "stop_threshold": STOP_COMMAND_THRESHOLD,
      },
    ),
    "extreme_body_impact": RewardTermCfg(
      func=mdp.body_ground_impact_velocity_l2,
      weight=REWARD_WEIGHTS["extreme_body_impact"],
      params={
        "sensor_name": body_ground_contact.name,
        "free_speed": IMPACT_FREE_SPEED,
        "normalization_speed": IMPACT_NORMALIZATION_SPEED,
        "asset_cfg": BODY_CFG,
      },
    ),
    "action_rate": RewardTermCfg(
      func=base_mdp.action_rate_l2,
      weight=REWARD_WEIGHTS["action_rate"],
    ),
    "action_smoothness": RewardTermCfg(
      func=base_mdp.action_acc_l2,
      weight=REWARD_WEIGHTS["action_smoothness"],
    ),
    "torque_saturation": RewardTermCfg(
      func=mdp.actuator_saturation_l2,
      weight=REWARD_WEIGHTS["torque_saturation"],
      params={
        "threshold": SATURATION_THRESHOLD,
        "asset_cfg": ROBOT_CFG,
      },
    ),
    "prolonged_saturation": RewardTermCfg(
      func=mdp.prolonged_actuator_saturation,
      weight=REWARD_WEIGHTS["prolonged_saturation"],
      params={
        "threshold": SATURATION_THRESHOLD,
        "allowed_fraction": SATURATION_ALLOWED_FRACTION,
        "window_s": SATURATION_WINDOW_S,
        "asset_cfg": ROBOT_CFG,
      },
    ),
    "mechanical_power": RewardTermCfg(
      func=mdp.normalized_joint_mechanical_power,
      weight=initial_stage["mechanical_power_weight"],
      params={
        "reference_joint_speed": POWER_REFERENCE_JOINT_SPEED,
        "free_budget": POWER_FREE_BUDGET,
        "asset_cfg": ROBOT_CFG,
      },
    ),
    "joint_velocity": RewardTermCfg(
      func=base_mdp.joint_vel_l2,
      weight=REWARD_WEIGHTS["joint_velocity"],
      params={"asset_cfg": ROBOT_CFG},
    ),
    "joint_acceleration": RewardTermCfg(
      func=base_mdp.joint_acc_l2,
      weight=REWARD_WEIGHTS["joint_acceleration"],
      params={"asset_cfg": ROBOT_CFG},
    ),
    "joint_limits": RewardTermCfg(
      func=base_mdp.joint_pos_limits,
      weight=REWARD_WEIGHTS["joint_limits"],
      params={"asset_cfg": ROBOT_CFG},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True),
    "non_finite_state": TerminationTermCfg(func=base_mdp.nan_detection),
  }

  curriculum = {}
  if not play:
    curriculum = {
      "training_stage": CurriculumTermCfg(
        func=mdp.crawler_training_curriculum,
        params={
          "command_name": "planar_velocity",
          "reset_event_name": "reset_crawler_state",
          "tracking_reward_name": "track_planar_velocity",
          "drift_reward_name": "perpendicular_drift",
          "power_reward_name": "mechanical_power",
          "stages": stages,
        },
      )
    }

  terrain_collision = CollisionCfg(
    geom_names_expr=("terrain.*",),
    friction=(1.10, 0.005, 0.0001),
    contype=1,
    conaffinity=1,
    condim=4,
    priority=0,
  )
  if rough_terrain:
    terrain = TerrainEntityCfg(
      terrain_type="generator",
      terrain_generator=ROUGH_TERRAIN_GENERATOR,
      collisions=(terrain_collision,),
    )
  else:
    terrain = TerrainEntityCfg(
      terrain_type="plane",
      env_spacing=1.3,
      collisions=(terrain_collision,),
    )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=terrain,
      entities={"robot": CRAWLER_CFG},
      sensors=(body_ground_contact,),
      num_envs=1 if play else 8192,
      env_spacing=1.3,
      extent=2.5 if rough_terrain else 1.0,
      spec_fn=None,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="segment_0",
      distance=1.2,
      elevation=-25.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=128 if rough_terrain else 32,
      njmax=1024 if rough_terrain else 128,
      mujoco=MujocoCfg(
        timestep=PHYSICS_TIMESTEP,
        gravity=(0.0, 0.0, -9.81),
        integrator="implicitfast",
        solver="newton",
        iterations=50,
        ls_iterations=20,
      ),
    ),
    decimation=DECIMATION,
    episode_length_s=1.0e9 if play else 10.0,
  )
