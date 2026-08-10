"""Crawler-specific vectorized observations, rewards, and randomization."""

from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
import torch

from mjlab.actuator import DcMotorActuator
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_from_euler_xyz,
  random_orientation,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_ROBOT_CFG = SceneEntityCfg("robot")


def root_height(
  env: "ManagerBasedRlEnv",
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Return root height as a one-column privileged observation."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2:3]


def velocity_command(
  env: "ManagerBasedRlEnv",
  command_name: str,
) -> torch.Tensor:
  """Return commanded heading-frame planar velocity."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command[:, :2]


def full_privileged_state(
  env: "ManagerBasedRlEnv",
  command_name: str,
  contact_sensor_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Return simulator state for the privileged teacher policy.

  This deliberately exposes state that is not available to the deployment policy:
  world-frame poses and velocities, exact joint state, actuator and generalized
  forces, IMU measurements, contact measurements, the command, and the previous
  action. It contains current simulator information only; no future terrain or
  command information is included.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[contact_sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  contact_data = contact_sensor.data
  if contact_data.found is None or contact_data.force is None:
    raise RuntimeError(
      "The privileged teacher requires found and force contact fields."
    )

  def flatten(value: torch.Tensor) -> torch.Tensor:
    return value.reshape(value.shape[0], -1)

  imu_lin_acc = env.scene["robot/imu_lin_acc"]
  imu_ang_vel = env.scene["robot/imu_ang_vel"]
  imu_upvector = env.scene["robot/imu_upvector"]
  if not isinstance(imu_lin_acc, BuiltinSensor):
    raise TypeError("robot/imu_lin_acc must be a built-in sensor")
  if not isinstance(imu_ang_vel, BuiltinSensor):
    raise TypeError("robot/imu_ang_vel must be a built-in sensor")
  if not isinstance(imu_upvector, BuiltinSensor):
    raise TypeError("robot/imu_upvector must be a built-in sensor")

  actuator = asset.actuators[0]
  if not isinstance(actuator, DcMotorActuator) or actuator.force_limit is None:
    raise TypeError("The privileged teacher requires one initialized DC motor group")

  terms = (
    asset.data.root_link_pose_w,
    asset.data.root_link_vel_w,
    asset.data.root_com_pose_w,
    asset.data.root_com_vel_w,
    asset.data.body_link_pose_w,
    asset.data.body_link_vel_w,
    asset.data.body_com_pose_w,
    asset.data.body_com_vel_w,
    asset.data.body_external_wrench,
    asset.data.joint_pos,
    asset.data.joint_pos_biased,
    asset.data.joint_vel,
    asset.data.joint_acc,
    asset.data.actuator_force,
    asset.data.qfrc_actuator,
    asset.data.qfrc_external,
    asset.data.encoder_bias,
    actuator.force_limit,
    imu_lin_acc.data,
    imu_ang_vel.data,
    imu_upvector.data,
    contact_data.found,
    contact_data.force,
    command[:, :2],
    env.action_manager.action,
  )
  return torch.cat([flatten(term) for term in terms], dim=1)


class CrawlerCurriculumStage(TypedDict):
  """One time-based crawler curriculum stage."""

  step: int
  max_command_speed: float
  hard_reset_probability: float
  easy_joint_position_range: tuple[float, float]
  tracking_std: float
  perpendicular_drift_weight: float
  rel_standing_envs: float
  mechanical_power_weight: float


def reset_crawler_state_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  hard_reset_probability: float,
  easy_joint_position_range: tuple[float, float],
  hard_joint_position_range: tuple[float, float],
  easy_roll_pitch_range: tuple[float, float],
  xy_range: tuple[float, float],
  height_range: tuple[float, float],
  planar_velocity_range: tuple[float, float],
  vertical_velocity_range: tuple[float, float],
  roll_pitch_rate_range: tuple[float, float],
  yaw_rate_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> None:
  """Reset root and joints using a shared easy/hard sample for each crawler."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.int64)

  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_state = default_root_state[env_ids].clone()

  position_ranges = torch.tensor(
    (xy_range, xy_range, height_range),
    device=env.device,
  )
  position_offset = sample_uniform(
    position_ranges[:, 0],
    position_ranges[:, 1],
    (len(env_ids), 3),
    env.device,
  )
  position = root_state[:, :3] + position_offset + env.scene.env_origins[env_ids]

  hard_reset = torch.rand(len(env_ids), device=env.device) < hard_reset_probability
  hard_orientation = random_orientation(len(env_ids), env.device)
  easy_roll_pitch = sample_uniform(
    easy_roll_pitch_range[0],
    easy_roll_pitch_range[1],
    (len(env_ids), 2),
    env.device,
  )
  easy_yaw = sample_uniform(
    -torch.pi,
    torch.pi,
    (len(env_ids),),
    env.device,
  )
  easy_orientation = quat_from_euler_xyz(
    easy_roll_pitch[:, 0],
    easy_roll_pitch[:, 1],
    easy_yaw,
  )
  orientation = torch.where(
    hard_reset.unsqueeze(1),
    hard_orientation,
    easy_orientation,
  )

  velocity_ranges = torch.tensor(
    (
      planar_velocity_range,
      planar_velocity_range,
      vertical_velocity_range,
      roll_pitch_rate_range,
      roll_pitch_rate_range,
      yaw_rate_range,
    ),
    device=env.device,
  )
  velocity_offset = sample_uniform(
    velocity_ranges[:, 0],
    velocity_ranges[:, 1],
    (len(env_ids), 6),
    env.device,
  )
  velocity = root_state[:, 7:13] + velocity_offset

  asset.write_root_link_pose_to_sim(
    torch.cat((position, orientation), dim=1),
    env_ids=env_ids,
  )
  asset.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert default_joint_pos is not None
  assert default_joint_vel is not None
  assert soft_joint_pos_limits is not None

  default_selected_joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids]
  joint_shape = default_selected_joint_pos.shape
  easy_joint_offset = sample_uniform(
    *easy_joint_position_range,
    joint_shape,
    env.device,
  )
  hard_joint_offset = sample_uniform(
    *hard_joint_position_range,
    joint_shape,
    env.device,
  )
  joint_offset = torch.where(
    hard_reset.unsqueeze(1),
    hard_joint_offset,
    easy_joint_offset,
  )
  joint_pos = default_selected_joint_pos + joint_offset
  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)
  asset.write_joint_state_to_sim(
    joint_pos,
    joint_vel,
    env_ids=env_ids,
    joint_ids=joint_ids,
  )


def crawler_training_curriculum(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | slice,
  command_name: str,
  reset_event_name: str,
  tracking_reward_name: str,
  drift_reward_name: str,
  power_reward_name: str,
  stages: list[CrawlerCurriculumStage],
) -> dict[str, torch.Tensor]:
  """Apply the latest command, reset, and reward curriculum stage."""
  del env_ids
  active_stage_index = 0
  for index, stage in enumerate(stages):
    if env.common_step_counter >= stage["step"]:
      active_stage_index = index

  stage = stages[active_stage_index]
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  command_cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  speed = stage["max_command_speed"]
  command_cfg.ranges.lin_vel_x = (-speed, speed)
  command_cfg.ranges.lin_vel_y = (-speed, speed)
  command_cfg.rel_standing_envs = stage["rel_standing_envs"]

  reset_cfg = env.event_manager.get_term_cfg(reset_event_name)
  reset_cfg.params["hard_reset_probability"] = stage["hard_reset_probability"]
  reset_cfg.params["easy_joint_position_range"] = stage["easy_joint_position_range"]

  tracking_cfg = env.reward_manager.get_term_cfg(tracking_reward_name)
  tracking_cfg.params["std"] = stage["tracking_std"]
  drift_cfg = env.reward_manager.get_term_cfg(drift_reward_name)
  drift_cfg.weight = stage["perpendicular_drift_weight"]
  power_cfg = env.reward_manager.get_term_cfg(power_reward_name)
  power_cfg.weight = stage["mechanical_power_weight"]

  return {
    "stage": torch.tensor(active_stage_index, device=env.device),
    "max_command_speed": torch.tensor(speed, device=env.device),
    "hard_reset_probability": torch.tensor(
      stage["hard_reset_probability"],
      device=env.device,
    ),
    "tracking_std": torch.tensor(stage["tracking_std"], device=env.device),
    "perpendicular_drift_weight": torch.tensor(
      stage["perpendicular_drift_weight"],
      device=env.device,
    ),
    "standing_environment_fraction": torch.tensor(
      stage["rel_standing_envs"],
      device=env.device,
    ),
    "mechanical_power_weight": torch.tensor(
      stage["mechanical_power_weight"],
      device=env.device,
    ),
  }


def lift_spawn_above_ground(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  clearance: float,
  cylinder_radius: float,
  cylinder_half_length: float,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Lift reset poses until every collision cylinder clears the ground plane."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int64)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.int64)

  asset: Entity = env.scene[asset_cfg.name]

  # Root and joint reset events immediately before this term write qpos. Refresh
  # kinematics once so collision-geom poses represent the newly sampled shape.
  env.sim.forward()
  geom_pose_w = asset.data.geom_pose_w[env_ids][:, asset_cfg.geom_ids]
  geom_quat_w = geom_pose_w[..., 3:7]
  cylinder_axis = torch.zeros_like(geom_pose_w[..., :3])
  cylinder_axis[..., 2] = 1.0
  cylinder_axis_w = quat_apply(geom_quat_w, cylinder_axis)

  axis_z = cylinder_axis_w[..., 2].abs().clamp(max=1.0)
  radial_z = torch.sqrt(torch.clamp(1.0 - torch.square(axis_z), min=0.0))
  half_extent_z = cylinder_half_length * axis_z + cylinder_radius * radial_z
  lowest_point_z = torch.amin(geom_pose_w[..., 2] - half_extent_z, dim=1)
  lift = torch.clamp(clearance - lowest_point_z, min=0.0)

  root_pose_w = asset.data.root_link_pose_w[env_ids].clone()
  root_pose_w[:, 2] += lift
  asset.write_root_link_pose_to_sim(root_pose_w, env_ids=env_ids)


class track_com_progress:
  """Track commanded speed using windowed whole-robot COM progress."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self._env = env
    self._sensor_name = str(cfg.params["sensor_name"])
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._command_name = str(cfg.params["command_name"])
    self._debug_vis_enabled = True
    window_s = float(cfg.params["window_s"])
    self._window_steps = max(1, round(window_s / env.step_dt))
    self._window_s = self._window_steps * env.step_dt
    self._displacement_history_heading = torch.zeros(
      env.num_envs,
      self._window_steps,
      2,
      device=env.device,
    )
    self._rolling_displacement_heading = torch.zeros(
      env.num_envs,
      2,
      device=env.device,
    )
    self._previous_com_xy_w = torch.zeros(
      env.num_envs,
      2,
      device=env.device,
    )
    self._sample_count = torch.zeros(
      env.num_envs,
      dtype=torch.long,
      device=env.device,
    )
    self._initialized = torch.zeros(
      env.num_envs,
      dtype=torch.bool,
      device=env.device,
    )
    self._average_velocity_heading = torch.zeros(
      env.num_envs,
      2,
      device=env.device,
    )
    self._window_ready = torch.zeros(
      env.num_envs,
      dtype=torch.bool,
      device=env.device,
    )
    self._cursor = 0

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    sensor_name: str,
    std: float,
    window_s: float,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
  ) -> torch.Tensor:
    del window_s
    asset: Entity = env.scene[asset_cfg.name]
    com_sensor: BuiltinSensor = env.scene[sensor_name]
    com_xy = com_sensor.data[:, :2]

    new_envs = ~self._initialized
    if torch.any(new_envs):
      self._displacement_history_heading[new_envs] = 0.0
      self._rolling_displacement_heading[new_envs] = 0.0
      self._previous_com_xy_w[new_envs] = com_xy[new_envs]
      self._initialized[new_envs] = True

    step_displacement_xy_w = com_xy - self._previous_com_xy_w
    self._previous_com_xy_w.copy_(com_xy)
    step_displacement_w = torch.nn.functional.pad(
      step_displacement_xy_w,
      (0, 1),
    )
    step_heading_quat_w = yaw_quat(asset.data.root_link_quat_w)
    step_displacement_heading = quat_apply_inverse(
      step_heading_quat_w,
      step_displacement_w,
    )[:, :2]

    expired_displacement = self._displacement_history_heading[:, self._cursor]
    self._rolling_displacement_heading.add_(
      step_displacement_heading - expired_displacement
    )
    self._displacement_history_heading[:, self._cursor] = step_displacement_heading
    self._cursor = (self._cursor + 1) % self._window_steps

    self._sample_count.add_(1).clamp_(max=self._window_steps + 1)
    window_ready = self._sample_count > self._window_steps

    average_velocity = self._rolling_displacement_heading / self._window_s
    self._average_velocity_heading.copy_(average_velocity)
    self._window_ready.copy_(window_ready)

    command = env.command_manager.get_command(command_name)
    assert command is not None
    target_velocity = command[:, :2]
    target_speed = torch.linalg.vector_norm(target_velocity, dim=1)
    command_direction = target_velocity / target_speed.unsqueeze(1).clamp_min(1.0e-6)
    progress_speed = torch.sum(average_velocity * command_direction, dim=1)

    moving = target_speed > 1.0e-6
    speed_error = target_speed - progress_speed
    reward = torch.exp(-torch.square(speed_error) / std**2)
    return reward * moving.float() * window_ready.float()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._initialized[env_ids] = False
    self._sample_count[env_ids] = 0
    self._displacement_history_heading[env_ids] = 0.0
    self._rolling_displacement_heading[env_ids] = 0.0
    self._previous_com_xy_w[env_ids] = 0.0
    self._average_velocity_heading[env_ids] = 0.0
    self._window_ready[env_ids] = False

  def perpendicular_drift_l2(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    velocity_reference: float,
    max_normalized_penalty: float,
  ) -> torch.Tensor:
    """Return bounded normalized COM drift perpendicular to a command."""
    command = env.command_manager.get_command(command_name)
    assert command is not None
    target_velocity = command[:, :2]
    target_speed = torch.linalg.vector_norm(target_velocity, dim=1)
    command_direction = target_velocity / target_speed.unsqueeze(1).clamp_min(1.0e-6)
    perpendicular_direction = torch.stack(
      (-command_direction[:, 1], command_direction[:, 0]),
      dim=1,
    )
    drift_speed = torch.sum(
      self._average_velocity_heading * perpendicular_direction,
      dim=1,
    )
    normalized_drift = torch.abs(drift_speed) / velocity_reference
    penalty = torch.square(normalized_drift).clamp(max=max_normalized_penalty)
    active = (target_speed > 1.0e-6) & self._window_ready
    return penalty * active.float()

  def stop_tracking(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    std: float,
    stop_threshold: float,
  ) -> torch.Tensor:
    """Reward low windowed COM speed during exact stop commands."""
    command = env.command_manager.get_command(command_name)
    assert command is not None
    target_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    com_speed = torch.linalg.vector_norm(self._average_velocity_heading, dim=1)
    reward = torch.exp(-torch.square(com_speed / std))
    standing = (target_speed < stop_threshold) & self._window_ready
    return reward * standing.float()

  def commanded_no_motion(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    command_threshold: float,
  ) -> torch.Tensor:
    """Penalize missing commanded-direction COM progress.

    Exact stop commands are excluded. The penalty uses the same rolling COM
    velocity and heading-frame command as the tracking reward, so sideways or
    reverse motion does not satisfy a nonzero velocity command.
    """
    command = env.command_manager.get_command(command_name)
    assert command is not None
    target_velocity = command[:, :2]
    target_speed = torch.linalg.vector_norm(target_velocity, dim=1)
    command_direction = target_velocity / target_speed.unsqueeze(1).clamp_min(1.0e-6)
    progress_speed = torch.sum(
      self._average_velocity_heading * command_direction,
      dim=1,
    )
    normalized_deficit = (target_speed - progress_speed).clamp_min(
      0.0
    ) / target_speed.clamp_min(1.0e-6)
    active = (target_speed > command_threshold) & self._window_ready
    return torch.square(normalized_deficit) * active.float()

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    """Draw commands and the actual windowed COM velocity used by the reward."""
    if not self._debug_vis_enabled:
      return
    env_indices = list(visualizer.get_env_indices(self._env.num_envs))
    if not env_indices:
      return

    asset: Entity = self._env.scene[self._asset_cfg.name]
    com_sensor: BuiltinSensor = self._env.scene[self._sensor_name]
    command = self._env.command_manager.get_command(self._command_name)
    assert command is not None

    com_positions = com_sensor.data.cpu().numpy()
    headings = asset.data.heading_w.cpu().numpy()
    commands = command.cpu().numpy()
    actual_velocities = self._average_velocity_heading.cpu().numpy()
    ready = self._window_ready.cpu().numpy()

    scale = 0.5
    z_offset = np.array([0.0, 0.0, 0.10])
    for env_index in env_indices:
      heading = headings[env_index]
      cos_h = np.cos(heading)
      sin_h = np.sin(heading)
      heading_rotation = np.array(
        [
          [cos_h, -sin_h],
          [sin_h, cos_h],
        ]
      )
      origin = com_positions[env_index] + z_offset
      command_xy_w = heading_rotation @ commands[env_index, :2]
      actual_xy_w = heading_rotation @ actual_velocities[env_index]

      visualizer.add_arrow(
        start=origin,
        end=origin + np.array([command_xy_w[0], command_xy_w[1], 0.0]) * scale,
        color=(0.2, 0.2, 0.6, 0.8),
        width=0.015,
        label="commanded COM velocity",
      )
      if ready[env_index]:
        visualizer.add_arrow(
          start=origin,
          end=origin + np.array([actual_xy_w[0], actual_xy_w[1], 0.0]) * scale,
          color=(0.0, 0.6, 1.0, 0.9),
          width=0.015,
          label=f"actual {self._window_s:.2f} s COM velocity",
        )


def com_perpendicular_drift_l2(
  env: "ManagerBasedRlEnv",
  tracking_term_name: str,
  command_name: str,
  velocity_reference: float,
  max_normalized_penalty: float,
) -> torch.Tensor:
  """Penalize windowed whole-body COM motion perpendicular to the command."""
  tracking_cfg = env.reward_manager.get_term_cfg(tracking_term_name)
  tracking_term = tracking_cfg.func
  assert isinstance(tracking_term, track_com_progress)
  return tracking_term.perpendicular_drift_l2(
    env,
    command_name,
    velocity_reference,
    max_normalized_penalty,
  )


def stop_com_velocity_tracking(
  env: "ManagerBasedRlEnv",
  tracking_term_name: str,
  command_name: str,
  std: float,
  stop_threshold: float,
) -> torch.Tensor:
  """Reward stopping using the COM velocity from the tracking window."""
  tracking_cfg = env.reward_manager.get_term_cfg(tracking_term_name)
  tracking_term = tracking_cfg.func
  assert isinstance(tracking_term, track_com_progress)
  return tracking_term.stop_tracking(
    env,
    command_name,
    std,
    stop_threshold,
  )


def commanded_no_motion_penalty(
  env: "ManagerBasedRlEnv",
  tracking_term_name: str,
  command_name: str,
  command_threshold: float,
) -> torch.Tensor:
  """Penalize missing COM progress for nonzero velocity commands."""
  tracking_cfg = env.reward_manager.get_term_cfg(tracking_term_name)
  tracking_term = tracking_cfg.func
  assert isinstance(tracking_term, track_com_progress)
  return tracking_term.commanded_no_motion(
    env,
    command_name,
    command_threshold,
  )


def body_ground_impact_velocity_l2(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  free_speed: float,
  normalization_speed: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize only unusually fast downward first contacts."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt).float()
  vertical_velocity = asset.data.body_link_vel_w[:, asset_cfg.body_ids, 2]
  downward_speed = torch.clamp(-vertical_velocity, min=0.0)
  normalized_excess = torch.clamp(
    (downward_speed - free_speed) / normalization_speed,
    min=0.0,
    max=2.0,
  )
  return torch.sum(torch.square(normalized_excess) * first_contact, dim=1)


def _motor_effort_and_limit(asset: Entity) -> tuple[torch.Tensor, torch.Tensor]:
  """Return applied motor effort and its randomized per-environment limit."""
  if len(asset.actuators) != 1 or not isinstance(
    asset.actuators[0],
    DcMotorActuator,
  ):
    raise TypeError("Crawler rewards require exactly one DcMotorActuator group.")
  motor = asset.actuators[0]
  if motor.force_limit is None:
    raise RuntimeError("DC motor force limits are not initialized.")
  return asset.data.actuator_force, motor.force_limit


def actuator_saturation_l2(
  env: "ManagerBasedRlEnv",
  threshold: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Penalize instantaneous motor effort above a normalized threshold."""
  asset: Entity = env.scene[asset_cfg.name]
  effort, effort_limit = _motor_effort_and_limit(asset)
  normalized_effort = torch.abs(effort) / effort_limit.clamp_min(1.0e-6)
  normalized_excess = torch.clamp(
    (normalized_effort - threshold) / (1.0 - threshold),
    min=0.0,
  )
  return torch.mean(torch.square(normalized_excess), dim=1)


class prolonged_actuator_saturation:
  """Penalize motors that remain near their effort limit for too long."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    window_s = float(cfg.params["window_s"])
    self._window_steps = max(1, round(window_s / env.step_dt))
    asset: Entity = env.scene[self._asset_cfg.name]
    effort, _ = _motor_effort_and_limit(asset)
    self._history = torch.zeros(
      env.num_envs,
      self._window_steps,
      effort.shape[1],
      device=env.device,
    )
    self._rolling_count = torch.zeros_like(effort)
    self._sample_count = torch.zeros(
      env.num_envs,
      dtype=torch.long,
      device=env.device,
    )
    self._cursor = 0

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    threshold: float,
    allowed_fraction: float,
    window_s: float,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
  ) -> torch.Tensor:
    del window_s
    asset: Entity = env.scene[asset_cfg.name]
    effort, effort_limit = _motor_effort_and_limit(asset)
    saturated = (torch.abs(effort) / effort_limit.clamp_min(1.0e-6) > threshold).float()
    expired = self._history[:, self._cursor]
    self._rolling_count.add_(saturated - expired)
    self._history[:, self._cursor] = saturated
    self._cursor = (self._cursor + 1) % self._window_steps
    self._sample_count.add_(1).clamp_(max=self._window_steps)

    sample_denominator = self._sample_count.clamp_min(1).unsqueeze(1)
    saturation_fraction = self._rolling_count / sample_denominator
    normalized_excess = torch.clamp(
      (saturation_fraction - allowed_fraction) / (1.0 - allowed_fraction),
      min=0.0,
    )
    return torch.mean(torch.square(normalized_excess), dim=1)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._history[env_ids] = 0.0
    self._rolling_count[env_ids] = 0.0
    self._sample_count[env_ids] = 0


def normalized_joint_mechanical_power(
  env: "ManagerBasedRlEnv",
  reference_joint_speed: float,
  free_budget: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Penalize normalized mechanical power above a free operating budget."""
  asset: Entity = env.scene[asset_cfg.name]
  joint_torque, effort_limit = _motor_effort_and_limit(asset)
  joint_velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
  power = torch.sum(torch.abs(joint_torque * joint_velocity), dim=1)
  power_reference = torch.sum(effort_limit * reference_joint_speed, dim=1)
  normalized_power = power / power_reference.clamp_min(1.0e-6)
  normalized_excess = torch.clamp(
    (normalized_power - free_budget) / (1.0 - free_budget),
    min=0.0,
    max=2.0,
  )
  return torch.square(normalized_excess)
