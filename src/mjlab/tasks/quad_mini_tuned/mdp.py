"""MDP terms specific to the Quad Mini Tuned task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as base_mdp
from mjlab.managers import RewardTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


class QuadMiniVelocityCommand(UniformVelocityCommand):
  """Velocity command term using the source task's zeroing/update rule."""

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    cfg = cast(QuadMiniVelocityCommandCfg, self.cfg)
    amplitudes = torch.tensor(cfg.command_amplitudes, device=self.device)
    zero_probabilities = torch.tensor(cfg.zero_probabilities, device=self.device)
    random = torch.rand((len(env_ids), 3), device=self.device)
    samples = (2.0 * random - 1.0) * amplitudes
    keep = (torch.rand((len(env_ids), 3), device=self.device) < 0.5).float()
    nonzero = (
      torch.rand((len(env_ids), 3), device=self.device) >= zero_probabilities
    ).float()
    previous = self.vel_command_b[env_ids, :3]
    self.vel_command_b[env_ids, :3] = previous - keep * (previous - samples * nonzero)

    self.is_standing_env[env_ids] = (
      torch.rand(len(env_ids), device=self.device) <= cfg.rel_standing_envs
    )
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False


@dataclass(kw_only=True)
class QuadMiniVelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for the Quad Mini command sampler."""

  command_amplitudes: tuple[float, float, float] = (1.5, 1.0, 1.5)
  zero_probabilities: tuple[float, float, float] = (0.1, 0.75, 0.5)

  def build(self, env: ManagerBasedRlEnv) -> QuadMiniVelocityCommand:
    return QuadMiniVelocityCommand(self, env)


def imu_acceleration(
  env: ManagerBasedRlEnv,
  accelerometer_sensor_name: str,
  gravity_sensor_name: str,
  gravity_magnitude: float = 9.81,
) -> torch.Tensor:
  """Return accelerometer data with the gravity component removed."""
  accelerometer = base_mdp.builtin_sensor(env, accelerometer_sensor_name)
  gravity = base_mdp.projected_gravity_from_sensor(env, gravity_sensor_name)
  return accelerometer - gravity_magnitude * gravity


def torque_observation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  clip_range: float = 24.0,
) -> torch.Tensor:
  """Return clipped, normalized joint actuator torque."""
  asset: Entity = env.scene[asset_cfg.name]
  torque = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
  return torch.clamp(torque, -clip_range, clip_range) / clip_range


def quad_tracking_linear_velocity(
  env: ManagerBasedRlEnv, command_name: str, sigma: float = 0.25
) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.sum(
    torch.square(command[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-error / sigma)


def quad_tracking_angular_velocity(
  env: ManagerBasedRlEnv, command_name: str, sigma: float = 0.25
) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-error / sigma)


def quad_lin_vel_z(env: ManagerBasedRlEnv) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  return torch.square(asset.data.root_link_lin_vel_w[:, 2])


def quad_ang_vel_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_w[:, :2]), dim=1)


def quad_orientation(env: ManagerBasedRlEnv) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def quad_base_height(
  env: ManagerBasedRlEnv, target_height: float = 0.25
) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)


def quad_pose(env: ManagerBasedRlEnv) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  default = asset.data.default_joint_pos
  assert default is not None
  weights = torch.tensor(
    [1.0, 1.0, 0.1] * 4, device=asset.data.joint_pos.device, dtype=default.dtype
  )
  return torch.exp(
    -torch.sum(torch.square(asset.data.joint_pos - default) * weights, dim=1)
  )


def quad_stand_still(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  default = asset.data.default_joint_pos
  assert default is not None
  return torch.sum(torch.abs(asset.data.joint_pos - default), dim=1) * (
    torch.linalg.vector_norm(command[:, :3], dim=1) < 0.01
  )


def quad_dof_acc(env: ManagerBasedRlEnv) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def quad_dof_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  asset: Entity = env.scene["robot"]
  return torch.sum(torch.square(asset.data.joint_vel), dim=1)


def _moving_command(
  env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.01
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return (torch.linalg.vector_norm(command[:, :3], dim=1) > threshold).float()


def quad_torques_cost(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG
) -> torch.Tensor:
  """Match the source's combined L2 and L1 torque cost."""
  asset: Entity = env.scene[asset_cfg.name]
  torque = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
  return torch.sqrt(torch.sum(torch.square(torque), dim=1)) + torch.sum(
    torch.abs(torque), dim=1
  )


def quad_energy_cost(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG
) -> torch.Tensor:
  """Match the source's absolute mechanical power cost."""
  asset: Entity = env.scene[asset_cfg.name]
  torque = asset.data.qfrc_actuator[:, asset_cfg.joint_ids]
  velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(velocity) * torch.abs(torque), dim=1)


def quad_feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  command_threshold: float = 0.01,
) -> torch.Tensor:
  """Penalize flat-ground foot clearance while the feet are moving."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  velocity_norm = torch.sqrt(torch.linalg.vector_norm(foot_vel_xy, dim=-1))
  cost = torch.sum(torch.abs(foot_z - target_height) * velocity_norm, dim=1)
  return cost * _moving_command(env, command_name, command_threshold)


class QuadFeetHeight:
  """Track the maximum swing height until each foot makes contact."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.peak_height = torch.zeros((env.num_envs, 4), device=env.device)
    self.sensor_name = "feet_ground_contact"

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    asset: Entity = env.scene[asset_cfg.name]
    foot_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    found = sensor.data.found
    assert found is not None
    in_air = found == 0
    self.peak_height = torch.where(
      in_air, torch.maximum(self.peak_height, foot_height), self.peak_height
    )
    first_contact = sensor.compute_first_contact(dt=env.step_dt)
    error = self.peak_height / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1)
    cost *= _moving_command(env, command_name, command_threshold)
    self.peak_height = torch.where(
      first_contact, torch.zeros_like(self.peak_height), self.peak_height
    )
    return cost

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self.peak_height[env_ids if env_ids is not None else slice(None)] = 0.0


def quad_feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
) -> torch.Tensor:
  """Reward the source task's air time at first contact."""
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  assert air_time is not None
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  reward = torch.sum((air_time - 0.1) * first_contact.float(), dim=1)
  return reward * _moving_command(env, command_name, command_threshold)


def quad_impact_feet_velocity(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize downward foot velocity at first contact."""
  sensor: ContactSensor = env.scene[sensor_name]
  asset: Entity = env.scene[asset_cfg.name]
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  downward_velocity = torch.clamp(
    -asset.data.site_lin_vel_w[:, asset_cfg.site_ids, 2], min=0.0
  )
  return torch.sum(downward_velocity * first_contact.float(), dim=1)


def quad_diagonal_trot(
  env: ManagerBasedRlEnv, sensor_name: str, command_name: str
) -> torch.Tensor:
  """Reward alternating RF+LH and LF+RH contact phases."""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  rf, lf, rh, lh = (found > 0).float().unbind(dim=1)
  pair_sync = 1.0 - 0.5 * (torch.abs(rf - lh) + torch.abs(lf - rh))
  pair_opposition = torch.abs(0.5 * (rf + lh) - 0.5 * (lf + rh))
  return pair_sync * pair_opposition * _moving_command(env, command_name, 0.1)


def quad_all_feet_sync(
  env: ManagerBasedRlEnv, sensor_name: str, command_name: str
) -> torch.Tensor:
  """Cost simultaneous all-feet contact or simultaneous flight."""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  contact = (found > 0).float()
  all_contact = torch.prod(contact, dim=1)
  all_air = torch.prod(1.0 - contact, dim=1)
  return (all_contact + all_air) * _moving_command(env, command_name, 0.1)


def quad_three_feet_support(
  env: ManagerBasedRlEnv, sensor_name: str, command_name: str
) -> torch.Tensor:
  """Cost the three-feet-support phase while moving."""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  three_contact = ((found > 0).sum(dim=1) == 3).float()
  return three_contact * _moving_command(env, command_name, 0.1)


def quad_privileged_observation(
  env: ManagerBasedRlEnv,
  command_name: str,
  contact_sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Build the source task's uncorrupted privileged critic observation."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  gravity = base_mdp.projected_gravity_from_sensor(env, "robot/upvector")
  accelerometer = base_mdp.builtin_sensor(env, "robot/imu_linacc")
  gyro = base_mdp.builtin_sensor(env, "robot/imu_angvel")
  joint_pos = asset.data.joint_pos - asset.data.default_joint_pos
  joint_vel = asset.data.joint_vel
  torque = torch.clamp(asset.data.qfrc_actuator, -24.0, 24.0) / 24.0
  last_action = base_mdp.last_action(env)
  state = torch.cat(
    (
      accelerometer - 9.81 * gravity,
      gyro,
      gravity,
      joint_pos,
      joint_vel,
      torque,
      last_action,
      command,
    ),
    dim=1,
  )
  contact_sensor: ContactSensor = env.scene[contact_sensor_name]
  found = contact_sensor.data.found
  air_time = contact_sensor.data.current_air_time
  assert found is not None and air_time is not None
  feet_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :].flatten(1)
  root_force = asset.data.body_external_force[:, :1].flatten(1)
  perturbation_active = torch.zeros(
    (env.num_envs, 1), dtype=state.dtype, device=state.device
  )
  return torch.cat(
    (
      state,
      gyro,
      accelerometer,
      gravity,
      asset.data.root_link_lin_vel_b,
      asset.data.root_link_ang_vel_w,
      joint_pos,
      joint_vel,
      asset.data.actuator_force,
      (found > 0).float(),
      feet_velocity,
      air_time,
      root_force,
      perturbation_active,
    ),
    dim=1,
  )


@requires_model_fields("body_inertia", recompute=RecomputeLevel.set_const)
def quad_body_inertia_scale(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
  """Scale each body's diagonal inertia independently, like the source task."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids].to(dtype=env_ids.dtype)
  default_inertia = env.sim.get_default_field("body_inertia")
  scale = sample_uniform(
    ranges[0],
    ranges[1],
    (len(env_ids), len(body_ids), 1),
    env.device,
  )
  env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
  env.sim.model.body_inertia[env_grid, body_grid] = (
    default_inertia[body_ids].unsqueeze(0) * scale
  )


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def quad_body_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float],
  operation: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
  """Apply the source task's per-body mass scale or torso payload."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids].to(dtype=env_ids.dtype)
  env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
  model_mass = env.sim.model.body_mass
  samples = sample_uniform(
    ranges[0], ranges[1], (len(env_ids), len(body_ids)), env.device
  )
  if operation == "scale":
    model_mass[env_grid, body_grid] *= samples
  elif operation == "add":
    model_mass[env_grid, body_grid] += samples
  else:
    raise ValueError(f"Unsupported Quad Mini mass operation: {operation!r}")


@requires_model_fields("qpos0", recompute=RecomputeLevel.set_const_0)
def quad_default_joint_pose(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> None:
  """Randomize qpos0 and the reset default used by the entity."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  asset: Entity = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    joint_ids = torch.arange(asset.num_joints, device=env.device)
  else:
    joint_ids = torch.tensor(joint_ids, device=env.device)
  qpos_ids = asset.indexing.joint_q_adr[joint_ids].to(dtype=env_ids.dtype)
  default_joint_pos = asset.data.default_joint_pos[0, joint_ids]
  samples = sample_uniform(
    ranges[0], ranges[1], (len(env_ids), len(qpos_ids)), env.device
  )
  env_grid, qpos_grid = torch.meshgrid(env_ids, qpos_ids, indexing="ij")
  randomized = default_joint_pos.unsqueeze(0) + samples
  env.sim.model.qpos0[env_grid, qpos_grid] = randomized
  asset.data.default_joint_pos[env_ids[:, None], joint_ids] = randomized
