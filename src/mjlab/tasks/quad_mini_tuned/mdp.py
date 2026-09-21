"""MDP terms specific to the Quad Mini Tuned task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, cast

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
  from mjlab.managers import CurriculumTermCfg


_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")


QuadRandomizationRange = tuple[float, float] | dict[int, tuple[float, float]]
QuadRandomizationParamTargets = dict[
  str, tuple[QuadRandomizationRange, QuadRandomizationRange]
]
QuadRandomizationEventTargets = dict[str, dict[str, QuadRandomizationParamTargets]]
QuadResetRandomizationTargets = dict[
  str, tuple[QuadRandomizationRange, QuadRandomizationRange]
]


class QuadDomainRandomizationStage(TypedDict):
  """One step-based Quad Mini domain-randomization stage."""

  step: int
  light: float
  physical: float


def _interpolate_randomization_range(
  neutral: QuadRandomizationRange,
  target: QuadRandomizationRange,
  strength: float,
) -> QuadRandomizationRange:
  """Interpolate a randomization range from nominal to its full target."""
  if isinstance(neutral, dict):
    if not isinstance(target, dict) or neutral.keys() != target.keys():
      raise ValueError("Randomization range dictionaries must have matching keys.")
    return {
      axis: (
        neutral_range[0] + strength * (target[axis][0] - neutral_range[0]),
        neutral_range[1] + strength * (target[axis][1] - neutral_range[1]),
      )
      for axis, neutral_range in neutral.items()
    }
  if isinstance(target, dict):
    raise ValueError("Randomization range types must match.")
  return (
    neutral[0] + strength * (target[0] - neutral[0]),
    neutral[1] + strength * (target[1] - neutral[1]),
  )


class quad_domain_randomization_curriculum:
  """Apply staged Teacher DR while keeping model parameters fixed between stages."""

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    self._stages: list[QuadDomainRandomizationStage] = cfg.params["stages"]
    self._event_targets: QuadRandomizationEventTargets = cfg.params["event_targets"]
    self._reset_targets: QuadResetRandomizationTargets = cfg.params["reset_targets"]
    self._validate(env)
    # Teacher startup events are configured to apply stage zero after all
    # managers have been built, so the first reset does not need to repeat the
    # expensive MuJoCo model-constant recomputation.
    self._applied_stage_index = 0

  def _validate(self, env: ManagerBasedRlEnv) -> None:
    if not self._stages or self._stages[0] != {
      "step": 0,
      "light": 0.0,
      "physical": 0.0,
    }:
      raise ValueError(
        "Quad Mini DR curriculum must begin at step zero with no randomization."
      )
    for previous, current in zip(self._stages, self._stages[1:], strict=False):
      if current["step"] <= previous["step"]:
        raise ValueError("Quad Mini DR curriculum steps must be strictly increasing.")
    for stage in self._stages:
      for key in ("light", "physical"):
        if not 0.0 <= stage[key] <= 1.0:
          raise ValueError(f"Quad Mini DR strength {key!r} must be within [0, 1].")
    active_startup_terms = set(env.event_manager.active_terms.get("startup", ()))
    configured_terms = {
      event_name
      for group_targets in self._event_targets.values()
      for event_name in group_targets
    }
    missing = configured_terms - active_startup_terms
    if missing:
      raise ValueError(f"Quad Mini DR startup events are missing: {sorted(missing)}")

  def _apply_event_strengths(
    self,
    env: ManagerBasedRlEnv,
    stage: QuadDomainRandomizationStage,
  ) -> None:
    for group_name, group_targets in self._event_targets.items():
      strength = stage[group_name]  # type: ignore[literal-required]
      for event_name, param_targets in group_targets.items():
        event_cfg = env.event_manager.get_term_cfg(event_name)
        for param_name, (neutral, target) in param_targets.items():
          event_cfg.params[param_name] = _interpolate_randomization_range(
            neutral, target, strength
          )

  def _apply_reset_strength(
    self,
    env: ManagerBasedRlEnv,
    light_strength: float,
  ) -> None:
    root_neutral, root_target = self._reset_targets["root_velocity_z"]
    root_cfg = env.event_manager.get_term_cfg("reset_root_state")
    root_cfg.params["velocity_range"]["z"] = _interpolate_randomization_range(
      root_neutral, root_target, light_strength
    )

    joint_neutral, joint_target = self._reset_targets["joint_position"]
    joint_cfg = env.event_manager.get_term_cfg("reset_joint_state")
    joint_cfg.params["position_range"] = _interpolate_randomization_range(
      joint_neutral, joint_target, light_strength
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    stages: list[QuadDomainRandomizationStage],
    event_targets: QuadRandomizationEventTargets,
    reset_targets: QuadResetRandomizationTargets,
  ) -> dict[str, torch.Tensor]:
    del env_ids, stages, event_targets, reset_targets
    stage_index = max(
      index
      for index, stage in enumerate(self._stages)
      if env.common_step_counter >= stage["step"]
    )
    stage = self._stages[stage_index]

    if stage_index != self._applied_stage_index:
      self._apply_event_strengths(env, stage)
      self._apply_reset_strength(env, stage["light"])
      # Resample every model once at the transition. Reusing startup mode keeps
      # physical properties fixed until the next stage instead of recomputing
      # expensive MuJoCo constants on every episode reset.
      env.event_manager.apply(mode="startup")
      self._applied_stage_index = stage_index

    return {
      "stage": torch.tensor(float(stage_index), device=env.device),
      "light": torch.tensor(stage["light"], device=env.device),
      "physical": torch.tensor(stage["physical"], device=env.device),
    }


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


def quad_joint_pos_limits(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  soft_limit_factor: float = 0.95,
) -> torch.Tensor:
  """Penalize joint positions outside the Playground soft limits.

  MuJoCo Playground scales each asymmetric hard-limit endpoint directly. MJLab's
  generic joint-limit term builds a centered soft interval instead, which gives
  different limits for the Quad Mini's asymmetric HFE and KFE ranges.
  """
  asset: Entity = env.scene[asset_cfg.name]
  hard_limits = asset.data.default_joint_pos_limits[:, asset_cfg.joint_ids]
  lower = hard_limits[..., 0] * soft_limit_factor
  upper = hard_limits[..., 1] * soft_limit_factor
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  out_of_limits = -(joint_pos - lower).clip(max=0.0)
  out_of_limits += (joint_pos - upper).clip(min=0.0)
  return torch.sum(out_of_limits, dim=1)


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
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Penalize deviation from the target foot clearance."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  velocity_norm = torch.sqrt(torch.linalg.vector_norm(foot_vel_xy, dim=-1))
  return torch.sum(torch.abs(foot_z - target_height) * velocity_norm, dim=1)


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
  air_time = sensor.data.last_air_time
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
  accelerometer_sensor_name: str = "robot/imu_linacc",
  gyro_sensor_name: str = "robot/imu_angvel",
  gravity_sensor_name: str = "robot/upvector",
) -> torch.Tensor:
  """Build the source task's uncorrupted privileged critic observation."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  gravity = base_mdp.projected_gravity_from_sensor(env, gravity_sensor_name)
  accelerometer = base_mdp.builtin_sensor(env, accelerometer_sensor_name)
  gyro = base_mdp.builtin_sensor(env, gyro_sensor_name)
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
  """Apply a repeatable per-body mass scale or composed torso payload.

  Scaling always starts from the compiled model default, which prevents
  curriculum stage transitions from compounding. Additive payload is applied
  to the current mass so it composes with a preceding whole-robot scale term.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids].to(dtype=env_ids.dtype)
  env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
  model_mass = env.sim.model.body_mass
  default_mass = env.sim.get_default_field("body_mass")
  samples = sample_uniform(
    ranges[0], ranges[1], (len(env_ids), len(body_ids)), env.device
  )
  if operation == "scale":
    model_mass[env_grid, body_grid] = default_mass[body_ids].unsqueeze(0) * samples
  elif operation == "add":
    model_mass[env_grid, body_grid] += samples
  else:
    raise ValueError(f"Unsupported Quad Mini mass operation: {operation!r}")
