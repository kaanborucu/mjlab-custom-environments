"""V1-only radial velocity commands and performance curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manager_based.tony5.tony5_aero_rewards import (
  roll_pitch_rate_weight,
  upright_reward_scale,
)
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


TONY5_VELOCITY_STAGE_SPEEDS = (
  2.0,
  6.0,
  10.0,
  27.78,
)
TONY5_VELOCITY_CURRICULUM_SWITCH_ITERATIONS = (
  50,
  100,
  200,
)
# Backward-compatible alias for callers that used the original first switch.
TONY5_VELOCITY_CURRICULUM_SWITCH_ITERATION = (
  TONY5_VELOCITY_CURRICULUM_SWITCH_ITERATIONS[0]
)
TONY5_VELOCITY_CURRICULUM_STEPS_PER_ITERATION = 24
TONY5_VELOCITY_PREVIOUS_SPEED_FRACTION = 0.50
TONY5_VELOCITY_CURRENT_SPEED_FRACTION = 0.30
TONY5_VELOCITY_HIGH_SPEED_FRACTION = 0.20
TONY5_VELOCITY_SAMPLING_HIGH_SPEED_MIN_FRACTION = 0.80
TONY5_VELOCITY_HIGH_SPEED_MIN_FRACTION = 0.70
CURRICULUM_SETTLING_TIME_S = 0.75
TONY5_VELOCITY_STEADY_HIGH_SPEED_MIN_FRACTION = 0.80
TONY5_VELOCITY_HOVER_FRACTION = 0.20
TONY5_VELOCITY_CURRICULUM_WINDOW_STEPS = 512
TONY5_VELOCITY_MAX_YAW_RATE = 1.5
TONY5_VELOCITY_HORIZONTAL_TURN_ACCELERATION = 5.0
TONY5_VELOCITY_HIGH_SPEED_COMMAND_START = 15.0
TONY5_VELOCITY_HIGH_SPEED_YAW_RATE_MAX = 0.10
TONY5_VELOCITY_VERTICAL_COMMAND_MAX = 3.0
TONY5_VELOCITY_HIGH_SPEED_VERTICAL_RATE_MAX = TONY5_VELOCITY_VERTICAL_COMMAND_MAX
TONY5_VELOCITY_FINAL_HORIZONTAL_SPEED = 27.78
TONY5_VELOCITY_KEYBOARD_MAX_SPEED = 100.0
TONY5_VELOCITY_DIRECTION_LIMIT_DEGREES = (180.0, 90.0, 45.0, 25.0, 10.0)
TONY5_VELOCITY_KEYBOARD_SPEED_STEP = 1.0
TONY5_VELOCITY_REWARD_BASE_SIGMA = 2.0
TONY5_VELOCITY_REWARD_SIGMA_PER_SPEED = 0.0


def high_speed_command_fraction(horizontal_speed: torch.Tensor) -> torch.Tensor:
  """Return the command-taper fraction between 15 and 27.78 m/s."""
  return torch.clamp(
    (horizontal_speed - TONY5_VELOCITY_HIGH_SPEED_COMMAND_START)
    / (TONY5_VELOCITY_FINAL_HORIZONTAL_SPEED - TONY5_VELOCITY_HIGH_SPEED_COMMAND_START),
    min=0.0,
    max=1.0,
  )


def yaw_rate_limit(horizontal_speed: torch.Tensor) -> torch.Tensor:
  """Return the speed-dependent yaw-rate limit for moving commands."""
  base_limit = torch.minimum(
    torch.full_like(horizontal_speed, TONY5_VELOCITY_MAX_YAW_RATE),
    TONY5_VELOCITY_HORIZONTAL_TURN_ACCELERATION
    / torch.maximum(horizontal_speed, torch.ones_like(horizontal_speed)),
  )
  taper = high_speed_command_fraction(horizontal_speed)
  return torch.lerp(
    base_limit,
    torch.full_like(horizontal_speed, TONY5_VELOCITY_HIGH_SPEED_YAW_RATE_MAX),
    taper,
  )


def vertical_velocity_command_limit(horizontal_speed: torch.Tensor) -> torch.Tensor:
  """Return the half-range for the V1 vertical command at each speed."""
  base_limit = torch.full_like(
    horizontal_speed,
    TONY5_VELOCITY_VERTICAL_COMMAND_MAX,
  )
  taper = high_speed_command_fraction(horizontal_speed)
  return torch.lerp(
    base_limit,
    torch.full_like(horizontal_speed, TONY5_VELOCITY_HIGH_SPEED_VERTICAL_RATE_MAX),
    taper,
  )


def command_direction_limit(horizontal_speed: torch.Tensor) -> torch.Tensor:
  """Return the maximum absolute direction angle in radians."""
  limit = torch.full_like(
    horizontal_speed,
    TONY5_VELOCITY_DIRECTION_LIMIT_DEGREES[-1] * torch.pi / 180.0,
  )
  limit = torch.where(
    horizontal_speed <= 25.0,
    torch.full_like(
      horizontal_speed,
      TONY5_VELOCITY_DIRECTION_LIMIT_DEGREES[3] * torch.pi / 180.0,
    ),
    limit,
  )
  limit = torch.where(
    horizontal_speed <= 20.0,
    torch.full_like(
      horizontal_speed,
      TONY5_VELOCITY_DIRECTION_LIMIT_DEGREES[2] * torch.pi / 180.0,
    ),
    limit,
  )
  limit = torch.where(
    horizontal_speed <= 15.0,
    torch.full_like(
      horizontal_speed,
      TONY5_VELOCITY_DIRECTION_LIMIT_DEGREES[1] * torch.pi / 180.0,
    ),
    limit,
  )
  return torch.where(
    horizontal_speed <= 10.0,
    torch.full_like(
      horizontal_speed,
      TONY5_VELOCITY_DIRECTION_LIMIT_DEGREES[0] * torch.pi / 180.0,
    ),
    limit,
  )


def sample_horizontal_direction(
  horizontal_speed: torch.Tensor,
  uniform_sample: torch.Tensor,
) -> torch.Tensor:
  """Sample a uniformly distributed direction within the speed limit."""
  return (2.0 * uniform_sample - 1.0) * command_direction_limit(horizontal_speed)


def velocity_reward_sigma(horizontal_speed: torch.Tensor) -> torch.Tensor:
  """Return the V1 velocity-tracking width for each command speed."""
  return (
    TONY5_VELOCITY_REWARD_BASE_SIGMA
    + TONY5_VELOCITY_REWARD_SIGMA_PER_SPEED * horizontal_speed
  )


def track_linear_velocity_adaptive(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track V1 linear velocity with a width that grows with command speed."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual_b = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual_b[:, :2]), dim=1)
  if command.shape[1] > 3:
    z_error = torch.square(command[:, 3] - asset.data.root_link_lin_vel_w[:, 2])
  else:
    z_error = torch.square(actual_b[:, 2])
  linear_velocity_error = xy_error + z_error
  commanded_speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
  sigma_v = velocity_reward_sigma(commanded_speed)
  return torch.exp(-linear_velocity_error / torch.square(sigma_v))


@dataclass(kw_only=True)
class Tony5RadialVelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for V1's radial horizontal-speed curriculum."""

  rel_standing_envs: float = TONY5_VELOCITY_HOVER_FRACTION
  stage_speeds: tuple[float, ...] = TONY5_VELOCITY_STAGE_SPEEDS
  previous_speed_fraction: float = TONY5_VELOCITY_PREVIOUS_SPEED_FRACTION
  current_speed_fraction: float = TONY5_VELOCITY_CURRENT_SPEED_FRACTION
  high_speed_fraction: float = TONY5_VELOCITY_HIGH_SPEED_FRACTION
  sampling_high_speed_min_fraction: float = (
    TONY5_VELOCITY_SAMPLING_HIGH_SPEED_MIN_FRACTION
  )
  high_speed_min_fraction: float = TONY5_VELOCITY_HIGH_SPEED_MIN_FRACTION
  curriculum_window_steps: int = TONY5_VELOCITY_CURRICULUM_WINDOW_STEPS
  curriculum_switch_iterations: tuple[int, ...] = (
    TONY5_VELOCITY_CURRICULUM_SWITCH_ITERATIONS
  )
  curriculum_steps_per_iteration: int = TONY5_VELOCITY_CURRICULUM_STEPS_PER_ITERATION
  play_high_speed_only: bool = False
  play_high_speed_min: float = 15.0
  play_high_speed_max: float = 27.78

  def __post_init__(self) -> None:
    super().__post_init__()
    if len(self.stage_speeds) != len(self.curriculum_switch_iterations) + 1:
      raise ValueError(
        "stage_speeds must contain one more entry than curriculum switches."
      )
    if any(speed <= 0.0 for speed in self.stage_speeds):
      raise ValueError("stage_speeds must contain only positive values.")
    if not 0.0 <= self.previous_speed_fraction <= 1.0:
      raise ValueError("previous_speed_fraction must be in [0, 1].")
    if not 0.0 <= self.current_speed_fraction <= 1.0:
      raise ValueError("current_speed_fraction must be in [0, 1].")
    if not 0.0 <= self.high_speed_fraction <= 1.0:
      raise ValueError("high_speed_fraction must be in [0, 1].")
    branch_fraction_sum = (
      self.previous_speed_fraction
      + self.current_speed_fraction
      + self.high_speed_fraction
    )
    if abs(branch_fraction_sum - 1.0) > 1.0e-6:
      raise ValueError("Moving sampling branch fractions must sum to 1.")
    if not 0.0 < self.sampling_high_speed_min_fraction <= 1.0:
      raise ValueError("sampling_high_speed_min_fraction must be in (0, 1].")
    if not 0.0 < self.high_speed_min_fraction <= 1.0:
      raise ValueError("high_speed_min_fraction must be in (0, 1].")
    if self.curriculum_window_steps <= 0:
      raise ValueError("curriculum_window_steps must be positive.")
    if any(switch <= 0 for switch in self.curriculum_switch_iterations) or any(
      later <= earlier
      for earlier, later in zip(
        self.curriculum_switch_iterations,
        self.curriculum_switch_iterations[1:],
        strict=False,
      )
    ):
      raise ValueError(
        "curriculum_switch_iterations must be strictly increasing and positive."
      )
    if self.curriculum_steps_per_iteration <= 0:
      raise ValueError("curriculum_steps_per_iteration must be positive.")
    if self.play_high_speed_min < 0.0:
      raise ValueError("play_high_speed_min must be non-negative.")
    if self.play_high_speed_max <= self.play_high_speed_min:
      raise ValueError("play_high_speed_max must exceed play_high_speed_min.")
    if self.play_high_speed_max > self.stage_speeds[-1]:
      raise ValueError("play_high_speed_max cannot exceed the final stage speed.")

  def build(self, env: ManagerBasedRlEnv) -> Tony5RadialVelocityCommand:
    return Tony5RadialVelocityCommand(self, env)


class Tony5RadialVelocityCommand(UniformVelocityCommand):
  """Sample symmetric body-frame velocities on a fixed multi-stage schedule."""

  def __init__(self, cfg: Tony5RadialVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._radial_cfg = cfg
    self._stage = len(cfg.stage_speeds) - 1 if cfg.play_high_speed_only else 0
    self._current_vmax = (
      cfg.play_high_speed_max if cfg.play_high_speed_only else cfg.stage_speeds[0]
    )
    window_size = cfg.curriculum_window_steps
    self._window_squared_error = torch.zeros(window_size, device=self.device)
    self._window_attitude_failure = torch.zeros(window_size, device=self.device)
    self._window_commanded_speed = torch.zeros(window_size, device=self.device)
    self._window_achieved_speed = torch.zeros(window_size, device=self.device)
    self._window_high_speed_squared_error = torch.zeros(window_size, device=self.device)
    self._window_high_speed_count = torch.zeros(window_size, device=self.device)
    self._window_high_speed_steady_squared_error = torch.zeros(
      window_size, device=self.device
    )
    self._window_settled_high_speed_count = torch.zeros(window_size, device=self.device)
    self._window_time_since_command = torch.zeros(window_size, device=self.device)
    self._window_yaw_rate_max = torch.zeros(window_size, device=self.device)
    self._window_reward_sigma = torch.zeros(window_size, device=self.device)
    self._window_upright_reward_scale = torch.zeros(window_size, device=self.device)
    self._window_roll_pitch_rate_weight = torch.zeros(window_size, device=self.device)
    self._window_direction_limit_degrees = torch.zeros(window_size, device=self.device)
    self._window_direction_count = torch.zeros(window_size, device=self.device)
    self._window_abs_direction_degrees = torch.zeros(window_size, device=self.device)
    self._window_high_speed_forward_count = torch.zeros(window_size, device=self.device)
    self._window_high_speed_forward_candidate_count = torch.zeros(
      window_size, device=self.device
    )
    self._window_index = 0
    self._window_count = 0
    self._last_curriculum_step = -1
    self._sample_branch = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._yaw_rate_max = torch.zeros(self.num_envs, device=self.device)
    self._time_since_command = torch.zeros(self.num_envs, device=self.device)
    self._reward_sigma = torch.full(
      (self.num_envs,), TONY5_VELOCITY_REWARD_BASE_SIGMA, device=self.device
    )
    self._keyboard_enabled = False
    self._keyboard_max_speed = min(
      self._current_vmax, TONY5_VELOCITY_KEYBOARD_MAX_SPEED
    )
    self._keyboard_mode = "hover"
    self._keyboard_command = torch.zeros(4, device=self.device)

    self._logged_stage = torch.full(
      (self.num_envs,), float(self._stage), device=self.device
    )
    self._logged_vmax = torch.full(
      (self.num_envs,), self._current_vmax, device=self.device
    )
    self._logged_rmse = torch.zeros(self.num_envs, device=self.device)
    self._logged_high_speed_rmse = torch.zeros(self.num_envs, device=self.device)
    self._logged_high_speed_sample_fraction = torch.zeros(
      self.num_envs, device=self.device
    )
    self._logged_high_speed_steady_rmse = torch.zeros(self.num_envs, device=self.device)
    self._logged_settled_high_speed_sample_fraction = torch.zeros(
      self.num_envs, device=self.device
    )
    self._logged_time_since_command = torch.zeros(self.num_envs, device=self.device)
    self._logged_yaw_rate_max = torch.zeros(self.num_envs, device=self.device)
    self._logged_reward_sigma = torch.full(
      (self.num_envs,), TONY5_VELOCITY_REWARD_BASE_SIGMA, device=self.device
    )
    self._logged_upright_reward_scale = torch.ones(self.num_envs, device=self.device)
    self._logged_roll_pitch_rate_weight = torch.ones(self.num_envs, device=self.device)
    self._logged_direction_limit_degrees = torch.zeros(
      self.num_envs, device=self.device
    )
    self._logged_abs_direction_degrees = torch.zeros(self.num_envs, device=self.device)
    self._logged_high_speed_forward_fraction = torch.zeros(
      self.num_envs, device=self.device
    )
    self._logged_failure_rate = torch.zeros(self.num_envs, device=self.device)
    self._logged_commanded_speed = torch.zeros(self.num_envs, device=self.device)
    self._logged_achieved_speed = torch.zeros(self.num_envs, device=self.device)
    self._logged_low_speed_fraction = torch.full(
      (self.num_envs,),
      0.0 if cfg.play_high_speed_only else cfg.previous_speed_fraction,
      device=self.device,
    )
    self._logged_general_speed_fraction = torch.full(
      (self.num_envs,),
      0.0 if cfg.play_high_speed_only else cfg.current_speed_fraction,
      device=self.device,
    )
    self._logged_high_speed_fraction = torch.full(
      (self.num_envs,),
      1.0 if cfg.play_high_speed_only else cfg.high_speed_fraction,
      device=self.device,
    )

  @property
  def curriculum_stage(self) -> int:
    return self._stage

  @property
  def current_vmax(self) -> float:
    return self._current_vmax

  @property
  def curriculum_stage_value(self) -> torch.Tensor:
    return self._logged_stage

  @property
  def current_vmax_value(self) -> torch.Tensor:
    return self._logged_vmax

  @property
  def horizontal_velocity_rmse_value(self) -> torch.Tensor:
    return self._logged_rmse

  @property
  def high_speed_horizontal_rmse_value(self) -> torch.Tensor:
    return self._logged_high_speed_rmse

  @property
  def high_speed_sample_fraction_value(self) -> torch.Tensor:
    return self._logged_high_speed_sample_fraction

  @property
  def high_speed_steady_rmse_value(self) -> torch.Tensor:
    return self._logged_high_speed_steady_rmse

  @property
  def settled_high_speed_sample_fraction_value(self) -> torch.Tensor:
    return self._logged_settled_high_speed_sample_fraction

  @property
  def mean_time_since_command_value(self) -> torch.Tensor:
    return self._logged_time_since_command

  @property
  def current_yaw_rate_max_mean_value(self) -> torch.Tensor:
    return self._logged_yaw_rate_max

  @property
  def velocity_reward_sigma_mean_value(self) -> torch.Tensor:
    return self._logged_reward_sigma

  @property
  def upright_reward_scale_mean_value(self) -> torch.Tensor:
    return self._logged_upright_reward_scale

  @property
  def roll_pitch_rate_weight_mean_value(self) -> torch.Tensor:
    return self._logged_roll_pitch_rate_weight

  @property
  def command_direction_limit_deg_mean_value(self) -> torch.Tensor:
    return self._logged_direction_limit_degrees

  @property
  def mean_abs_command_direction_deg_value(self) -> torch.Tensor:
    return self._logged_abs_direction_degrees

  @property
  def high_speed_forward_fraction_value(self) -> torch.Tensor:
    return self._logged_high_speed_forward_fraction

  @property
  def attitude_failure_rate_value(self) -> torch.Tensor:
    return self._logged_failure_rate

  @property
  def mean_commanded_horizontal_speed_value(self) -> torch.Tensor:
    return self._logged_commanded_speed

  @property
  def mean_achieved_horizontal_speed_value(self) -> torch.Tensor:
    return self._logged_achieved_speed

  @property
  def sample_branch(self) -> torch.Tensor:
    """Most recently sampled branch: 0 hover, 1 low, 2 general, 3 high."""
    return self._sample_branch

  @property
  def low_speed_fraction_value(self) -> torch.Tensor:
    return self._logged_low_speed_fraction

  @property
  def general_speed_fraction_value(self) -> torch.Tensor:
    return self._logged_general_speed_fraction

  @property
  def high_speed_fraction_value(self) -> torch.Tensor:
    return self._logged_high_speed_fraction

  @property
  def keyboard_max_speed(self) -> float:
    """Return the current manual command-speed ceiling in m/s."""
    return self._keyboard_max_speed

  @property
  def keyboard_mode(self) -> str:
    """Return the active manual command mode."""
    return self._keyboard_mode

  def enable_keyboard_control(self) -> None:
    """Enable a persistent manual velocity override for play mode."""
    self._keyboard_enabled = True
    self._keyboard_max_speed = min(
      self._current_vmax, TONY5_VELOCITY_KEYBOARD_MAX_SPEED
    )
    self._keyboard_mode = "hover"
    self._set_keyboard_command()

  def enable_gamepad_control(self) -> None:
    """Enable a persistent analog gamepad velocity override for play mode."""
    self._keyboard_enabled = True
    self._keyboard_max_speed = min(
      self._current_vmax, TONY5_VELOCITY_KEYBOARD_MAX_SPEED
    )
    self._keyboard_mode = "gamepad"
    self._keyboard_command.zero_()
    self._apply_keyboard_command()

  def set_gamepad_command(self, command: tuple[float, float, float, float]) -> None:
    """Set an analog command from ``(x, y, yaw, z)`` gamepad input.

    Horizontal axes use this task's configured linear-velocity frame. Thus
    V3 receives world-frame X/Y input, while body-frame tasks retain body
    semantics. Vertical input is always aligned with world Z.
    """
    if not self._keyboard_enabled:
      return
    requested = torch.tensor(command, device=self.device)
    horizontal_speed = torch.linalg.vector_norm(requested[:2])
    if horizontal_speed > self._keyboard_max_speed:
      requested[:2] *= self._keyboard_max_speed / horizontal_speed
      horizontal_speed = torch.tensor(self._keyboard_max_speed, device=self.device)
    requested[2] = torch.clamp(
      requested[2],
      min=-yaw_rate_limit(horizontal_speed),
      max=yaw_rate_limit(horizontal_speed),
    )
    vertical_limit = vertical_velocity_command_limit(horizontal_speed)
    requested[3] = torch.clamp(requested[3], -vertical_limit, vertical_limit)
    self._time_since_command.zero_()
    self._keyboard_mode = "gamepad"
    self._keyboard_command.copy_(requested)
    self._apply_keyboard_command()

  def adjust_keyboard_max_speed(self, delta: float) -> None:
    """Change the manual horizontal command ceiling by ``delta`` m/s."""
    if not self._keyboard_enabled:
      return
    self._keyboard_max_speed = float(
      torch.clamp(
        torch.tensor(
          self._keyboard_max_speed + delta,
          device=self.device,
        ),
        min=0.0,
        max=TONY5_VELOCITY_KEYBOARD_MAX_SPEED,
      ).item()
    )
    self._set_keyboard_command()

  def resample_now(self) -> None:
    """Immediately resample the velocity command for every environment."""
    env_ids = torch.arange(self.num_envs, device=self.device)
    self._resample(env_ids)

  def set_keyboard_mode(self, mode: str) -> None:
    """Set the persistent manual velocity command mode."""
    if not self._keyboard_enabled:
      return
    valid_modes = {
      "forward",
      "backward",
      "left",
      "right",
      "yaw_left",
      "yaw_right",
      "up",
      "down",
      "hover",
    }
    if mode not in valid_modes:
      raise ValueError(f"Unknown TONY5 keyboard mode: {mode}")
    self._keyboard_mode = mode
    self._set_keyboard_command()

  def _set_keyboard_command(self) -> None:
    """Build and apply a command in the configured linear-velocity frame."""
    self._time_since_command.zero_()
    self._keyboard_command.zero_()
    speed = torch.tensor([self._keyboard_max_speed], device=self.device)
    yaw_limit = yaw_rate_limit(speed)[0]
    vertical_limit = vertical_velocity_command_limit(speed)[0]
    if self._keyboard_mode == "forward":
      self._keyboard_command[0] = self._keyboard_max_speed
    elif self._keyboard_mode == "backward":
      self._keyboard_command[0] = -self._keyboard_max_speed
    elif self._keyboard_mode == "left":
      self._keyboard_command[1] = self._keyboard_max_speed
    elif self._keyboard_mode == "right":
      self._keyboard_command[1] = -self._keyboard_max_speed
    elif self._keyboard_mode == "yaw_left":
      self._keyboard_command[2] = yaw_limit
    elif self._keyboard_mode == "yaw_right":
      self._keyboard_command[2] = -yaw_limit
    elif self._keyboard_mode == "up":
      self._keyboard_command[3] = vertical_limit
    elif self._keyboard_mode == "down":
      self._keyboard_command[3] = -vertical_limit
    self._apply_keyboard_command()

  def _apply_keyboard_command(self, env_ids: torch.Tensor | None = None) -> None:
    """Apply the manual command after normal command-manager updates."""
    if not self._keyboard_enabled:
      return
    target = slice(None) if env_ids is None else env_ids
    self.vel_command_b[target] = self._keyboard_command
    self.vel_command_w[target] = 0.0
    self.vel_command_w[target, :2] = self._keyboard_command[:2]
    self.vel_command_w[target, 2] = self._keyboard_command[3]
    self.is_standing_env[target] = False
    self.is_heading_env[target] = False
    self.is_world_env[target] = False
    commanded_speed = torch.linalg.vector_norm(self._keyboard_command[:2])
    self._yaw_rate_max[target] = yaw_rate_limit(
      torch.full_like(self._yaw_rate_max[target], commanded_speed)
    )
    self._reward_sigma[target] = velocity_reward_sigma(
      torch.full_like(self._reward_sigma[target], commanded_speed)
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    """Reset sampled state and restore the manual command when enabled."""
    extras = super().reset(env_ids)
    if self._keyboard_enabled:
      assert isinstance(env_ids, torch.Tensor)
      self._apply_keyboard_command(env_ids)
    return extras

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Sample radial horizontal velocity, vertical velocity, and yaw rate."""
    self._time_since_command[env_ids] = 0.0
    self.vel_command_b[env_ids] = 0.0
    self.vel_command_w[env_ids] = 0.0
    self.is_heading_env[env_ids] = False
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False
    self._sample_branch[env_ids] = 0
    self._yaw_rate_max[env_ids] = 0.0
    self._reward_sigma[env_ids] = TONY5_VELOCITY_REWARD_BASE_SIGMA

    sample_count = len(env_ids)
    if sample_count == 0:
      return

    random_values = torch.rand((sample_count, 6), device=self.device)
    hover_mask = (
      torch.zeros(sample_count, dtype=torch.bool, device=self.device)
      if self._radial_cfg.play_high_speed_only
      else random_values[:, 0] < self.cfg.rel_standing_envs
    )
    self.is_standing_env[env_ids] = hover_mask
    moving_ids = env_ids[~hover_mask]
    if len(moving_ids) == 0:
      return

    moving_random = random_values[~hover_mask]
    branch_random = moving_random[:, 1]
    if self._radial_cfg.play_high_speed_only:
      previous_speed_mask = torch.zeros_like(branch_random, dtype=torch.bool)
      current_speed_mask = torch.zeros_like(branch_random, dtype=torch.bool)
      high_speed_mask = torch.ones_like(branch_random, dtype=torch.bool)
      branch = torch.full_like(branch_random, 3, dtype=torch.long)
    else:
      previous_speed_mask = branch_random < self._radial_cfg.previous_speed_fraction
      high_speed_mask = branch_random >= (
        self._radial_cfg.previous_speed_fraction
        + self._radial_cfg.current_speed_fraction
      )
      current_speed_mask = ~(previous_speed_mask | high_speed_mask)
      branch = torch.where(
        previous_speed_mask,
        torch.ones_like(branch_random, dtype=torch.long),
        torch.where(
          current_speed_mask,
          torch.full_like(branch_random, 2, dtype=torch.long),
          torch.full_like(branch_random, 3, dtype=torch.long),
        ),
      )
    self._sample_branch[moving_ids] = branch

    if self._radial_cfg.play_high_speed_only:
      speed_lower = torch.full_like(branch_random, self._radial_cfg.play_high_speed_min)
      speed_upper = torch.full_like(branch_random, self._radial_cfg.play_high_speed_max)
    else:
      previous_vmax = (
        self.current_vmax
        if self._stage == 0
        else self._radial_cfg.stage_speeds[self._stage - 1]
      )
      speed_lower = torch.where(
        high_speed_mask,
        torch.full_like(
          branch_random,
          self.current_vmax * self._radial_cfg.sampling_high_speed_min_fraction,
        ),
        torch.zeros_like(branch_random),
      )
      speed_upper = torch.where(
        previous_speed_mask,
        torch.full_like(branch_random, previous_vmax),
        torch.full_like(branch_random, self.current_vmax),
      )
    speed = speed_lower + moving_random[:, 2] * (speed_upper - speed_lower)
    theta = sample_horizontal_direction(speed, moving_random[:, 3])
    self.vel_command_b[moving_ids, 0] = speed * torch.cos(theta)
    self.vel_command_b[moving_ids, 1] = speed * torch.sin(theta)
    yaw_rate_max = yaw_rate_limit(speed)
    self._yaw_rate_max[moving_ids] = yaw_rate_max
    self.vel_command_b[moving_ids, 2] = (2.0 * moving_random[:, 4] - 1.0) * yaw_rate_max
    self._reward_sigma[moving_ids] = velocity_reward_sigma(speed)
    if self.cfg.ranges.lin_vel_z is not None:
      z_low, z_high = self.cfg.ranges.lin_vel_z
      z_center = 0.5 * (z_low + z_high)
      z_base_limit = 0.5 * (z_high - z_low)
      z_limit = vertical_velocity_command_limit(speed)
      z_limit = z_limit * (z_base_limit / TONY5_VELOCITY_VERTICAL_COMMAND_MAX)
      self.vel_command_b[moving_ids, 3] = (
        z_center + (2.0 * moving_random[:, 5] - 1.0) * z_limit
      )

  def compute(
    self,
    dt: float | torch.Tensor,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    if env_ids is None:
      self._time_since_command += dt
    else:
      self._time_since_command[env_ids] += dt
    super().compute(dt, env_ids)
    self._apply_keyboard_command(env_ids)

  def update_curriculum_metrics(self) -> None:
    """Record one control-step sample and apply the fixed schedule switch."""
    step = self._env.common_step_counter
    if step == self._last_curriculum_step:
      return
    self._last_curriculum_step = step

    switch_step = (
      self._radial_cfg.curriculum_switch_iterations[self._stage]
      * self._radial_cfg.curriculum_steps_per_iteration
      if self._stage < len(self._radial_cfg.stage_speeds) - 1
      else None
    )
    if (
      not self._radial_cfg.play_high_speed_only
      and switch_step is not None
      and step >= switch_step
    ):
      self._stage += 1
      self._current_vmax = self._radial_cfg.stage_speeds[self._stage]
      self._window_index = 0
      self._window_count = 0
      self._logged_stage.fill_(float(self._stage))
      self._logged_vmax.fill_(self._current_vmax)

    if self.cfg.linear_velocity_frame == "world":
      actual_velocity = self.robot.data.root_link_lin_vel_w[:, :2]
    else:
      actual_velocity = self.robot.data.root_link_lin_vel_b[:, :2]
    velocity_error = self.vel_command_b[:, :2] - actual_velocity
    index = self._window_index
    self._window_squared_error[index] = torch.mean(
      torch.sum(torch.square(velocity_error), dim=-1)
    )
    # V1 intentionally has no angle/attitude termination. Keep the metric for
    # log compatibility, but report no attitude failures.
    self._window_attitude_failure[index] = 0.0
    self._window_commanded_speed[index] = torch.mean(
      torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=-1)
    )
    self._window_achieved_speed[index] = torch.mean(
      torch.linalg.vector_norm(actual_velocity, dim=-1)
    )
    self._window_yaw_rate_max[index] = self._yaw_rate_max.mean()
    self._window_reward_sigma[index] = self._reward_sigma.mean()
    commanded_speed = torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=-1)
    self._window_time_since_command[index] = self._time_since_command.mean()
    self._window_upright_reward_scale[index] = upright_reward_scale(
      commanded_speed
    ).mean()
    self._window_roll_pitch_rate_weight[index] = roll_pitch_rate_weight(
      commanded_speed
    ).mean()
    moving_mask = commanded_speed > 1.0e-6
    command_theta = torch.atan2(self.vel_command_b[:, 1], self.vel_command_b[:, 0])
    direction_limit_deg = command_direction_limit(commanded_speed) * 180.0 / torch.pi
    self._window_direction_limit_degrees[index] = torch.sum(
      direction_limit_deg[moving_mask]
    )
    self._window_abs_direction_degrees[index] = torch.sum(
      command_theta.abs()[moving_mask] * 180.0 / torch.pi
    )
    self._window_direction_count[index] = torch.count_nonzero(moving_mask)
    high_speed_mask = commanded_speed >= (
      self.current_vmax * self._radial_cfg.high_speed_min_fraction
    )
    high_speed_forward_mask = (commanded_speed >= 20.0) & (
      command_theta.abs() <= 25.0 * torch.pi / 180.0
    )
    self._window_high_speed_forward_count[index] = torch.count_nonzero(
      high_speed_forward_mask
    )
    self._window_high_speed_forward_candidate_count[index] = torch.count_nonzero(
      commanded_speed >= 20.0
    )
    squared_error = torch.sum(torch.square(velocity_error), dim=-1)
    self._window_high_speed_squared_error[index] = torch.sum(
      squared_error[high_speed_mask]
    )
    self._window_high_speed_count[index] = torch.count_nonzero(high_speed_mask)
    settled_high_speed_mask = (
      self._time_since_command >= CURRICULUM_SETTLING_TIME_S
    ) & (
      commanded_speed
      >= self.current_vmax * TONY5_VELOCITY_STEADY_HIGH_SPEED_MIN_FRACTION
    )
    self._window_high_speed_steady_squared_error[index] = torch.sum(
      squared_error[settled_high_speed_mask]
    )
    self._window_settled_high_speed_count[index] = torch.count_nonzero(
      settled_high_speed_mask
    )
    self._window_index = (index + 1) % self._radial_cfg.curriculum_window_steps
    self._window_count = min(
      self._window_count + 1, self._radial_cfg.curriculum_window_steps
    )

    self._refresh_logged_values()

  def _refresh_logged_values(self) -> None:
    if self._window_count == 0:
      return
    window_slice = slice(0, self._window_count)
    self._logged_stage.fill_(float(self._stage))
    self._logged_vmax.fill_(self._current_vmax)
    self._logged_rmse.fill_(torch.sqrt(self._window_squared_error[window_slice].mean()))
    high_speed_count = self._window_high_speed_count[window_slice].sum()
    if bool(high_speed_count.item() > 0.0):
      self._logged_high_speed_rmse.fill_(
        torch.sqrt(
          self._window_high_speed_squared_error[window_slice].sum() / high_speed_count
        )
      )
    else:
      self._logged_high_speed_rmse.zero_()
    self._logged_high_speed_sample_fraction.fill_(
      high_speed_count / (self._window_count * self.num_envs)
    )
    settled_high_speed_count = self._window_settled_high_speed_count[window_slice].sum()
    if bool(settled_high_speed_count.item() > 0.0):
      self._logged_high_speed_steady_rmse.fill_(
        torch.sqrt(
          self._window_high_speed_steady_squared_error[window_slice].sum()
          / settled_high_speed_count
        )
      )
    else:
      self._logged_high_speed_steady_rmse.zero_()
    self._logged_settled_high_speed_sample_fraction.fill_(
      settled_high_speed_count / (self._window_count * self.num_envs)
    )
    self._logged_time_since_command.fill_(
      self._window_time_since_command[window_slice].mean()
    )
    self._logged_failure_rate.fill_(self._window_attitude_failure[window_slice].mean())
    self._logged_commanded_speed.fill_(
      self._window_commanded_speed[window_slice].mean()
    )
    self._logged_achieved_speed.fill_(self._window_achieved_speed[window_slice].mean())
    self._logged_yaw_rate_max.fill_(self._window_yaw_rate_max[window_slice].mean())
    self._logged_reward_sigma.fill_(self._window_reward_sigma[window_slice].mean())
    self._logged_upright_reward_scale.fill_(
      self._window_upright_reward_scale[window_slice].mean()
    )
    self._logged_roll_pitch_rate_weight.fill_(
      self._window_roll_pitch_rate_weight[window_slice].mean()
    )
    direction_count = self._window_direction_count[window_slice].sum()
    if bool(direction_count.item() > 0.0):
      self._logged_direction_limit_degrees.fill_(
        self._window_direction_limit_degrees[window_slice].sum() / direction_count
      )
      self._logged_abs_direction_degrees.fill_(
        self._window_abs_direction_degrees[window_slice].sum() / direction_count
      )
    else:
      self._logged_direction_limit_degrees.zero_()
      self._logged_abs_direction_degrees.zero_()
    high_speed_forward_candidate_count = (
      self._window_high_speed_forward_candidate_count[window_slice].sum()
    )
    if bool(high_speed_forward_candidate_count.item() > 0.0):
      self._logged_high_speed_forward_fraction.fill_(
        self._window_high_speed_forward_count[window_slice].sum()
        / high_speed_forward_candidate_count
      )
    else:
      self._logged_high_speed_forward_fraction.zero_()


def _radial_command(env: ManagerBasedRlEnv) -> Tony5RadialVelocityCommand:
  command = env.command_manager.get_term("velocity")
  return cast(Tony5RadialVelocityCommand, command)


def _update_and_return(
  env: ManagerBasedRlEnv,
  value: torch.Tensor,
) -> torch.Tensor:
  command = _radial_command(env)
  command.update_curriculum_metrics()
  return value


def radial_curriculum_stage(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  command.update_curriculum_metrics()
  return command.curriculum_stage_value


def radial_current_vmax(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.current_vmax_value)


def radial_horizontal_velocity_rmse(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.horizontal_velocity_rmse_value)


def radial_high_speed_horizontal_rmse(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.high_speed_horizontal_rmse_value)


def radial_high_speed_sample_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.high_speed_sample_fraction_value)


def radial_high_speed_steady_rmse(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.high_speed_steady_rmse_value)


def radial_settled_high_speed_sample_fraction(
  env: ManagerBasedRlEnv,
) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.settled_high_speed_sample_fraction_value)


def radial_mean_time_since_command(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.mean_time_since_command_value)


def radial_current_yaw_rate_max_mean(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.current_yaw_rate_max_mean_value)


def radial_velocity_reward_sigma_mean(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.velocity_reward_sigma_mean_value)


def radial_upright_reward_scale_mean(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.upright_reward_scale_mean_value)


def radial_roll_pitch_rate_weight_mean(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.roll_pitch_rate_weight_mean_value)


def radial_command_direction_limit_deg_mean(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.command_direction_limit_deg_mean_value)


def radial_mean_abs_command_direction_deg(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.mean_abs_command_direction_deg_value)


def radial_high_speed_forward_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.high_speed_forward_fraction_value)


def radial_attitude_failure_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.attitude_failure_rate_value)


def radial_mean_commanded_horizontal_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.mean_commanded_horizontal_speed_value)


def radial_mean_achieved_horizontal_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.mean_achieved_horizontal_speed_value)


def radial_low_speed_branch_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.low_speed_fraction_value)


def radial_general_speed_branch_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.general_speed_fraction_value)


def radial_high_speed_branch_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _radial_command(env)
  return _update_and_return(env, command.high_speed_fraction_value)


def tony5_radial_velocity_metrics() -> dict[str, MetricsTermCfg]:
  """Return V1-only rolling curriculum metrics for episode logging."""
  return {
    "curriculum_stage": MetricsTermCfg(
      func=radial_curriculum_stage,
      reduce="last",
    ),
    "current_vmax": MetricsTermCfg(
      func=radial_current_vmax,
      reduce="last",
    ),
    "horizontal_velocity_rmse": MetricsTermCfg(
      func=radial_horizontal_velocity_rmse,
      reduce="last",
    ),
    "high_speed_steady_rmse": MetricsTermCfg(
      func=radial_high_speed_steady_rmse,
      reduce="last",
    ),
    "mean_commanded_horizontal_speed": MetricsTermCfg(
      func=radial_mean_commanded_horizontal_speed,
      reduce="last",
    ),
    "mean_achieved_horizontal_speed": MetricsTermCfg(
      func=radial_mean_achieved_horizontal_speed,
      reduce="last",
    ),
  }


__all__ = [
  "CURRICULUM_SETTLING_TIME_S",
  "TONY5_VELOCITY_CURRICULUM_WINDOW_STEPS",
  "TONY5_VELOCITY_CURRICULUM_STEPS_PER_ITERATION",
  "TONY5_VELOCITY_CURRICULUM_SWITCH_ITERATION",
  "TONY5_VELOCITY_CURRICULUM_SWITCH_ITERATIONS",
  "TONY5_VELOCITY_HIGH_SPEED_FRACTION",
  "TONY5_VELOCITY_HIGH_SPEED_MIN_FRACTION",
  "TONY5_VELOCITY_HOVER_FRACTION",
  "TONY5_VELOCITY_CURRENT_SPEED_FRACTION",
  "TONY5_VELOCITY_HORIZONTAL_TURN_ACCELERATION",
  "TONY5_VELOCITY_HIGH_SPEED_COMMAND_START",
  "TONY5_VELOCITY_HIGH_SPEED_YAW_RATE_MAX",
  "TONY5_VELOCITY_VERTICAL_COMMAND_MAX",
  "TONY5_VELOCITY_HIGH_SPEED_VERTICAL_RATE_MAX",
  "TONY5_VELOCITY_FINAL_HORIZONTAL_SPEED",
  "TONY5_VELOCITY_KEYBOARD_MAX_SPEED",
  "TONY5_VELOCITY_DIRECTION_LIMIT_DEGREES",
  "TONY5_VELOCITY_MAX_YAW_RATE",
  "TONY5_VELOCITY_PREVIOUS_SPEED_FRACTION",
  "TONY5_VELOCITY_REWARD_BASE_SIGMA",
  "TONY5_VELOCITY_REWARD_SIGMA_PER_SPEED",
  "TONY5_VELOCITY_SAMPLING_HIGH_SPEED_MIN_FRACTION",
  "TONY5_VELOCITY_STEADY_HIGH_SPEED_MIN_FRACTION",
  "TONY5_VELOCITY_STAGE_SPEEDS",
  "Tony5RadialVelocityCommand",
  "Tony5RadialVelocityCommandCfg",
  "command_direction_limit",
  "sample_horizontal_direction",
  "track_linear_velocity_adaptive",
  "tony5_radial_velocity_metrics",
  "velocity_reward_sigma",
  "vertical_velocity_command_limit",
  "yaw_rate_limit",
  "high_speed_command_fraction",
]
