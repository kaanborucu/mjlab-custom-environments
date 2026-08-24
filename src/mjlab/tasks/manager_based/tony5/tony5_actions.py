"""TONY5 low-level rotor-speed action term."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  BATTERY_VOLTAGE,
  DERIVATIVE_FILTER_TIME,
  INTEGRAL_LIMIT_VELOCITY,
  KD_VELOCITY,
  KI_VELOCITY,
  KP_VELOCITY,
  OMEGA_HOVER,
  OMEGA_MAX,
  ROTOR_JOINTS,
)
from mjlab.tasks.manager_based.tony5.tony5_physics import Tony5RotorAerodynamics

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def map_normalized_action_to_speed(action: torch.Tensor) -> torch.Tensor:
  """Map normalized actions to hover-centered desired rotor speeds."""
  action = action.clamp(-1.0, 1.0)
  lower = OMEGA_HOVER * (1.0 + action)
  upper = OMEGA_HOVER + action * (OMEGA_MAX - OMEGA_HOVER)
  return torch.where(action <= 0.0, lower, upper).clamp(0.0, OMEGA_MAX)


@dataclass(kw_only=True)
class Tony5RotorSpeedActionCfg(ActionTermCfg):
  """Configuration for four normalized individual rotor-speed commands."""

  def build(self, env: ManagerBasedRlEnv) -> Tony5RotorSpeedAction:
    return Tony5RotorSpeedAction(self, env)


class Tony5RotorSpeedAction(ActionTerm):
  """Send desired speeds to DC motor targets and update rotor wrenches."""

  cfg: Tony5RotorSpeedActionCfg

  def __init__(self, cfg: Tony5RotorSpeedActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    joint_ids, joint_names = self._entity.find_joints(ROTOR_JOINTS, preserve_order=True)
    if tuple(joint_names) != ROTOR_JOINTS:
      raise ValueError(f"Unexpected TONY5 action joint order: {joint_names}")
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._action_dim = len(joint_ids)
    self._raw_actions = torch.zeros(
      (self.num_envs, self.action_dim), device=self.device
    )
    self._previous_actions = torch.zeros_like(self._raw_actions)
    self._speed_targets = torch.full_like(self._raw_actions, OMEGA_HOVER)
    self._integral_error = torch.zeros_like(self._raw_actions)
    self._previous_omega = torch.zeros_like(self._raw_actions)
    self._filtered_omega_rate = torch.zeros_like(self._raw_actions)
    self._pid_initialized = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._last_voltage = torch.zeros_like(self._raw_actions)
    self._aerodynamics = Tony5RotorAerodynamics(env, cfg.entity_name)

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def previous_action(self) -> torch.Tensor:
    return self._previous_actions

  @property
  def speed_targets(self) -> torch.Tensor:
    return self._speed_targets

  @property
  def last_voltage(self) -> torch.Tensor:
    """Most recently computed per-rotor PID voltage command."""
    return self._last_voltage

  def process_actions(self, actions: torch.Tensor) -> None:
    self._previous_actions[:] = self._raw_actions
    self._raw_actions[:] = actions.to(self.device).clamp(-1.0, 1.0)
    self._speed_targets[:] = map_normalized_action_to_speed(self._raw_actions)

  def apply_actions(self) -> None:
    # This method is called before every decimation substep. The controller reads
    # actual qvel and computes voltage directly, so thrust and propeller drag use
    # the measured rotor speed rather than the stale policy target.
    omega = self._entity.data.joint_vel[:, self._joint_ids]
    dt = self._env.physics_dt
    first_update = ~self._pid_initialized
    if torch.any(first_update):
      self._previous_omega[first_update] = omega[first_update]
      self._integral_error[first_update] = 0.0

    error = self._speed_targets - omega
    omega_rate = (omega - self._previous_omega) / dt
    filter_alpha = dt / (DERIVATIVE_FILTER_TIME + dt)
    self._filtered_omega_rate += filter_alpha * (omega_rate - self._filtered_omega_rate)
    integral_candidate = (self._integral_error + error * dt).clamp(
      -INTEGRAL_LIMIT_VELOCITY, INTEGRAL_LIMIT_VELOCITY
    )
    voltage_without_clamp = (
      KP_VELOCITY * error
      + KI_VELOCITY * integral_candidate
      - KD_VELOCITY * self._filtered_omega_rate
    )
    voltage = voltage_without_clamp.clamp(-BATTERY_VOLTAGE, BATTERY_VOLTAGE)

    # Conditional anti-windup: do not integrate farther into an active voltage
    # rail, but keep integrating when the error is driving the controller back
    # out of saturation. This avoids a braking kick after a target reversal.
    saturated = voltage != voltage_without_clamp
    integrate = (
      ~saturated
      | ((voltage >= BATTERY_VOLTAGE) & (error < 0.0))
      | ((voltage <= -BATTERY_VOLTAGE) & (error > 0.0))
    )
    self._integral_error[:] = torch.where(
      integrate, integral_candidate, self._integral_error
    )
    self._entity.set_joint_effort_target(voltage, joint_ids=self._joint_ids)
    self._last_voltage[:] = voltage
    self._previous_omega[:] = omega
    self._pid_initialized[:] = True
    self._aerodynamics.apply()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_actions[env_ids] = 0.0
    self._previous_actions[env_ids] = 0.0
    self._speed_targets[env_ids] = OMEGA_HOVER
    self._integral_error[env_ids] = 0.0
    self._previous_omega[env_ids] = 0.0
    self._filtered_omega_rate[env_ids] = 0.0
    self._pid_initialized[env_ids] = False
    self._last_voltage[env_ids] = 0.0


__all__ = [
  "Tony5RotorSpeedAction",
  "Tony5RotorSpeedActionCfg",
  "map_normalized_action_to_speed",
]
