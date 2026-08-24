"""V1 action wrapper that preserves the TONY5 V0 motor controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.manager_based.tony5.tony5_actions import (
  Tony5RotorSpeedAction,
  Tony5RotorSpeedActionCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_physics import (
  Tony5AeroRotorAerodynamics,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_safety import (
  Tony5NumericalSafetyMonitor,
)
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  BATTERY_VOLTAGE,
  DERIVATIVE_FILTER_TIME,
  INTEGRAL_LIMIT_VELOCITY,
  KD_VELOCITY,
  KI_VELOCITY,
  KP_VELOCITY,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class Tony5AeroRotorSpeedActionCfg(Tony5RotorSpeedActionCfg):
  """V1 rotor action config with the unchanged V0 PID and action mapping."""

  def build(self, env: ManagerBasedRlEnv) -> Tony5AeroRotorSpeedAction:
    return Tony5AeroRotorSpeedAction(self, env)


class Tony5AeroRotorSpeedAction(Tony5RotorSpeedAction):
  """V1 rotor control with a unidirectional ESC voltage output."""

  def __init__(
    self,
    cfg: Tony5AeroRotorSpeedActionCfg,
    env: ManagerBasedRlEnv,
  ) -> None:
    super().__init__(cfg, env)
    self._aerodynamics = Tony5AeroRotorAerodynamics(env, cfg.entity_name)
    self._safety_monitor = Tony5NumericalSafetyMonitor(
      env,
      self._entity,
      self._joint_ids,
    )

  @property
  def safety_monitor(self) -> Tony5NumericalSafetyMonitor:
    return self._safety_monitor

  def apply_actions(self) -> None:
    """Apply the V1 PID with asymmetric voltage saturation at zero volts."""
    self._safety_monitor.check()
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
    voltage = voltage_without_clamp.clamp(0.0, BATTERY_VOLTAGE)

    # Asymmetric anti-windup: allow integration only when it moves an active
    # lower/upper rail back toward the valid voltage interval.
    saturated = voltage != voltage_without_clamp
    integrate = (
      ~saturated
      | ((voltage >= BATTERY_VOLTAGE) & (error < 0.0))
      | ((voltage <= 0.0) & (error > 0.0))
    )
    self._integral_error[:] = torch.where(
      integrate, integral_candidate, self._integral_error
    )
    self._entity.set_joint_effort_target(voltage, joint_ids=self._joint_ids)
    self._last_voltage[:] = voltage
    self._previous_omega[:] = omega
    self._pid_initialized[:] = True
    cast(Tony5AeroRotorAerodynamics, self._aerodynamics).apply()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self._safety_monitor.reset(env_ids)


__all__ = ["Tony5AeroRotorSpeedAction", "Tony5AeroRotorSpeedActionCfg"]
