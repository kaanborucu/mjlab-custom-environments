"""Wind-aware rotor action for the TONY5 Omni V3 task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.manager_based.tony5.tony5_aero_actions import (
  Tony5AeroRotorSpeedAction,
  Tony5AeroRotorSpeedActionCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  DERIVATIVE_FILTER_TIME,
  INTEGRAL_LIMIT_VELOCITY,
  KD_VELOCITY,
  KI_VELOCITY,
  KP_VELOCITY,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_battery import (
  Tony5OmniV3Battery,
  Tony5OmniV3BatteryCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_physics import (
  Tony5OmniV3PhysicsCfg,
  Tony5OmniV3RotorAerodynamics,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_wind import (
  Tony5OmniV3GlobalWind,
  Tony5OmniV3WindCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class Tony5OmniV3RotorSpeedActionCfg(Tony5AeroRotorSpeedActionCfg):
  """V3 action config with a global shared wind process."""

  wind: Tony5OmniV3WindCfg = field(default_factory=Tony5OmniV3WindCfg)
  physics: Tony5OmniV3PhysicsCfg = field(default_factory=Tony5OmniV3PhysicsCfg)
  battery: Tony5OmniV3BatteryCfg = field(default_factory=Tony5OmniV3BatteryCfg)

  def build(self, env: ManagerBasedRlEnv) -> Tony5OmniV3RotorSpeedAction:
    return Tony5OmniV3RotorSpeedAction(self, env)


class Tony5OmniV3RotorSpeedAction(Tony5AeroRotorSpeedAction):
  """Keep V1 control unchanged while updating wind before each physics step."""

  def __init__(
    self,
    cfg: Tony5OmniV3RotorSpeedActionCfg,
    env: ManagerBasedRlEnv,
  ) -> None:
    super().__init__(cfg, env)
    self._battery = Tony5OmniV3Battery(
      cfg.battery,
      env.num_envs,
      len(self._joint_ids),
      str(env.device),
    )
    self._wind = Tony5OmniV3GlobalWind(env, cfg.wind)
    self._aerodynamics = Tony5OmniV3RotorAerodynamics(
      env,
      cfg.entity_name,
      self._wind,
      cfg.physics,
    )

  @property
  def wind(self) -> Tony5OmniV3GlobalWind:
    return self._wind

  @property
  def battery(self) -> Tony5OmniV3Battery:
    return self._battery

  def apply_actions(self) -> None:
    self._battery.update_voltage_limit()
    self._wind.step()
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
    voltage = voltage_without_clamp.clamp(min=0.0)
    voltage_limit = self._battery.loaded_voltage.unsqueeze(-1)
    voltage = torch.minimum(voltage, voltage_limit)

    saturated = voltage != voltage_without_clamp
    integrate = (
      ~saturated
      | ((voltage >= voltage_limit) & (error < 0.0))
      | ((voltage <= 0.0) & (error > 0.0))
    )
    self._integral_error[:] = torch.where(
      integrate, integral_candidate, self._integral_error
    )
    self._entity.set_joint_effort_target(voltage, joint_ids=self._joint_ids)
    self._last_voltage[:] = voltage
    self._previous_omega[:] = omega
    self._pid_initialized[:] = True
    self._battery.record_motor_current(voltage, omega)
    cast(Tony5OmniV3RotorAerodynamics, self._aerodynamics).apply()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    self._wind.reset(env_ids)
    self._battery.reset(env_ids)


__all__ = [
  "Tony5OmniV3RotorSpeedAction",
  "Tony5OmniV3RotorSpeedActionCfg",
]
