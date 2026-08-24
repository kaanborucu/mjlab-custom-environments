"""Optional pack-level voltage-sag support for TONY5 Omni V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.manager_based.tony5.tony5_constants import (
  BATTERY_VOLTAGE,
  MOTOR_KE,
  MOTOR_KT,
  MOTOR_RESISTANCE,
  MOTOR_TORQUE_LIMIT,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

BATTERY_RESISTANCE = 0.02
"""Estimated pack resistance [ohm]; TODO MEASURE on the real battery and wiring."""


@dataclass(frozen=True, kw_only=True)
class Tony5OmniV3BatteryCfg:
  """Configuration for the optional algebraic pack-voltage model."""

  enable_battery_sag: bool = False
  open_circuit_voltage: float = BATTERY_VOLTAGE
  battery_resistance: float = BATTERY_RESISTANCE


class Tony5OmniV3Battery:
  """Estimate positive motor current and expose a safe loaded ESC voltage.

  MuJoCo does not expose winding current as a native tensor in this model. The
  current estimate therefore reconstructs the native voltage/back-EMF/resistance
  relation for diagnostics and the optional pack load, without adding another
  motor torque or electrical state equation.
  """

  def __init__(
    self,
    cfg: Tony5OmniV3BatteryCfg,
    num_envs: int,
    num_rotors: int,
    device: str,
  ) -> None:
    if cfg.open_circuit_voltage <= 0.0:
      raise ValueError("open_circuit_voltage must be positive.")
    if cfg.battery_resistance < 0.0:
      raise ValueError("battery_resistance must be non-negative.")
    self.cfg = cfg
    self._num_envs = num_envs
    self._num_rotors = num_rotors
    self._device = device
    self._motor_current = torch.zeros(
      (num_envs, num_rotors),
      dtype=torch.float,
      device=device,
    )
    self._loaded_voltage = torch.full(
      (num_envs,),
      cfg.open_circuit_voltage,
      dtype=torch.float,
      device=device,
    )

  @property
  def motor_current(self) -> torch.Tensor:
    """Estimated positive current for each motor [A]."""
    return self._motor_current

  @property
  def total_current(self) -> torch.Tensor:
    """Pack current, summing only positive motor currents [A]."""
    return self._motor_current.clamp(min=0.0).sum(dim=-1)

  @property
  def loaded_voltage(self) -> torch.Tensor:
    """Safe per-environment ESC upper voltage limit [V]."""
    return self._loaded_voltage

  @property
  def voltage_sag(self) -> torch.Tensor:
    """Open-circuit minus loaded pack voltage [V]."""
    return torch.clamp(
      self.cfg.open_circuit_voltage - self._loaded_voltage,
      min=0.0,
    )

  @property
  def electrical_power(self) -> torch.Tensor:
    """Pack electrical power estimate [W]."""
    return self.loaded_voltage * self.total_current

  def update_voltage_limit(self) -> None:
    """Update loaded voltage from the previous motor-current estimate."""
    if not self.cfg.enable_battery_sag:
      self._loaded_voltage.fill_(self.cfg.open_circuit_voltage)
      return
    loaded = (
      self.cfg.open_circuit_voltage - self.total_current * self.cfg.battery_resistance
    )
    finite = torch.isfinite(loaded)
    safe_loaded = loaded.clamp(min=0.0, max=self.cfg.open_circuit_voltage)
    self._loaded_voltage[:] = torch.where(finite, safe_loaded, torch.zeros_like(loaded))

  def record_motor_current(
    self,
    voltage: torch.Tensor,
    omega: torch.Tensor,
  ) -> None:
    """Record the native-model current estimate for the next load update."""
    back_emf = MOTOR_KE * omega
    current = (voltage - back_emf) / MOTOR_RESISTANCE
    current_limit = MOTOR_TORQUE_LIMIT / MOTOR_KT
    finite = torch.isfinite(current)
    safe_current = current.clamp(min=0.0, max=current_limit)
    self._motor_current[:] = torch.where(
      finite, safe_current, torch.zeros_like(current)
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._motor_current[env_ids] = 0.0
    self._loaded_voltage[env_ids] = self.cfg.open_circuit_voltage


def _get_battery(env: ManagerBasedRlEnv) -> Tony5OmniV3Battery:
  action = env.action_manager.get_term("rotor_speed")
  battery = getattr(action, "battery", None)
  if not isinstance(battery, Tony5OmniV3Battery):
    raise TypeError("TONY5 Omni V3 battery model is not installed.")
  return battery


def total_current(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_battery(env).total_current


def loaded_voltage(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_battery(env).loaded_voltage


def voltage_sag(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_battery(env).voltage_sag


def electrical_power(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_battery(env).electrical_power


__all__ = [
  "BATTERY_RESISTANCE",
  "Tony5OmniV3Battery",
  "Tony5OmniV3BatteryCfg",
  "electrical_power",
  "loaded_voltage",
  "total_current",
  "voltage_sag",
]
