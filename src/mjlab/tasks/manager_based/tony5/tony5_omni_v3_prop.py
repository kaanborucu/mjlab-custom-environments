"""Optional CT(J)/CQ(J) infrastructure for TONY5 Omni V3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.tasks.manager_based.tony5.tony5_aero_physics import (
  BODY_AIR_DENSITY,
)
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  KQ_TORQUE,
  KT_THRUST,
)

PROP_DIAMETER = 0.127
"""5-inch propeller diameter in metres."""

MIN_ROTOR_OMEGA = 50.0
"""Minimum rotor speed used by the safe advance-ratio calculation [rad/s]."""

MAX_ABS_ADVANCE_RATIO = 10.0
"""Safety bound for J before curve lookup; not a physical extrapolation."""


def static_ct0(
  *,
  rho: float = BODY_AIR_DENSITY,
  diameter: float = PROP_DIAMETER,
) -> float:
  """Derive static CT from the existing validated kT model."""
  return KT_THRUST * (2.0 * math.pi) ** 2 / (rho * diameter**4)


def static_cq0(
  *,
  rho: float = BODY_AIR_DENSITY,
  diameter: float = PROP_DIAMETER,
) -> float:
  """Derive static CQ from the existing validated kQ model."""
  return KQ_TORQUE * (2.0 * math.pi) ** 2 / (rho * diameter**5)


def smooth_rotor_gate(omega_abs: torch.Tensor, omega_min: float) -> torch.Tensor:
  """Smoothly fade aerodynamic loads to zero below the minimum rotor speed."""
  ratio = torch.clamp(omega_abs / omega_min, min=0.0, max=1.0)
  return ratio.square() * (3.0 - 2.0 * ratio)


@dataclass(frozen=True, kw_only=True)
class Tony5OmniV3PropCfg:
  """Optional propeller curve configuration.

  Curve arrays are intentionally empty until measured or manufacturer data are
  supplied. Enabling the advanced model without explicit arrays raises an error.
  """

  enable_advanced_prop_model: bool = False
  prop_diameter: float = PROP_DIAMETER
  min_rotor_omega: float = MIN_ROTOR_OMEGA
  max_abs_advance_ratio: float = MAX_ABS_ADVANCE_RATIO
  ct_curve_j: tuple[float, ...] = ()
  ct_curve_values: tuple[float, ...] = ()
  cq_curve_j: tuple[float, ...] = ()
  cq_curve_values: tuple[float, ...] = ()


class Tony5OmniV3PropModel:
  """Vectorized CT(J)/CQ(J) lookup with static-model-compatible coefficients."""

  def __init__(self, cfg: Tony5OmniV3PropCfg, device: str) -> None:
    if cfg.prop_diameter <= 0.0:
      raise ValueError("prop_diameter must be positive.")
    if cfg.min_rotor_omega <= 0.0:
      raise ValueError("min_rotor_omega must be positive.")
    if cfg.max_abs_advance_ratio <= 0.0:
      raise ValueError("max_abs_advance_ratio must be positive.")
    self.cfg = cfg
    self._device = device
    self.rho = BODY_AIR_DENSITY
    self.diameter = cfg.prop_diameter
    self.ct0 = static_ct0(diameter=self.diameter)
    self.cq0 = static_cq0(diameter=self.diameter)
    ct_j, ct_values = self._build_curve(
      cfg.ct_curve_j,
      cfg.ct_curve_values,
      "CT",
    )
    cq_j, cq_values = self._build_curve(
      cfg.cq_curve_j,
      cfg.cq_curve_values,
      "CQ",
    )
    self._ct_j = ct_j.to(device=device) if ct_j is not None else None
    self._ct_values = ct_values.to(device=device) if ct_values is not None else None
    self._cq_j = cq_j.to(device=device) if cq_j is not None else None
    self._cq_values = cq_values.to(device=device) if cq_values is not None else None
    if cfg.enable_advanced_prop_model and (self._ct_j is None or self._cq_j is None):
      raise ValueError(
        "Advanced CT(J)/CQ(J) requires explicit measured or manufacturer curve data."
      )

  @staticmethod
  def _build_curve(
    curve_j: tuple[float, ...],
    curve_values: tuple[float, ...],
    name: str,
  ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not curve_j and not curve_values:
      return None, None
    if len(curve_j) != len(curve_values) or not curve_j:
      raise ValueError(f"{name} curve J and value arrays must have equal nonzero size.")
    if any(not math.isfinite(value) for value in (*curve_j, *curve_values)):
      raise ValueError(f"{name} curve data must be finite.")
    if any(
      right <= left for left, right in zip(curve_j[:-1], curve_j[1:], strict=True)
    ):
      raise ValueError(f"{name} curve J values must be strictly increasing.")
    return (
      torch.tensor(curve_j, dtype=torch.float),
      torch.tensor(curve_values, dtype=torch.float),
    )

  @staticmethod
  def _lookup(
    query: torch.Tensor,
    curve_j: torch.Tensor,
    curve_values: torch.Tensor,
  ) -> torch.Tensor:
    if curve_j.numel() == 1:
      return curve_values[0].expand_as(query)
    query_clamped = query.clamp(curve_j[0], curve_j[-1])
    upper = torch.searchsorted(curve_j, query_clamped).clamp(1, curve_j.numel() - 1)
    lower = upper - 1
    j0 = curve_j[lower]
    j1 = curve_j[upper]
    c0 = curve_values[lower]
    c1 = curve_values[upper]
    fraction = (query_clamped - j0) / (j1 - j0)
    return c0 + fraction * (c1 - c0)

  def lookup_ct(self, advance_ratio: torch.Tensor) -> torch.Tensor:
    """Return CT(J), using only explicitly supplied curve data."""
    if self._ct_j is None or self._ct_values is None:
      return torch.full_like(advance_ratio, self.ct0)
    return self._lookup(
      advance_ratio,
      self._ct_j,
      self._ct_values,
    )

  def lookup_cq(self, advance_ratio: torch.Tensor) -> torch.Tensor:
    """Return CQ(J), using only explicitly supplied curve data."""
    if self._cq_j is None or self._cq_values is None:
      return torch.full_like(advance_ratio, self.cq0)
    return self._lookup(
      advance_ratio,
      self._cq_j,
      self._cq_values,
    )

  def compute_coefficients(
    self,
    omega: torch.Tensor,
    axial_velocity: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return safe J, CT, and CQ tensors for all environments and rotors."""
    omega_abs = omega.abs()
    revolutions_per_second = omega_abs / (2.0 * math.pi)
    n_safe = torch.clamp(
      revolutions_per_second,
      min=self.cfg.min_rotor_omega / (2.0 * math.pi),
    )
    advance_ratio = axial_velocity / (n_safe * self.diameter)
    advance_ratio = advance_ratio.clamp(
      -self.cfg.max_abs_advance_ratio,
      self.cfg.max_abs_advance_ratio,
    )
    ct = self.lookup_ct(advance_ratio)
    cq = self.lookup_cq(advance_ratio)
    gate = smooth_rotor_gate(omega_abs, self.cfg.min_rotor_omega)
    ct = ct * gate
    cq = cq * gate
    finite = torch.isfinite(advance_ratio) & torch.isfinite(ct) & torch.isfinite(cq)
    return (
      torch.where(finite, advance_ratio, torch.zeros_like(advance_ratio)),
      torch.where(finite, ct, torch.zeros_like(ct)),
      torch.where(finite, cq, torch.zeros_like(cq)),
    )


__all__ = [
  "MAX_ABS_ADVANCE_RATIO",
  "MIN_ROTOR_OMEGA",
  "PROP_DIAMETER",
  "Tony5OmniV3PropCfg",
  "Tony5OmniV3PropModel",
  "smooth_rotor_gate",
  "static_cq0",
  "static_ct0",
]
