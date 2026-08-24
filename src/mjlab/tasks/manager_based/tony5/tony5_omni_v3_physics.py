"""Wind-aware rotor aerodynamics for TONY5 Omni V3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.manager_based.tony5.tony5_aero_physics import (
  BODY_AIR_DENSITY,
  ROTOR_DRAG_KH,
  Tony5AeroRotorAerodynamics,
)
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  KQ_TORQUE,
  KT_THRUST,
  ROTOR_AXIS_SIGNS,
  ROTOR_BODIES,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_prop import (
  Tony5OmniV3PropCfg,
  Tony5OmniV3PropModel,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_wind import (
  Tony5OmniV3GlobalWind,
)
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


K_FLAP = 1.0e-7
"""Estimated blade-flapping coefficient; TODO CALIBRATE from prop data."""


def compute_blade_flapping_moment(
  omega: torch.Tensor,
  rotor_plane_velocity_w: torch.Tensor,
  rotor_axis_w: torch.Tensor,
  k_flap: float,
) -> torch.Tensor:
  """Return a finite, opposing blade-flapping moment for each rotor."""
  moment = (
    -k_flap
    * omega.abs().unsqueeze(-1)
    * torch.cross(rotor_plane_velocity_w, rotor_axis_w, dim=-1)
  )
  finite = torch.isfinite(moment).all(dim=-1, keepdim=True)
  return torch.where(finite, moment, torch.zeros_like(moment))


@dataclass(frozen=True, kw_only=True)
class Tony5OmniV3PhysicsCfg:
  """Independent optional V3 physics switches and provisional parameters."""

  prop: Tony5OmniV3PropCfg = field(default_factory=Tony5OmniV3PropCfg)
  enable_blade_flapping: bool = False
  k_flap: float = K_FLAP


class Tony5OmniV3RotorAerodynamics(Tony5AeroRotorAerodynamics):
  """V3 wind-aware rotor physics with optional prop curves and flapping."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    entity_name: str,
    wind: Tony5OmniV3GlobalWind,
    physics_cfg: Tony5OmniV3PhysicsCfg | None = None,
  ) -> None:
    super().__init__(env, entity_name)
    self._wind = wind
    self._physics_cfg = physics_cfg or Tony5OmniV3PhysicsCfg()
    if self._physics_cfg.k_flap < 0.0:
      raise ValueError("k_flap must be non-negative.")
    self._prop_model = Tony5OmniV3PropModel(
      self._physics_cfg.prop,
      str(env.device),
    )
    self._axis_signs = torch.tensor(
      ROTOR_AXIS_SIGNS,
      dtype=torch.float,
      device=env.device,
    )
    self._wind_body = torch.zeros_like(self._local_air_velocity_b)
    self._rotor_relative_velocity_w = torch.zeros_like(self._local_air_velocity_b)
    self._axial_velocity = torch.zeros(
      (self.num_envs, len(ROTOR_BODIES)),
      device=env.device,
    )
    self._advance_ratio = torch.zeros_like(self._axial_velocity)
    self._ct = torch.zeros_like(self._axial_velocity)
    self._cq = torch.zeros_like(self._axial_velocity)
    self._flapping_moment_w = torch.zeros_like(self._local_air_velocity_b)

  @property
  def wind_body(self) -> torch.Tensor:
    """Canonical wind transformed into the root body frame per rotor."""
    return self._wind_body

  @property
  def rotor_relative_velocity_w(self) -> torch.Tensor:
    """Full wind-aware rotor-relative velocity in world coordinates."""
    return self._rotor_relative_velocity_w

  @property
  def axial_velocity(self) -> torch.Tensor:
    """Rotor-relative velocity projected along the positive thrust axis."""
    return self._axial_velocity

  @property
  def advance_ratio(self) -> torch.Tensor:
    return self._advance_ratio

  @property
  def ct(self) -> torch.Tensor:
    return self._ct

  @property
  def cq(self) -> torch.Tensor:
    return self._cq

  @property
  def flapping_moment_w(self) -> torch.Tensor:
    return self._flapping_moment_w

  def compute_wrenches(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply existing H-force plus optional CT/CQ and blade flapping."""
    forces, torques = super().compute_wrenches()
    no_wind_h_force_w = self._h_force_w.clone()

    root_quat_w = self._asset.data.root_link_quat_w
    root_quat_expanded = root_quat_w.unsqueeze(1).expand(
      -1,
      len(ROTOR_BODIES),
      -1,
    )
    wind_world = self._wind.wind_world.view(1, 1, 3).expand(
      self.num_envs,
      len(ROTOR_BODIES),
      3,
    )
    wind_body = quat_apply_inverse(root_quat_expanded, wind_world)

    global_body_ids = self._asset.data.indexing.body_ids[self._body_ids]
    body_xmat = self._env.sim.data.xmat[:, global_body_ids].reshape(
      self.num_envs,
      len(ROTOR_BODIES),
      3,
      3,
    )
    rotor_axis_w = body_xmat[..., :, 2]
    local_air_velocity_b = self._local_air_velocity_b - wind_body
    rotor_relative_velocity_w = torch.matmul(
      body_xmat,
      local_air_velocity_b.unsqueeze(-1),
    ).squeeze(-1)
    axial_velocity = torch.sum(
      rotor_relative_velocity_w * rotor_axis_w,
      dim=-1,
    )
    rotor_plane_air_velocity_w = (
      rotor_relative_velocity_w - axial_velocity.unsqueeze(-1) * rotor_axis_w
    )
    omega = self._asset.data.joint_vel[:, self._joint_ids]
    h_force_w = -ROTOR_DRAG_KH * omega.abs().unsqueeze(-1) * rotor_plane_air_velocity_w

    advance_ratio, ct, cq = self._prop_model.compute_coefficients(
      omega,
      axial_velocity,
    )
    static_thrust = KT_THRUST * omega.abs().square()
    static_drag_joint = -KQ_TORQUE * omega * omega.abs()
    static_force_w = rotor_axis_w * static_thrust.unsqueeze(-1)
    signed_rotor_axis_w = rotor_axis_w * self._axis_signs.view(1, -1, 1)
    static_torque_w = signed_rotor_axis_w * static_drag_joint.unsqueeze(-1)

    if self._physics_cfg.prop.enable_advanced_prop_model:
      revolutions_per_second = omega.abs() / (2.0 * torch.pi)
      thrust = (
        BODY_AIR_DENSITY
        * revolutions_per_second.square()
        * self._prop_model.diameter**4
        * ct
      )
      torque_magnitude = (
        BODY_AIR_DENSITY
        * revolutions_per_second.square()
        * self._prop_model.diameter**5
        * cq
      )
      drag_joint = -torch.sign(omega) * torque_magnitude
      advanced_force_w = rotor_axis_w * thrust.unsqueeze(-1)
      advanced_torque_w = signed_rotor_axis_w * drag_joint.unsqueeze(-1)
      # ``forces`` already contains the parent V1 static thrust and H-force.
      # Remove only static thrust here; the common replacement below swaps the
      # old H-force for the wind-aware H-force exactly once.
      forces = forces - static_force_w + advanced_force_w
      torques = torques - static_torque_w + advanced_torque_w

    if self._physics_cfg.enable_blade_flapping:
      flapping_moment_w = compute_blade_flapping_moment(
        omega,
        rotor_plane_air_velocity_w,
        rotor_axis_w,
        self._physics_cfg.k_flap,
      )
      torques = torques + flapping_moment_w
    else:
      flapping_moment_w = torch.zeros_like(rotor_plane_air_velocity_w)

    self._wind_body[:] = wind_body
    self._local_air_velocity_b[:] = local_air_velocity_b
    self._rotor_relative_velocity_w[:] = rotor_relative_velocity_w
    self._axial_velocity[:] = axial_velocity
    self._advance_ratio[:] = advance_ratio
    self._ct[:] = ct
    self._cq[:] = cq
    self._rotor_plane_air_velocity_w[:] = rotor_plane_air_velocity_w
    self._h_force_w[:] = h_force_w
    self._flapping_moment_w[:] = flapping_moment_w
    forces = forces - no_wind_h_force_w + h_force_w
    finite_forces = torch.isfinite(forces).all(dim=-1, keepdim=True)
    finite_torques = torch.isfinite(torques).all(dim=-1, keepdim=True)
    return (
      torch.where(finite_forces, forces, torch.zeros_like(forces)),
      torch.where(finite_torques, torques, torch.zeros_like(torques)),
    )


def _get_v3_aero(env: ManagerBasedRlEnv) -> Tony5OmniV3RotorAerodynamics:
  action = env.action_manager.get_term("rotor_speed")
  aero = getattr(action, "_aerodynamics", None)
  if not isinstance(aero, Tony5OmniV3RotorAerodynamics):
    raise TypeError("TONY5 Omni V3 aerodynamics are not installed.")
  return aero


def mean_advance_ratio(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_v3_aero(env).advance_ratio.mean(dim=-1)


def max_abs_advance_ratio(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_v3_aero(env).advance_ratio.abs().amax(dim=-1)


def mean_ct(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_v3_aero(env).ct.mean(dim=-1)


def min_ct(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_v3_aero(env).ct.amin(dim=-1)


def max_ct(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_v3_aero(env).ct.amax(dim=-1)


def mean_cq(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_v3_aero(env).cq.mean(dim=-1)


def min_cq(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_v3_aero(env).cq.amin(dim=-1)


def max_cq(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_v3_aero(env).cq.amax(dim=-1)


def mean_flapping_moment(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_v3_aero(env).flapping_moment_w.norm(dim=-1).mean(dim=-1)


def max_flapping_moment(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _get_v3_aero(env).flapping_moment_w.norm(dim=-1).amax(dim=-1)


__all__ = [
  "K_FLAP",
  "Tony5OmniV3PhysicsCfg",
  "Tony5OmniV3RotorAerodynamics",
  "compute_blade_flapping_moment",
  "max_abs_advance_ratio",
  "max_cq",
  "max_ct",
  "max_flapping_moment",
  "mean_advance_ratio",
  "mean_cq",
  "mean_ct",
  "mean_flapping_moment",
  "min_cq",
  "min_ct",
]
