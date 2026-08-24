"""V1-only TONY5 body-fluid and rotor H-force physics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

import mujoco
import torch

from mjlab.entity import EntityCfg
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  ROTOR_BODIES,
  get_tony5_robot_cfg,
  get_tony5_spec,
)
from mjlab.tasks.manager_based.tony5.tony5_physics import Tony5RotorAerodynamics
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


ROTOR_DRAG_KH = 1.0e-5
"""Estimated rotor H-force coefficient; TODO calibrate from measurements."""

BODY_AIR_DENSITY = 1.225
BODY_AIR_VISCOSITY = 0.0
BODY_AERO_SIZE = (0.065, 0.050, 0.022)
BODY_AERO_POS = (0.0, 0.0, 0.015)
BODY_AERO_FLUID_COEFS = (0.5, 0.25, 1.0, 0.0, 0.0)
ROTOR_FLUID_DISABLE_SIZE = (1.0e-5, 1.0e-5, 1.0e-5)


def get_tony5_aero_spec() -> mujoco.MjSpec:
  """Build the V1 robot spec with MuJoCo's ellipsoid fluid geom."""
  spec = get_tony5_spec()
  spec.add_sensor(
    name="imu_angacc",
    type=mujoco.mjtSensor.mjSENS_FRAMEANGACC,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu_site",
  )
  body = spec.body("quad_base")
  body_aero = cast(Any, body.add_geom(name="body_aero"))
  body_aero.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
  body_aero.size = BODY_AERO_SIZE
  body_aero.pos = BODY_AERO_POS
  body_aero.contype = 0
  body_aero.conaffinity = 0
  body_aero.density = 0.0
  body_aero.fluid_ellipsoid = 1.0
  body_aero.fluid_coefs = BODY_AERO_FLUID_COEFS
  body_aero.rgba = (0.0, 0.0, 0.0, 0.0)
  # MuJoCo's body-level fluid model is enabled by the global air density and
  # can otherwise contribute a generic rotor-axis fluid load to child hinges.
  # A zero-coefficient, negligible-size fluid geom on each rotor body keeps
  # that body in the fluid model while making its generic load numerically zero.
  for rotor_name in ROTOR_BODIES:
    rotor_body = spec.body(rotor_name)
    rotor_fluid_disable = cast(
      Any,
      rotor_body.add_geom(name=f"{rotor_name}_fluid_disable"),
    )
    rotor_fluid_disable.type = mujoco.mjtGeom.mjGEOM_ELLIPSOID
    rotor_fluid_disable.size = ROTOR_FLUID_DISABLE_SIZE
    rotor_fluid_disable.density = 0.0
    rotor_fluid_disable.contype = 0
    rotor_fluid_disable.conaffinity = 0
    rotor_fluid_disable.fluid_ellipsoid = 1.0
    rotor_fluid_disable.fluid_coefs = (0.0, 0.0, 0.0, 0.0, 0.0)
    rotor_fluid_disable.rgba = (0.0, 0.0, 0.0, 0.0)
  # Entity options are not propagated by MjSpec.attach(); the V1 scene callback
  # applies these values again to the assembled scene spec.
  spec.option.density = BODY_AIR_DENSITY
  spec.option.viscosity = BODY_AIR_VISCOSITY
  return spec


def get_tony5_aero_robot_cfg() -> EntityCfg:
  """Return a fresh TONY5 robot config using the V1 fluid-enabled spec."""
  cfg = get_tony5_robot_cfg()
  cfg.spec_fn = get_tony5_aero_spec
  return cfg


def configure_tony5_aero_scene(spec: mujoco.MjSpec) -> None:
  """Configure V1 air density and retain the visual-only ground plane."""
  terrain = spec.geom("terrain")
  terrain.contype = 0
  terrain.conaffinity = 0
  spec.option.density = BODY_AIR_DENSITY
  spec.option.viscosity = BODY_AIR_VISCOSITY


class Tony5AeroRotorAerodynamics(Tony5RotorAerodynamics):
  """V1 rotor thrust, reaction torque, local airflow, and H-force."""

  def __init__(self, env: ManagerBasedRlEnv, entity_name: str) -> None:
    super().__init__(env, entity_name)
    shape = (self.num_envs, len(ROTOR_BODIES), 3)
    self._local_air_velocity_b = torch.zeros(shape, device=env.device)
    self._rotor_plane_air_velocity_w = torch.zeros(shape, device=env.device)
    self._h_force_w = torch.zeros(shape, device=env.device)

  @property
  def local_air_velocity_b(self) -> torch.Tensor:
    """Local no-wind air velocity at each rotor in body coordinates."""
    return self._local_air_velocity_b

  @property
  def rotor_plane_air_velocity_w(self) -> torch.Tensor:
    """Rotor-plane component of local air velocity in world coordinates."""
    return self._rotor_plane_air_velocity_w

  @property
  def h_force_w(self) -> torch.Tensor:
    """Most recently computed rotor H-force in world coordinates."""
    return self._h_force_w

  def apply(self) -> None:
    """Write V1 rotor wrenches for every environment."""
    forces, torques = self.compute_wrenches()
    self._asset.write_external_wrench_to_sim(
      forces,
      torques,
      body_ids=cast(Sequence[int], self._body_ids),
    )

  def compute_wrenches(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Add local-airflow H-force to the unchanged V0 rotor wrenches."""
    forces, torques = super().compute_wrenches()

    root_pos_w = self._asset.data.root_link_pos_w
    root_quat_w = self._asset.data.root_link_quat_w
    root_lin_vel_b = self._asset.data.root_link_lin_vel_b
    root_ang_vel_b = self._asset.data.root_link_ang_vel_b
    rotor_pos_w = self._asset.data.body_link_pos_w[:, self._body_ids]
    rotor_offset_w = rotor_pos_w - root_pos_w.unsqueeze(1)

    root_quat_expanded = root_quat_w.unsqueeze(1).expand(-1, len(ROTOR_BODIES), -1)
    rotor_offset_b = quat_apply_inverse(root_quat_expanded, rotor_offset_w)
    local_air_velocity_b = root_lin_vel_b.unsqueeze(1) + torch.cross(
      root_ang_vel_b.unsqueeze(1).expand_as(rotor_offset_b),
      rotor_offset_b,
      dim=-1,
    )

    global_body_ids = self._asset.data.indexing.body_ids[self._body_ids]
    body_xmat = self._env.sim.data.xmat[:, global_body_ids].reshape(
      self.num_envs, len(ROTOR_BODIES), 3, 3
    )
    rotor_axis_w = body_xmat[..., :, 2]
    rotor_plane_air_velocity_w = local_air_velocity_b
    rotor_plane_air_velocity_w = torch.matmul(
      body_xmat,
      rotor_plane_air_velocity_w.unsqueeze(-1),
    ).squeeze(-1)
    axial_velocity_w = torch.sum(
      rotor_plane_air_velocity_w * rotor_axis_w,
      dim=-1,
      keepdim=True,
    )
    rotor_plane_air_velocity_w = (
      rotor_plane_air_velocity_w - axial_velocity_w * rotor_axis_w
    )

    omega = self._asset.data.joint_vel[:, self._joint_ids]
    h_force_w = -ROTOR_DRAG_KH * omega.abs().unsqueeze(-1) * rotor_plane_air_velocity_w

    self._local_air_velocity_b[:] = local_air_velocity_b
    self._rotor_plane_air_velocity_w[:] = rotor_plane_air_velocity_w
    self._h_force_w[:] = h_force_w
    return forces + h_force_w, torques


__all__ = [
  "BODY_AERO_FLUID_COEFS",
  "BODY_AERO_POS",
  "BODY_AERO_SIZE",
  "BODY_AIR_DENSITY",
  "BODY_AIR_VISCOSITY",
  "ROTOR_FLUID_DISABLE_SIZE",
  "ROTOR_DRAG_KH",
  "Tony5AeroRotorAerodynamics",
  "configure_tony5_aero_scene",
  "get_tony5_aero_robot_cfg",
  "get_tony5_aero_spec",
]
