"""Per-substep TONY5 rotor aerodynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, cast

import torch

from mjlab.entity import Entity
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  KQ_TORQUE,
  KT_THRUST,
  ROTOR_AXIS_SIGNS,
  ROTOR_BODIES,
  ROTOR_JOINTS,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class Tony5RotorAerodynamics:
  """Apply actual-qvel thrust and rotor-body drag before each physics step."""

  def __init__(self, env: ManagerBasedRlEnv, entity_name: str) -> None:
    self._env = env
    self._asset: Entity = env.scene[entity_name]
    joint_ids, joint_names = self._asset.find_joints(ROTOR_JOINTS, preserve_order=True)
    body_ids, body_names = self._asset.find_bodies(ROTOR_BODIES, preserve_order=True)
    if tuple(joint_names) != ROTOR_JOINTS:
      raise ValueError(f"Unexpected TONY5 joint order: {joint_names}")
    if tuple(body_names) != ROTOR_BODIES:
      raise ValueError(f"Unexpected TONY5 body order: {body_names}")
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    self._body_ids = torch.tensor(body_ids, device=env.device, dtype=torch.long)
    joint_specs = {
      joint.name.split("/")[-1]: joint for joint in self._asset.spec.joints
    }
    axis_signs = tuple(float(joint_specs[name].axis[2]) for name in ROTOR_JOINTS)
    if axis_signs != ROTOR_AXIS_SIGNS:
      raise ValueError(f"Unexpected TONY5 hinge-axis signs: {axis_signs}")
    self._axis_signs = torch.tensor(axis_signs, device=env.device, dtype=torch.float)

  @property
  def joint_ids(self) -> torch.Tensor:
    return self._joint_ids

  @property
  def body_ids(self) -> torch.Tensor:
    return self._body_ids

  def compute_wrenches(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame rotor forces and drag torques for all environments."""
    omega = self._asset.data.joint_vel[:, self._joint_ids]
    omega_abs = omega.abs()
    thrust = KT_THRUST * omega_abs.square()
    drag_joint = -KQ_TORQUE * omega * omega_abs

    global_body_ids = self._asset.data.indexing.body_ids[self._body_ids]
    body_xmat = self._env.sim.data.xmat[:, global_body_ids].reshape(
      self.num_envs, len(ROTOR_BODIES), 3, 3
    )
    body_z_world = body_xmat[..., :, 2]
    rotor_axis_world = body_z_world * self._axis_signs.view(1, -1, 1)
    forces = body_z_world * thrust.unsqueeze(-1)
    torques = rotor_axis_world * drag_joint.unsqueeze(-1)
    return forces, torques

  @property
  def num_envs(self) -> int:
    return self._env.num_envs

  def apply(self) -> None:
    """Write the current rotor wrenches to MuJoCo's external-force buffer."""
    forces, torques = self.compute_wrenches()
    self._asset.write_external_wrench_to_sim(
      forces,
      torques,
      # EntityData uses tensor indexing here for batched local-to-global mapping.
      body_ids=cast(Sequence[int], self._body_ids),
    )


__all__ = ["Tony5RotorAerodynamics"]
