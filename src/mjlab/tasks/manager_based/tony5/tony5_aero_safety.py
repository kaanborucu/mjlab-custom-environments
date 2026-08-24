"""V1-only numerical safety monitoring for the TONY5 aero task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


ROOT_SPEED_LIMIT = 150.0
BODY_ANGULAR_SPEED_LIMIT = 200.0
QACC_LIMIT = 4.0e5


class Tony5NumericalSafetyMonitor:
  """Track V1 numerical limits and persist per-episode safety state."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    asset: Entity,
    rotor_joint_ids: torch.Tensor,
  ) -> None:
    self._env = env
    self._rotor_joint_ids = rotor_joint_ids
    self._free_v_adr = asset.indexing.free_joint_v_adr
    shape = (env.num_envs,)
    self.failed = torch.zeros(shape, dtype=torch.bool, device=env.device)
    self.failure_event = torch.zeros(shape, dtype=torch.bool, device=env.device)
    self.root_speed_count = torch.zeros(shape, dtype=torch.float, device=env.device)
    self.body_omega_count = torch.zeros(shape, dtype=torch.float, device=env.device)
    self.rotor_speed_count = torch.zeros(shape, dtype=torch.float, device=env.device)
    self.qacc_count = torch.zeros(shape, dtype=torch.float, device=env.device)
    self.nonfinite_count = torch.zeros(shape, dtype=torch.float, device=env.device)
    self.total_failure_count = torch.zeros((), dtype=torch.long, device=env.device)
    self.total_root_speed_count = torch.zeros((), dtype=torch.long, device=env.device)
    self.total_body_omega_count = torch.zeros((), dtype=torch.long, device=env.device)
    self.total_rotor_speed_count = torch.zeros((), dtype=torch.long, device=env.device)
    self.total_qacc_count = torch.zeros((), dtype=torch.long, device=env.device)
    self.total_nonfinite_count = torch.zeros((), dtype=torch.long, device=env.device)
    self.max_root_speed = torch.zeros(shape, dtype=torch.float, device=env.device)
    self.max_body_omega = torch.zeros(shape, dtype=torch.float, device=env.device)
    self.max_rotor_speed = torch.zeros(shape, dtype=torch.float, device=env.device)
    self.max_abs_qacc = torch.zeros(shape, dtype=torch.float, device=env.device)

  def _finite_max_update(
    self,
    destination: torch.Tensor,
    value: torch.Tensor,
  ) -> None:
    finite = torch.isfinite(value)
    finite_value = torch.where(finite, value, torch.zeros_like(value))
    torch.maximum(destination, finite_value, out=destination)

  def check(self) -> torch.Tensor:
    """Check current simulator state and return persistent failure flags."""
    data = self._env.sim.data
    qpos = data.qpos
    qvel = data.qvel
    qacc = data.qacc
    root_linear_velocity = qvel[:, self._free_v_adr[:3]]
    body_angular_velocity = qvel[:, self._free_v_adr[3:6]]
    rotor_velocity = qvel[:, self._rotor_joint_ids]

    root_speed = torch.linalg.vector_norm(root_linear_velocity, dim=-1)
    body_omega = torch.linalg.vector_norm(body_angular_velocity, dim=-1)
    rotor_speed = rotor_velocity.abs().amax(dim=-1)
    max_abs_qacc = qacc.abs().amax(dim=-1)

    self._finite_max_update(self.max_root_speed, root_speed)
    self._finite_max_update(self.max_body_omega, body_omega)
    self._finite_max_update(self.max_rotor_speed, rotor_speed)
    self._finite_max_update(self.max_abs_qacc, max_abs_qacc)

    nonfinite = ~(
      torch.isfinite(qpos).all(dim=-1)
      & torch.isfinite(qvel).all(dim=-1)
      & torch.isfinite(qacc).all(dim=-1)
    )
    root_bad = torch.isfinite(root_speed) & (root_speed > ROOT_SPEED_LIMIT)
    body_bad = torch.isfinite(body_omega) & (body_omega > BODY_ANGULAR_SPEED_LIMIT)
    qacc_bad = torch.isfinite(max_abs_qacc) & (max_abs_qacc > QACC_LIMIT)
    invalid = nonfinite | root_bad | body_bad | qacc_bad
    new_failure = invalid & ~self.failed

    self.root_speed_count += (new_failure & root_bad).float()
    self.body_omega_count += (new_failure & body_bad).float()
    self.qacc_count += (new_failure & qacc_bad).float()
    self.nonfinite_count += (new_failure & nonfinite).float()
    self.total_failure_count += new_failure.sum()
    self.total_root_speed_count += (new_failure & root_bad).sum()
    self.total_body_omega_count += (new_failure & body_bad).sum()
    self.total_qacc_count += (new_failure & qacc_bad).sum()
    self.total_nonfinite_count += (new_failure & nonfinite).sum()
    self.failed |= invalid
    self.failure_event |= new_failure
    return self.failed

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.failed[env_ids] = False
    self.failure_event[env_ids] = False
    self.root_speed_count[env_ids] = 0.0
    self.body_omega_count[env_ids] = 0.0
    self.rotor_speed_count[env_ids] = 0.0
    self.qacc_count[env_ids] = 0.0
    self.nonfinite_count[env_ids] = 0.0
    self.max_root_speed[env_ids] = 0.0
    self.max_body_omega[env_ids] = 0.0
    self.max_rotor_speed[env_ids] = 0.0
    self.max_abs_qacc[env_ids] = 0.0


def _monitor(env: ManagerBasedRlEnv) -> Tony5NumericalSafetyMonitor:
  action = env.action_manager.get_term("rotor_speed")
  monitor = getattr(action, "safety_monitor", None)
  if not isinstance(monitor, Tony5NumericalSafetyMonitor):
    raise TypeError("TONY5 Aero numerical safety monitor is not installed")
  return monitor


def numerical_safety_failure(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate V1 environments that exceed numerical safety limits."""
  return _monitor(env).check()


def numerical_safety_failure_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _monitor(env).failure_event.float()


def numerical_safety_root_speed_count(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _monitor(env).root_speed_count


def numerical_safety_body_omega_count(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _monitor(env).body_omega_count


def numerical_safety_rotor_speed_count(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _monitor(env).rotor_speed_count


def numerical_safety_qacc_count(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _monitor(env).qacc_count


def numerical_safety_nonfinite_count(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _monitor(env).nonfinite_count


def max_root_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _monitor(env).max_root_speed


def max_body_angular_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _monitor(env).max_body_omega


def max_rotor_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _monitor(env).max_rotor_speed


def max_abs_qacc(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _monitor(env).max_abs_qacc


__all__ = [
  "BODY_ANGULAR_SPEED_LIMIT",
  "QACC_LIMIT",
  "ROOT_SPEED_LIMIT",
  "Tony5NumericalSafetyMonitor",
  "max_abs_qacc",
  "max_body_angular_speed",
  "max_root_speed",
  "max_rotor_speed",
  "numerical_safety_body_omega_count",
  "numerical_safety_failure",
  "numerical_safety_failure_rate",
  "numerical_safety_nonfinite_count",
  "numerical_safety_qacc_count",
  "numerical_safety_root_speed_count",
  "numerical_safety_rotor_speed_count",
]
