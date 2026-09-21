"""Episode diagnostics for the TONY5 minimum-time task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manager_based.tony5 import tony5_mdp as mdp
from mjlab.tasks.manager_based.tony5.tony5_position_min_time_terminations import (
  success_condition,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class Tony5PositionMinTimeMetric(ManagerTermBase):
  """Stateful episode metric with one instance per configured metric."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    self._kind = str(cfg.params["kind"])
    self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._initial_distance = torch.zeros(self.num_envs, device=self.device)
    self._minimum_distance = torch.full(
      (self.num_envs,), float("inf"), device=self.device
    )
    self._previous_distance = torch.zeros(self.num_envs, device=self.device)

  def __call__(self, env: ManagerBasedRlEnv, kind: str | None = None) -> torch.Tensor:
    del kind
    asset = env.scene["robot"]
    distance = torch.linalg.vector_norm(
      mdp.position_error_w(env, "target_pose"), dim=-1
    )
    if self._kind in {
      "target_distance",
      "minimum_distance_reached",
      "distance_progress",
    }:
      new_envs = ~self._initialized
      self._initial_distance[new_envs] = distance[new_envs]
      self._minimum_distance[new_envs] = distance[new_envs]
      self._previous_distance[new_envs] = distance[new_envs]
      self._initialized[new_envs] = True
      torch.minimum(self._minimum_distance, distance, out=self._minimum_distance)

    if self._kind == "target_distance":
      return self._initial_distance
    if self._kind == "minimum_distance_reached":
      return self._minimum_distance
    if self._kind == "distance_progress":
      progress = self._previous_distance - distance
      self._previous_distance[:] = distance
      return progress
    if self._kind == "current_distance":
      return distance
    if self._kind in {"episode_success", "success_rate"}:
      if "success" in env.termination_manager.active_terms:
        return env.termination_manager.get_term("success").float()
      return success_condition(env).float()
    if self._kind == "time_to_success":
      if "success" in env.termination_manager.active_terms:
        success = env.termination_manager.get_term("success")
      else:
        success = success_condition(env)
      return torch.where(
        success,
        env.episode_length_buf.float() * env.step_dt,
        torch.zeros_like(distance),
      )
    if self._kind == "final_position_error":
      return distance
    if self._kind == "final_yaw_error":
      return mdp.yaw_error(env, "target_pose").abs()
    if self._kind == "final_linear_speed":
      return torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1)
    if self._kind == "current_linear_speed":
      return torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1)
    if self._kind == "peak_linear_speed":
      return torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1)
    if self._kind == "peak_horizontal_speed":
      return torch.linalg.vector_norm(asset.data.root_link_lin_vel_w[:, :2], dim=-1)
    if self._kind == "current_angular_speed":
      return torch.linalg.vector_norm(asset.data.root_link_ang_vel_b, dim=-1)
    if self._kind == "height":
      return mdp.root_height(env).squeeze(-1)
    if self._kind == "crash_rate":
      active_terms = env.termination_manager.active_terms
      if (
        "rapid_descent" not in active_terms
        or "numerical_safety_failure" not in active_terms
      ):
        return torch.zeros(env.num_envs, device=env.device)
      return (
        env.termination_manager.get_term("rapid_descent")
        | env.termination_manager.get_term("numerical_safety_failure")
      ).float()
    raise ValueError(f"Unknown TONY5 minimum-time metric kind: {self._kind}")

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._initialized[env_ids] = False
    self._initial_distance[env_ids] = 0.0
    self._minimum_distance[env_ids] = float("inf")
    self._previous_distance[env_ids] = 0.0


def tony5_position_min_time_metrics() -> dict[str, MetricsTermCfg]:
  """Return the compact V5 metric set."""

  def metric(kind: str, reduce: str = "last") -> MetricsTermCfg:
    return MetricsTermCfg(
      func=Tony5PositionMinTimeMetric,
      params={"kind": kind},
      reduce=reduce,  # type: ignore[arg-type]
    )

  return {
    "target_distance": metric("target_distance"),
    "current_distance": metric("current_distance"),
    "distance_progress": metric("distance_progress", "sum"),
    "minimum_distance_reached": metric("minimum_distance_reached"),
    "episode_success": metric("episode_success"),
    "success_rate": metric("success_rate"),
    "time_to_success": metric("time_to_success"),
    "final_position_error": metric("final_position_error"),
    "final_yaw_error": metric("final_yaw_error"),
    "final_linear_speed": metric("final_linear_speed"),
    "current_linear_speed": metric("current_linear_speed"),
    "peak_linear_speed": metric("peak_linear_speed", "max"),
    "peak_horizontal_speed": metric("peak_horizontal_speed", "max"),
    "current_angular_speed": metric("current_angular_speed"),
    "height": metric("height"),
    "crash_rate": metric("crash_rate"),
  }


__all__ = ["Tony5PositionMinTimeMetric", "tony5_position_min_time_metrics"]
