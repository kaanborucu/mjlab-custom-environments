"""Play-mode diagnostics for the TONY5 minimum-time task."""

from __future__ import annotations

import torch

from mjlab.tasks.manager_based.tony5 import tony5_mdp as mdp
from mjlab.tasks.manager_based.tony5.tony5_position_min_time_terminations import (
  success_condition,
)


class Tony5PositionMinTimeStatus:
  """Format target and flight diagnostics for the native/Viser status panels."""

  def __init__(self, env) -> None:
    self._env = env
    self._peak_speed = 0.0
    self._last_episode_step = -1

  def __call__(self) -> tuple[str, str]:
    env = self._env
    env_idx = env.cfg.viewer.env_idx
    command = env.command_manager.get_command("target_pose")[env_idx]
    target_position = command[:3] + env.scene.env_origins[env_idx]
    position = env.scene["robot"].data.root_link_pos_w[env_idx]
    velocity = env.scene["robot"].data.root_link_lin_vel_w[env_idx]
    yaw_error = mdp.yaw_error(env, "target_pose")[env_idx].abs()
    position_error = torch.linalg.vector_norm(target_position - position)
    speed = torch.linalg.vector_norm(velocity)
    episode_step = int(env.episode_length_buf[env_idx].item())
    if episode_step <= self._last_episode_step:
      self._peak_speed = 0.0
    self._last_episode_step = episode_step
    self._peak_speed = max(self._peak_speed, float(speed.item()))
    elapsed = episode_step * env.step_dt
    if "success" in env.termination_manager.active_terms:
      success = bool(env.termination_manager.get_term("success")[env_idx].item())
    else:
      success = bool(success_condition(env)[env_idx].item())

    labels = (
      "Target position\n"
      "Target yaw\n"
      "Distance\n"
      "Elapsed\n"
      "Current speed\n"
      "Peak speed\n"
      "Position error\n"
      "Yaw error\n"
      "Success"
    )
    values = (
      f"[{target_position[0].item():+.2f}, {target_position[1].item():+.2f}, "
      f"{target_position[2].item():+.2f}] m\n"
      f"{torch.rad2deg(command[3]).item():+.1f} deg\n"
      f"{position_error.item():.2f} m\n"
      f"{elapsed:.2f} s\n"
      f"{speed.item():.2f} m/s\n"
      f"{self._peak_speed:.2f} m/s\n"
      f"{position_error.item():.2f} m\n"
      f"{torch.rad2deg(yaw_error).item():.1f} deg\n"
      f"{'yes' if success else 'no'}"
    )
    return labels, values


__all__ = ["Tony5PositionMinTimeStatus"]
