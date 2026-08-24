"""Global wind and gust support for the TONY5 Omni V3 task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.tasks.manager_based.tony5.tony5_constants import PHYSICS_DT

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


@dataclass(kw_only=True)
class Tony5OmniV3WindCfg:
  """Configuration for one global world-frame wind process."""

  enable_wind: bool = False
  wind_x: float = 0.0
  wind_y: float = 0.0
  wind_z: float = 0.0
  randomize_background_wind: bool = False
  background_horizontal_speed_range: tuple[float, float] = (0.0, 0.0)
  background_vertical_speed_range: tuple[float, float] = (0.0, 0.0)
  enable_gusts: bool = False
  gust_sigma: float = 1.0
  gust_tau: float = 1.0
  max_gust_speed: float = 5.0
  reset_enable_probability: float = 0.5


class Tony5OmniV3GlobalWind:
  """Compose one global wind and write it to native MJWarp in place."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    cfg: Tony5OmniV3WindCfg,
  ) -> None:
    if abs(env.physics_dt - PHYSICS_DT) > 1.0e-12:
      raise ValueError(
        f"TONY5 Omni V3 requires a {PHYSICS_DT:g} s physics step, "
        f"got {env.physics_dt:g} s."
      )
    if cfg.gust_sigma < 0.0:
      raise ValueError("gust_sigma must be non-negative.")
    if cfg.gust_tau <= 0.0:
      raise ValueError("gust_tau must be positive.")
    if cfg.enable_gusts and cfg.max_gust_speed <= 0.0:
      raise ValueError("max_gust_speed must be positive when gusts are enabled.")
    if not 0.0 <= cfg.reset_enable_probability <= 1.0:
      raise ValueError("reset_enable_probability must be between 0 and 1.")
    min_horizontal_speed, max_horizontal_speed = cfg.background_horizontal_speed_range
    if min_horizontal_speed < 0.0 or max_horizontal_speed < min_horizontal_speed:
      raise ValueError(
        "background_horizontal_speed_range must satisfy 0 <= min <= max."
      )
    min_vertical_speed, max_vertical_speed = cfg.background_vertical_speed_range
    if max_vertical_speed < min_vertical_speed:
      raise ValueError("background_vertical_speed_range must satisfy min <= max.")

    native_wind = env.sim.model.opt.wind
    native_wind_storage_shape = tuple(native_wind.wp_array.shape)
    if native_wind_storage_shape != (1,):
      raise ValueError(
        "TONY5 Omni V3 requires MJWarp's global Vec3 wind buffer with logical "
        f"shape (1, 3), got underlying shape {native_wind_storage_shape}."
      )
    if not env.sim.model.has_fluid:
      raise ValueError("TONY5 Omni V3 requires native fluid physics to be enabled.")

    self.cfg = cfg
    self._env = env
    self._native_wind: Any = native_wind
    background_wind = torch.tensor(
      (cfg.wind_x, cfg.wind_y, cfg.wind_z),
      dtype=torch.float,
      device=env.device,
    )
    self._background_wind_world = background_wind.clone()
    self._gust_world = torch.zeros(3, dtype=torch.float, device=env.device)
    self._wind_world = torch.zeros_like(self._gust_world)
    self._episode_enabled = False
    self._gust_decay = math.exp(-env.physics_dt / cfg.gust_tau)
    self._gust_noise_scale = cfg.gust_sigma * math.sqrt(
      1.0 - self._gust_decay * self._gust_decay
    )
    self._sample_episode_enable()
    self._sample_background_wind()
    self._compose_and_write()

  @property
  def wind_world(self) -> torch.Tensor:
    """The canonical global world-frame wind vector."""
    return self._wind_world

  @property
  def gust_world(self) -> torch.Tensor:
    """The current global world-frame gust component."""
    return self._gust_world

  @property
  def episode_enabled(self) -> bool:
    """Whether wind and gusts are enabled for the current full reset."""
    return self._episode_enabled

  def step(self) -> None:
    """Advance gust state and update native wind before the next physics step."""
    if self.cfg.enable_gusts and self._episode_enabled:
      noise = torch.randn(3, dtype=torch.float, device=self._env.device)
      self._gust_world.mul_(self._gust_decay).add_(self._gust_noise_scale * noise)
      gust_speed = torch.linalg.vector_norm(self._gust_world)
      gust_scale = torch.clamp(
        self.cfg.max_gust_speed / torch.clamp(gust_speed, min=1.0e-12),
        max=1.0,
      )
      self._gust_world.mul_(gust_scale)
    self._compose_and_write()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset global wind state only when all environments reset together."""
    if not self._is_full_reset(env_ids):
      return
    self._sample_episode_enable()
    self._sample_background_wind()
    self._gust_world.zero_()
    self._compose_and_write()

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    """Draw the total world-frame wind as one green arrow."""
    env_indices = visualizer.get_env_indices(self._env.num_envs)
    if not env_indices:
      return

    positions = self._env.scene["robot"].data.root_link_pos_w.detach().cpu().numpy()
    vector_np = self._wind_world.detach().cpu().numpy()
    magnitude = float(np.linalg.norm(vector_np))
    if magnitude < 1.0e-9:
      return
    for env_idx in env_indices:
      start = positions[env_idx].copy()
      visualizer.add_arrow(
        start=start,
        end=start + 0.08 * vector_np,
        color=(0.2, 1.0, 0.2, 1.0),
        width=0.018,
        label=f"tony5_total_wind_{env_idx}",
      )

  def _is_full_reset(self, env_ids: torch.Tensor | slice | None) -> bool:
    if env_ids is None:
      return True
    if isinstance(env_ids, slice):
      return env_ids == slice(None)
    return env_ids.numel() == self._env.num_envs

  def _compose_and_write(self) -> None:
    """Compose background plus gust once, then update native wind in place."""
    self._wind_world.zero_()
    if self.cfg.enable_wind and self._episode_enabled:
      self._wind_world.add_(self._background_wind_world)
    if self.cfg.enable_gusts and self._episode_enabled:
      self._wind_world.add_(self._gust_world)
    self._native_wind[0] = self._wind_world

  def _sample_episode_enable(self) -> None:
    """Choose once whether the shared process is active for this reset."""
    enabled = torch.rand((), dtype=torch.float, device=self._env.device)
    self._episode_enabled = bool(enabled.item() < self.cfg.reset_enable_probability)

  def _sample_background_wind(self) -> None:
    """Sample one shared 3-D background wind vector for a full reset."""
    if not self.cfg.randomize_background_wind:
      return
    min_horizontal_speed, max_horizontal_speed = (
      self.cfg.background_horizontal_speed_range
    )
    horizontal_speed = torch.rand((), dtype=torch.float, device=self._env.device)
    horizontal_speed = (
      min_horizontal_speed
      + (max_horizontal_speed - min_horizontal_speed) * horizontal_speed
    )
    azimuth = (
      2.0 * torch.pi * torch.rand((), dtype=torch.float, device=self._env.device)
    )
    min_vertical_speed, max_vertical_speed = self.cfg.background_vertical_speed_range
    vertical_speed = torch.rand((), dtype=torch.float, device=self._env.device)
    vertical_speed = (
      min_vertical_speed + (max_vertical_speed - min_vertical_speed) * vertical_speed
    )
    self._background_wind_world.copy_(
      torch.stack(
        (
          horizontal_speed * torch.cos(azimuth),
          horizontal_speed * torch.sin(azimuth),
          vertical_speed,
        )
      )
    )


def _global_wind(env: ManagerBasedRlEnv) -> Tony5OmniV3GlobalWind:
  action = env.action_manager.get_term("rotor_speed")
  wind = getattr(action, "wind", None)
  if not isinstance(wind, Tony5OmniV3GlobalWind):
    raise TypeError("TONY5 Omni V3 global wind is not installed.")
  return wind


def _broadcast_global(value: torch.Tensor, env: ManagerBasedRlEnv) -> torch.Tensor:
  return value.expand(env.num_envs)


def wind_x(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _broadcast_global(_global_wind(env).wind_world[0], env)


def wind_y(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _broadcast_global(_global_wind(env).wind_world[1], env)


def wind_z(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _broadcast_global(_global_wind(env).wind_world[2], env)


def wind_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _broadcast_global(
    torch.linalg.vector_norm(_global_wind(env).wind_world),
    env,
  )


def gust_x(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _broadcast_global(_global_wind(env).gust_world[0], env)


def gust_y(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _broadcast_global(_global_wind(env).gust_world[1], env)


def gust_z(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _broadcast_global(_global_wind(env).gust_world[2], env)


def gust_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _broadcast_global(
    torch.linalg.vector_norm(_global_wind(env).gust_world),
    env,
  )


def tony5_omni_v3_wind_metrics() -> dict[str, MetricsTermCfg]:
  """Return global wind and gust metrics for V3 logging."""
  return {
    "wind_x": MetricsTermCfg(func=wind_x),
    "wind_y": MetricsTermCfg(func=wind_y),
    "wind_z": MetricsTermCfg(func=wind_z),
    "wind_speed": MetricsTermCfg(func=wind_speed),
    "gust_x": MetricsTermCfg(func=gust_x),
    "gust_y": MetricsTermCfg(func=gust_y),
    "gust_z": MetricsTermCfg(func=gust_z),
    "gust_speed": MetricsTermCfg(func=gust_speed),
  }


__all__ = [
  "Tony5OmniV3GlobalWind",
  "Tony5OmniV3WindCfg",
  "gust_speed",
  "gust_x",
  "gust_y",
  "gust_z",
  "tony5_omni_v3_wind_metrics",
  "wind_speed",
  "wind_x",
  "wind_y",
  "wind_z",
]
