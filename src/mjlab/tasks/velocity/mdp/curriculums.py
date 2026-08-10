from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch
from typing_extensions import NotRequired

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  name: NotRequired[str]
  lin_vel_x: NotRequired[tuple[float, float] | None]
  lin_vel_y: NotRequired[tuple[float, float] | None]
  lin_vel_z: NotRequired[tuple[float, float] | None]
  ang_vel_z: NotRequired[tuple[float, float] | None]
  rel_standing_envs: NotRequired[float]
  rel_forward_envs: NotRequired[float]
  resampling_time_range: NotRequired[tuple[float, float]]
  reset_pose_range: NotRequired[dict[str, tuple[float, float]]]
  reset_velocity_range: NotRequired[dict[str, tuple[float, float]]]
  joint_position_ranges: NotRequired[dict[str, tuple[float, float]]]
  joint_velocity_ranges: NotRequired[dict[str, tuple[float, float]]]


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to
  # simpler terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # On the initial reset (before any env step) the robot is still at its spawn
  # pose rather than a walked-to position, so ``distance`` is meaningless and
  # would spuriously promote every env from level 0 to 1, ignoring
  # ``max_init_terrain_level``. Freeze levels on that first reset.
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter >= stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "lin_vel_z" in stage and stage["lin_vel_z"] is not None:
        if cfg.ranges.lin_vel_z is None:
          raise ValueError(
            "A velocity curriculum cannot enable lin_vel_z after the command term "
            "has been built. Configure an initial lin_vel_z range first."
          )
        cfg.ranges.lin_vel_z = stage["lin_vel_z"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
      if "rel_standing_envs" in stage:
        cfg.rel_standing_envs = stage["rel_standing_envs"]
      if "rel_forward_envs" in stage:
        cfg.rel_forward_envs = stage["rel_forward_envs"]
      if "resampling_time_range" in stage:
        cfg.resampling_time_range = stage["resampling_time_range"]
      if "reset_pose_range" in stage:
        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        reset_base_cfg.params["pose_range"].update(stage["reset_pose_range"])
      if "reset_velocity_range" in stage:
        reset_base_cfg = env.event_manager.get_term_cfg("reset_base")
        reset_base_cfg.params["velocity_range"].update(stage["reset_velocity_range"])
      if "joint_position_ranges" in stage:
        for event_name, position_range in stage["joint_position_ranges"].items():
          event_cfg = env.event_manager.get_term_cfg(event_name)
          event_cfg.params["position_range"] = position_range
      if "joint_velocity_ranges" in stage:
        for event_name, velocity_range in stage["joint_velocity_ranges"].items():
          event_cfg = env.event_manager.get_term_cfg(event_name)
          event_cfg.params["velocity_range"] = velocity_range
  result = {
    "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
    "rel_standing_envs": torch.tensor(cfg.rel_standing_envs),
    "rel_forward_envs": torch.tensor(cfg.rel_forward_envs),
    "resampling_time_min": torch.tensor(cfg.resampling_time_range[0]),
    "resampling_time_max": torch.tensor(cfg.resampling_time_range[1]),
  }
  if cfg.ranges.lin_vel_z is not None:
    result["lin_vel_z_min"] = torch.tensor(cfg.ranges.lin_vel_z[0])
    result["lin_vel_z_max"] = torch.tensor(cfg.ranges.lin_vel_z[1])
  return result
