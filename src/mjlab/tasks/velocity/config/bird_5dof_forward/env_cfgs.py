"""Forward-biased 3D velocity tracking for the five-DoF bird."""

from mjlab.asset_zoo.robots.bird_5dof_forward import get_forward_spec
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.tasks.velocity.config.bird_5dof.env_cfgs import (
  bird_5dof_velocity_env_cfg,
)
from mjlab.tasks.velocity.config.bird_5dof.flight_profile import (
  DIRECTION_MIN_SPEED,
  DIRECTION_REWARD_STD,
  DIRECTIONAL_FLIGHT_REWARD_WEIGHTS,
  ROLL_FULL_SPEED,
  ROLL_MIN_HORIZONTAL,
  ROLL_REWARD_STD,
  ROLL_ZERO_SPEED,
  TILT_LIMIT_DEG,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from .events import reset_heading_forward_velocity

FORWARD_REWARD_WEIGHTS = DIRECTIONAL_FLIGHT_REWARD_WEIGHTS
FORWARD_DIRECTION_REWARD_STD = DIRECTION_REWARD_STD
FORWARD_DIRECTION_MIN_SPEED = DIRECTION_MIN_SPEED
FORWARD_ROLL_REWARD_STD = ROLL_REWARD_STD
FORWARD_ROLL_MIN_HORIZONTAL = ROLL_MIN_HORIZONTAL
FORWARD_ROLL_FULL_SPEED = ROLL_FULL_SPEED
FORWARD_ROLL_ZERO_SPEED = ROLL_ZERO_SPEED
FORWARD_TILT_LIMIT_DEG = TILT_LIMIT_DEG


def bird_5dof_forward_3d_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a dedicated forward-flight task with 3D linear commands."""
  cfg = bird_5dof_velocity_env_cfg(play=play)
  cfg.scene.entities["robot"].spec_fn = get_forward_spec
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)

  command.ranges.lin_vel_x = (0.0, 10.0)
  command.ranges.lin_vel_y = (-4.0, 4.0)
  command.ranges.lin_vel_z = (-6.0, 6.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  command.resampling_time_range = (2.0, 4.0)
  command.rel_standing_envs = 0.0
  command.rel_world_envs = 1.0
  command.rel_forward_envs = 0.0
  command.yaw_command_speed_threshold = None
  command.debug_vis = play

  cfg.events["reset_forward_velocity"] = EventTermCfg(
    func=reset_heading_forward_velocity,
    mode="reset",
    params={"speed_range": (0.75, 1.5)},
  )

  # This task always samples the same forward-biased 3D command distribution.
  cfg.curriculum = {}
  return cfg
