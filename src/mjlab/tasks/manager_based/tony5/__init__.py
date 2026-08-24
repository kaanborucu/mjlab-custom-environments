"""TONY5 V0 position+yaw quadrotor task."""

from mjlab.tasks.registry import register_mjlab_task

from .tony5_aero_env_cfg import (
  tony5_velocity_aero_env_cfg,
  tony5_velocity_aero_play_env_cfg,
)
from .tony5_aero_ppo_cfg import tony5_aero_ppo_runner_cfg
from .tony5_env_cfg import (
  tony5_position_yaw_env_cfg,
  tony5_position_yaw_play_env_cfg,
)
from .tony5_omni_env_cfg import (
  tony5_velocity_aero_omni_env_cfg,
  tony5_velocity_aero_omni_play_env_cfg,
)
from .tony5_omni_ppo_cfg import tony5_omni_ppo_runner_cfg
from .tony5_omni_v3_env_cfg import (
  tony5_velocity_aero_omni_v3_env_cfg,
  tony5_velocity_aero_omni_v3_play_env_cfg,
)
from .tony5_omni_v3_ppo_cfg import tony5_omni_v3_ppo_runner_cfg
from .tony5_ppo_cfg import tony5_ppo_runner_cfg
from .tony5_velocity_env_cfg import (
  tony5_velocity_env_cfg,
  tony5_velocity_play_env_cfg,
)

POSITION_YAW_TASK_ID = "Mjlab-Tony5-PositionYaw-v0"
VELOCITY_TASK_ID = "Mjlab-Tony5-Velocity-v0"
VELOCITY_AERO_TASK_ID = "Mjlab-Tony5-Velocity-Aero-v1"
VELOCITY_AERO_OMNI_TASK_ID = "Mjlab-Tony5-Velocity-Aero-Omni-v0"
VELOCITY_AERO_OMNI_V3_TASK_ID = "Mjlab-Tony5-Velocity-Aero-Omni-v3"
TASK_ID = POSITION_YAW_TASK_ID

register_mjlab_task(
  task_id=POSITION_YAW_TASK_ID,
  env_cfg=tony5_position_yaw_env_cfg(),
  play_env_cfg=tony5_position_yaw_play_env_cfg(),
  rl_cfg=tony5_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id=VELOCITY_TASK_ID,
  env_cfg=tony5_velocity_env_cfg(),
  play_env_cfg=tony5_velocity_play_env_cfg(),
  rl_cfg=tony5_ppo_runner_cfg(experiment_name="tony5_velocity_v0_history5"),
)

register_mjlab_task(
  task_id=VELOCITY_AERO_TASK_ID,
  env_cfg=tony5_velocity_aero_env_cfg(),
  play_env_cfg=tony5_velocity_aero_play_env_cfg(),
  rl_cfg=tony5_aero_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id=VELOCITY_AERO_OMNI_TASK_ID,
  env_cfg=tony5_velocity_aero_omni_env_cfg(),
  play_env_cfg=tony5_velocity_aero_omni_play_env_cfg(),
  rl_cfg=tony5_omni_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id=VELOCITY_AERO_OMNI_V3_TASK_ID,
  env_cfg=tony5_velocity_aero_omni_v3_env_cfg(),
  play_env_cfg=tony5_velocity_aero_omni_v3_play_env_cfg(),
  rl_cfg=tony5_omni_v3_ppo_runner_cfg(),
)

__all__ = [
  "POSITION_YAW_TASK_ID",
  "TASK_ID",
  "VELOCITY_AERO_TASK_ID",
  "VELOCITY_AERO_OMNI_TASK_ID",
  "VELOCITY_AERO_OMNI_V3_TASK_ID",
  "VELOCITY_TASK_ID",
]
