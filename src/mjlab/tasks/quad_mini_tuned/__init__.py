"""Quad Mini Tuned task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import quad_mini_tuned_env_cfg
from .rl_cfg import quad_mini_tuned_ppo_runner_cfg

TASK_ID = "Mjlab-QuadMiniTuned-Joystick-FlatTerrain"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=quad_mini_tuned_env_cfg(),
  play_env_cfg=quad_mini_tuned_env_cfg(play=True),
  rl_cfg=quad_mini_tuned_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

__all__ = ["TASK_ID"]
