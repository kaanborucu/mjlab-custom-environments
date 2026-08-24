"""Quad Mini Tuned task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import quad_mini_tuned_env_cfg
from .rl_cfg import (
  QUAD_MINI_TUNED_TEACHER_EXPERIMENT,
  quad_mini_tuned_distillation_runner_cfg,
  quad_mini_tuned_ppo_runner_cfg,
)
from .teacher_student_env_cfg import (
  quad_mini_tuned_student_env_cfg,
  quad_mini_tuned_teacher_env_cfg,
)
from .teacher_student_runner import (
  QuadMiniTunedDistillationRunner,
  QuadMiniTunedTeacherRunner,
)

TASK_ID = "Mjlab-QuadMiniTuned-Joystick-FlatTerrain"
TEACHER_TASK_ID = "Mjlab-QuadMiniTuned-Joystick-FlatTerrain-Teacher"
STUDENT_TASK_ID = "Mjlab-QuadMiniTuned-Joystick-FlatTerrain-Student"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=quad_mini_tuned_env_cfg(),
  play_env_cfg=quad_mini_tuned_env_cfg(play=True),
  rl_cfg=quad_mini_tuned_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=TEACHER_TASK_ID,
  env_cfg=quad_mini_tuned_teacher_env_cfg(),
  play_env_cfg=quad_mini_tuned_teacher_env_cfg(play=True),
  rl_cfg=quad_mini_tuned_ppo_runner_cfg(
    experiment_name=QUAD_MINI_TUNED_TEACHER_EXPERIMENT,
  ),
  runner_cls=QuadMiniTunedTeacherRunner,
)

register_mjlab_task(
  task_id=STUDENT_TASK_ID,
  env_cfg=quad_mini_tuned_student_env_cfg(),
  play_env_cfg=quad_mini_tuned_student_env_cfg(play=True),
  rl_cfg=quad_mini_tuned_distillation_runner_cfg(),
  runner_cls=QuadMiniTunedDistillationRunner,
)

__all__ = ["STUDENT_TASK_ID", "TASK_ID", "TEACHER_TASK_ID"]
