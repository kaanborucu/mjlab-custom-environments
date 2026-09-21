"""Teacher and student variants of the original Unitree Go1 tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import (
  go1_original_rough_student_env_cfg,
  go1_original_rough_teacher_env_cfg,
  go1_original_student_env_cfg,
  go1_original_teacher_env_cfg,
)
from .rl_cfg import (
  go1_original_rough_student_distillation_runner_cfg,
  go1_original_rough_teacher_ppo_runner_cfg,
  go1_original_student_distillation_runner_cfg,
  go1_original_teacher_ppo_runner_cfg,
)
from .teacher_student_runner import (
  Go1OriginalDistillationRunner,
  Go1OriginalRoughDistillationRunner,
)

TEACHER_TASK_ID = "Mjlab-Velocity-Flat-Unitree-Go1-Teacher"
STUDENT_TASK_ID = "Mjlab-Velocity-Flat-Unitree-Go1-Student"
ROUGH_TEACHER_TASK_ID = "Mjlab-Velocity-Rough-Unitree-Go1-Teacher"
ROUGH_STUDENT_TASK_ID = "Mjlab-Velocity-Rough-Unitree-Go1-Student"

register_mjlab_task(
  task_id=TEACHER_TASK_ID,
  env_cfg=go1_original_teacher_env_cfg(),
  play_env_cfg=go1_original_teacher_env_cfg(play=True),
  rl_cfg=go1_original_teacher_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=STUDENT_TASK_ID,
  env_cfg=go1_original_student_env_cfg(),
  play_env_cfg=go1_original_student_env_cfg(play=True),
  rl_cfg=go1_original_student_distillation_runner_cfg(),
  runner_cls=Go1OriginalDistillationRunner,
)

register_mjlab_task(
  task_id=ROUGH_TEACHER_TASK_ID,
  env_cfg=go1_original_rough_teacher_env_cfg(),
  play_env_cfg=go1_original_rough_teacher_env_cfg(play=True),
  rl_cfg=go1_original_rough_teacher_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=ROUGH_STUDENT_TASK_ID,
  env_cfg=go1_original_rough_student_env_cfg(),
  play_env_cfg=go1_original_rough_student_env_cfg(play=True),
  rl_cfg=go1_original_rough_student_distillation_runner_cfg(),
  runner_cls=Go1OriginalRoughDistillationRunner,
)

__all__ = [
  "ROUGH_STUDENT_TASK_ID",
  "ROUGH_TEACHER_TASK_ID",
  "STUDENT_TASK_ID",
  "TEACHER_TASK_ID",
]
