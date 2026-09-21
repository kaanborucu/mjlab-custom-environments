"""RL runner configurations for the original Go1 teacher and student."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.quad_mini_tuned.rl_cfg import (
  QuadMiniTunedDistillationRunnerCfg,
  quad_mini_tuned_distillation_runner_cfg,
)
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg

GO1_ORIGINAL_TEACHER_EXPERIMENT = "go1_original_teacher"
GO1_ORIGINAL_ROUGH_TEACHER_EXPERIMENT = "go1_original_rough_teacher"


def go1_original_teacher_ppo_runner_cfg(
  *,
  max_iterations: int = 10_000,
) -> RslRlOnPolicyRunnerCfg:
  """Create the original Go1 PPO settings under a teacher run directory."""
  cfg = unitree_go1_ppo_runner_cfg()
  cfg.max_iterations = max_iterations
  cfg.experiment_name = GO1_ORIGINAL_TEACHER_EXPERIMENT
  return cfg


def go1_original_student_distillation_runner_cfg(
  *,
  max_iterations: int = 5_000,
) -> QuadMiniTunedDistillationRunnerCfg:
  """Create the student distillation settings with original Go1 MLP sizes."""
  cfg = quad_mini_tuned_distillation_runner_cfg(
    max_iterations=max_iterations,
    experiment_name="go1_original_student",
  )
  # The original Go1 teacher is trained without action clipping. Preserve the
  # same action semantics during distillation so the executed student action
  # matches the teacher target.
  cfg.clip_actions = None
  return cfg


def go1_original_rough_teacher_ppo_runner_cfg(
  *,
  max_iterations: int = 10_000,
) -> RslRlOnPolicyRunnerCfg:
  """Create the original rough Go1 teacher PPO settings."""
  cfg = unitree_go1_ppo_runner_cfg()
  cfg.max_iterations = max_iterations
  cfg.experiment_name = GO1_ORIGINAL_ROUGH_TEACHER_EXPERIMENT
  return cfg


def go1_original_rough_student_distillation_runner_cfg(
  *,
  max_iterations: int = 5_000,
) -> QuadMiniTunedDistillationRunnerCfg:
  """Create the original rough Go1 student distillation settings."""
  cfg = quad_mini_tuned_distillation_runner_cfg(
    max_iterations=max_iterations,
    experiment_name="go1_original_rough_student",
  )
  # Match the unclipped rough-teacher runner. Clipping only the student would
  # change most trajectories even when its raw actions imitate the teacher.
  cfg.clip_actions = None
  return cfg


__all__ = [
  "GO1_ORIGINAL_ROUGH_TEACHER_EXPERIMENT",
  "GO1_ORIGINAL_TEACHER_EXPERIMENT",
  "go1_original_rough_student_distillation_runner_cfg",
  "go1_original_rough_teacher_ppo_runner_cfg",
  "go1_original_student_distillation_runner_cfg",
  "go1_original_teacher_ppo_runner_cfg",
]
