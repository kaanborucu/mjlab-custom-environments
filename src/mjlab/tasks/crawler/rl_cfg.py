"""RSL-RL PPO configuration for the crawler."""

from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import (
  RslRlBaseRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)
from mjlab.tasks.crawler.constants import PPO_ROLLOUT_STEPS

CRAWLER_ROUGH_TEACHER_EXPERIMENT = "crawler_3dof_rough_teacher"
CRAWLER_FLAT_TEACHER_EXPERIMENT = "crawler_3dof_flat_teacher"
_DISTILLATION_ACTION_STD = 1.0e-6


def crawler_ppo_runner_cfg(
  max_iterations: int = 600,
  experiment_name: str = "crawler_3dof_velocity",
) -> RslRlOnPolicyRunnerCfg:
  """Return the crawler PPO configuration."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.5,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    logger="tensorboard",
    upload_model=False,
    clip_actions=1.0,
    save_interval=50,
    num_steps_per_env=PPO_ROLLOUT_STEPS,
    max_iterations=max_iterations,
  )


def _crawler_policy_model_cfg() -> RslRlModelCfg:
  """Return the shared MLP shape used by the teacher and student."""
  return RslRlModelCfg(
    hidden_dims=(256, 128, 64),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.6,
      "std_type": "scalar",
    },
  )


def _crawler_student_policy_model_cfg() -> RslRlModelCfg:
  """Return a student policy with effectively deterministic rollouts."""
  cfg = _crawler_policy_model_cfg()
  assert cfg.distribution_cfg is not None
  cfg.distribution_cfg.update(
    {
      "init_std": _DISTILLATION_ACTION_STD,
      "std_range": (
        _DISTILLATION_ACTION_STD,
        _DISTILLATION_ACTION_STD,
      ),
      "learn_std": False,
    }
  )
  return cfg


@dataclass
class CrawlerDistillationAlgorithmCfg:
  """RSL-RL distillation algorithm settings."""

  class_name: str = "Distillation"
  num_learning_epochs: int = 1
  gradient_length: int = PPO_ROLLOUT_STEPS
  learning_rate: float = 1.0e-3
  max_grad_norm: float = 1.0
  loss_type: str = "huber"
  optimizer: str = "adam"


@dataclass
class CrawlerDistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Runner configuration for a crawler student."""

  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "student": ("actor",),
      "teacher": ("critic",),
    }
  )
  student: RslRlModelCfg = field(default_factory=_crawler_student_policy_model_cfg)
  teacher: RslRlModelCfg = field(default_factory=_crawler_policy_model_cfg)
  algorithm: CrawlerDistillationAlgorithmCfg = field(
    default_factory=CrawlerDistillationAlgorithmCfg
  )
  experiment_name: str = "crawler_3dof_rough_student"
  logger: Literal["wandb", "tensorboard"] = "tensorboard"
  upload_model: bool = False
  teacher_checkpoint: str | None = None
  """Optional local PPO checkpoint used to initialize the teacher policy."""
  teacher_experiment_name: str = CRAWLER_ROUGH_TEACHER_EXPERIMENT
  """Teacher experiment searched when no explicit checkpoint is provided."""


def crawler_distillation_runner_cfg(
  max_iterations: int = 5000,
  experiment_name: str = "crawler_3dof_rough_student",
  teacher_experiment_name: str = CRAWLER_ROUGH_TEACHER_EXPERIMENT,
) -> CrawlerDistillationRunnerCfg:
  """Return a crawler student distillation configuration."""
  cfg = CrawlerDistillationRunnerCfg(
    max_iterations=max_iterations,
    experiment_name=experiment_name,
    teacher_experiment_name=teacher_experiment_name,
  )
  cfg.num_steps_per_env = PPO_ROLLOUT_STEPS
  cfg.save_interval = 50
  cfg.clip_actions = 1.0
  return cfg
