"""RL runner configuration for Quad Mini Tuned."""

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlBaseRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

QUAD_MINI_TUNED_TEACHER_EXPERIMENT = "quad_mini_tuned_teacher"


def quad_mini_tuned_ppo_runner_cfg(
  *,
  max_iterations: int = 10_000,
  experiment_name: str = "quad_mini_tuned_velocity",
) -> RslRlOnPolicyRunnerCfg:
  """Create the default PPO configuration for the Quad Mini task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
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
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=max_iterations,
    clip_actions=1.0,
  )


def _quad_policy_model_cfg() -> RslRlModelCfg:
  """Return the shared MLP shape used by teacher and student policies."""
  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


@dataclass
class QuadMiniTunedDistillationAlgorithmCfg:
  """RSL-RL behavior-cloning settings for the Quad Mini student."""

  class_name: str = "Distillation"
  num_learning_epochs: int = 1
  gradient_length: int = 15
  learning_rate: float = 1.0e-3
  max_grad_norm: float = 1.0
  loss_type: str = "huber"
  optimizer: str = "adam"


@dataclass
class QuadMiniTunedDistillationRunnerCfg(RslRlBaseRunnerCfg):
  """Runner configuration for the five-frame Quad Mini student."""

  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "student": ("actor",),
      "teacher": ("critic",),
    }
  )
  student: RslRlModelCfg = field(default_factory=_quad_policy_model_cfg)
  teacher: RslRlModelCfg = field(default_factory=_quad_policy_model_cfg)
  algorithm: QuadMiniTunedDistillationAlgorithmCfg = field(
    default_factory=QuadMiniTunedDistillationAlgorithmCfg
  )
  experiment_name: str = "quad_mini_tuned_student"
  teacher_checkpoint: str | None = None
  """Optional local PPO checkpoint used to initialize the teacher policy."""
  upload_model: bool = True


def quad_mini_tuned_distillation_runner_cfg(
  *,
  max_iterations: int = 5_000,
  experiment_name: str = "quad_mini_tuned_student",
) -> QuadMiniTunedDistillationRunnerCfg:
  """Create the behavior-cloning configuration for a Quad Mini student."""
  cfg = QuadMiniTunedDistillationRunnerCfg(
    max_iterations=max_iterations,
    experiment_name=experiment_name,
  )
  cfg.num_steps_per_env = 24
  cfg.save_interval = 50
  cfg.clip_actions = 1.0
  return cfg
