"""RSL-RL PPO configuration for TONY5 V0."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def tony5_ppo_runner_cfg(
  experiment_name: str = "tony5_position_yaw_v0",
) -> RslRlOnPolicyRunnerCfg:
  """Return the initial TONY5 PPO configuration."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.6,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      gamma=0.99,
      lam=0.95,
      clip_param=0.2,
      entropy_coef=0.01,
      learning_rate=3.0e-4,
      desired_kl=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      schedule="adaptive",
    ),
    experiment_name=experiment_name,
    logger="tensorboard",
    upload_model=False,
    clip_actions=1.0,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10000,
  )


__all__ = ["tony5_ppo_runner_cfg"]
