"""PPO configuration for the forward-flight bird task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.bird_5dof.rl_cfg import (
  bird_5dof_ppo_runner_cfg,
)


def bird_5dof_forward_3d_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create a separate PPO run for forward-biased 3D flight."""
  cfg = bird_5dof_ppo_runner_cfg()
  cfg.experiment_name = "bird_5dof_forward_3d"
  cfg.max_iterations = 1000
  return cfg
