"""PPO configuration for the original task with the current bird XML."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity.config.bird_5dof_original.rl_cfg import (
  bird_5dof_original_ppo_runner_cfg,
)


def bird_5dof_original_new_xml_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Keep the original PPO settings in a separate experiment directory."""
  cfg = bird_5dof_original_ppo_runner_cfg()
  cfg.experiment_name = "bird_5dof_velocity_original_new_xml"
  return cfg
