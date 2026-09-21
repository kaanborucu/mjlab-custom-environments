"""PPO configuration for the TONY5 V5 minimum-time task."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_ppo_cfg import (
  tony5_omni_v3_ppo_runner_cfg,
)


def tony5_position_min_time_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return a fresh PPO run configuration for the V5 observation layout."""
  cfg = tony5_omni_v3_ppo_runner_cfg()
  cfg.experiment_name = "tony5_position_min_time_v5"
  # At 100 Hz, 0.999 retains useful credit across this task's 10-second horizon.
  cfg.algorithm.gamma = 0.999
  cfg.algorithm.entropy_coef = 0.01
  cfg.resume = False
  cfg.load_run = ".*"
  return cfg


def tony5_position_min_time_teacher_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return the compact PPO configuration for the privileged V5 teacher."""
  cfg = tony5_position_min_time_ppo_runner_cfg()
  cfg.experiment_name = "tony5_position_min_time_v5_teacher"
  cfg.actor.hidden_dims = (256, 128, 64)
  cfg.critic.hidden_dims = (256, 128, 64)
  return cfg


__all__ = [
  "tony5_position_min_time_ppo_runner_cfg",
  "tony5_position_min_time_teacher_ppo_runner_cfg",
]
