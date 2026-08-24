"""PPO configuration for the fixed omnidirectional TONY5 Aero task."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.manager_based.tony5.tony5_aero_ppo_cfg import (
  tony5_aero_ppo_runner_cfg,
)


def tony5_omni_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return V1 PPO settings with an experiment name unique to this task."""
  cfg = tony5_aero_ppo_runner_cfg()
  cfg.experiment_name = "tony5_velocity_aero_omni_v0"
  return cfg


__all__ = ["tony5_omni_ppo_runner_cfg"]
