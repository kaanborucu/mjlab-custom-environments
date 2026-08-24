"""PPO configuration for the TONY5 Omni V3 task."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.manager_based.tony5.tony5_omni_ppo_cfg import (
  tony5_omni_ppo_runner_cfg,
)


def tony5_omni_v3_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return V3 PPO settings with V3-only automatic checkpoint selection."""
  cfg = tony5_omni_ppo_runner_cfg()
  cfg.experiment_name = "tony5_velocity_aero_omni_v3"
  # V3 resume must never select the incompatible 110-input bootstrap run.
  cfg.load_run = r"\d{4}-\d{2}-\d{2}_.*"
  return cfg


__all__ = ["tony5_omni_v3_ppo_runner_cfg"]
