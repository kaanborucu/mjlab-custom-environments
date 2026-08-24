"""V1-only PPO overrides for the TONY5 aerodynamic velocity task."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.manager_based.tony5.tony5_ppo_cfg import tony5_ppo_runner_cfg

TONY5_AERO_ENTROPY_COEF = 0.003
TONY5_AERO_ACTOR_INIT_STD = 0.6


def tony5_aero_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Return the V1 PPO config with learnable actor exploration."""
  cfg = tony5_ppo_runner_cfg(experiment_name="tony5_velocity_aero_v1")
  cfg.algorithm.entropy_coef = TONY5_AERO_ENTROPY_COEF
  assert cfg.actor.distribution_cfg is not None
  cfg.actor.distribution_cfg["init_std"] = TONY5_AERO_ACTOR_INIT_STD
  cfg.actor.distribution_cfg["learn_std"] = True
  cfg.actor.distribution_cfg.pop("std_range", None)
  return cfg


__all__ = [
  "TONY5_AERO_ACTOR_INIT_STD",
  "TONY5_AERO_ENTROPY_COEF",
  "tony5_aero_ppo_runner_cfg",
]
