"""Crawler-specific action processing."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class LowPassJointPositionActionCfg(JointPositionActionCfg):
  """Joint-position action with a first-order target low-pass filter."""

  new_target_weight: float = 0.80
  """Fraction of the new target used at each policy step."""

  def __post_init__(self) -> None:
    super().__post_init__()
    if not 0.0 < self.new_target_weight <= 1.0:
      raise ValueError("new_target_weight must be in the interval (0, 1].")

  def build(self, env: "ManagerBasedRlEnv") -> "LowPassJointPositionAction":
    return LowPassJointPositionAction(self, env)


class LowPassJointPositionAction(JointPositionAction):
  """Apply mildly filtered joint-position targets."""

  def __init__(
    self,
    cfg: LowPassJointPositionActionCfg,
    env: "ManagerBasedRlEnv",
  ):
    super().__init__(cfg, env)
    self._filtered_target = self._entity.data.default_joint_pos[
      :, self._target_ids
    ].clone()

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    cfg = cast(LowPassJointPositionActionCfg, self.cfg)
    self._filtered_target.lerp_(
      self._processed_actions,
      cfg.new_target_weight,
    )

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = self._filtered_target - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    super().reset(env_ids)
    self._filtered_target[env_ids] = self._entity.data.joint_pos[env_ids][
      :, self._target_ids
    ]
