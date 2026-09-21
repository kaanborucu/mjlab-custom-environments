"""Tests for the original Go1 teacher and student task variants."""

import pytest

from mjlab.tasks.go1_original import ROUGH_STUDENT_TASK_ID, STUDENT_TASK_ID
from mjlab.tasks.go1_original.env_cfg import (
  STUDENT_HISTORY_LENGTH,
  STUDENT_HISTORY_TERMS,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


@pytest.mark.parametrize("task_id", [STUDENT_TASK_ID, ROUGH_STUDENT_TASK_ID])
def test_student_uses_selective_proprioceptive_history(task_id: str) -> None:
  cfg = load_env_cfg(task_id)
  actor = cfg.observations["actor"]
  critic = cfg.observations["critic"]

  assert actor.history_length is None
  assert critic.history_length == 1
  for name, term in actor.terms.items():
    expected_length = STUDENT_HISTORY_LENGTH if name in STUDENT_HISTORY_TERMS else 0
    assert term.history_length == expected_length

  assert actor.terms["command"].history_length == 0
  if "height_scan" in actor.terms:
    assert actor.terms["height_scan"].history_length == 0


@pytest.mark.parametrize("task_id", [STUDENT_TASK_ID, ROUGH_STUDENT_TASK_ID])
def test_student_preserves_teacher_action_range(task_id: str) -> None:
  cfg = load_rl_cfg(task_id)
  assert cfg.clip_actions is None
