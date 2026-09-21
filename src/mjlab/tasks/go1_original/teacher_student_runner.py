"""RSL-RL runner glue for original Go1 teacher/student distillation."""

from pathlib import Path
from typing import Any

import torch
from rsl_rl.env import VecEnv

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.utils.os import get_checkpoint_path

from .rl_cfg import (
  GO1_ORIGINAL_ROUGH_TEACHER_EXPERIMENT,
  GO1_ORIGINAL_TEACHER_EXPERIMENT,
)


def _get_latest_teacher_checkpoint(log_dir: str, experiment_name: str) -> Path:
  """Find the newest checkpoint in the latest original Go1 teacher run."""
  student_experiment_root = Path(log_dir).parent
  log_root = student_experiment_root.parent
  teacher_experiment_root = log_root / experiment_name
  try:
    return get_checkpoint_path(
      teacher_experiment_root,
      run_dir=r".*",
      checkpoint=r"^model_\d+\.pt$",
    )
  except ValueError as exc:
    raise ValueError(
      "No original Go1 teacher checkpoint was found. Train the teacher first "
      "or provide `--agent.teacher-checkpoint /path/to/model.pt`."
    ) from exc


class Go1OriginalDistillationRunner(MjlabOnPolicyRunner):
  """Distillation runner with automatic original Go1 teacher loading."""

  teacher_experiment = GO1_ORIGINAL_TEACHER_EXPERIMENT

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict[str, Any],
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    teacher_checkpoint = train_cfg.pop("teacher_checkpoint", None)
    auto_selected = False
    if teacher_checkpoint is None and log_dir is not None:
      teacher_checkpoint = str(
        _get_latest_teacher_checkpoint(log_dir, self.teacher_experiment)
      )
      auto_selected = True
    for key in ("student", "teacher"):
      if key in train_cfg:
        if train_cfg[key].get("cnn_cfg") is None:
          train_cfg[key].pop("cnn_cfg", None)
        if train_cfg[key].get("rnn_type") is None:
          for option in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(option, None)
    super().__init__(env, train_cfg, log_dir, device)
    if teacher_checkpoint is not None:
      checkpoint = torch.load(
        teacher_checkpoint, map_location=device, weights_only=False
      )
      self.alg.load(
        checkpoint,
        load_cfg={"teacher": True, "iteration": False},
        strict=True,
      )
      selection = "auto-selected" if auto_selected else "provided"
      print(
        f"[INFO] Loaded {selection} original Go1 teacher checkpoint: "
        f"{teacher_checkpoint}"
      )

  def load(
    self,
    path: str,
    load_cfg: dict[str, Any] | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Map the generic play actor key to the distillation student key."""
    if load_cfg is not None and "actor" in load_cfg:
      load_cfg = {**load_cfg, "student": load_cfg["actor"]}
      load_cfg.pop("actor", None)
    return super().load(
      path,
      load_cfg=load_cfg,
      strict=strict,
      map_location=map_location,
    )


class Go1OriginalRoughDistillationRunner(Go1OriginalDistillationRunner):
  """Distillation runner that loads the original rough Go1 teacher."""

  teacher_experiment = GO1_ORIGINAL_ROUGH_TEACHER_EXPERIMENT


__all__ = [
  "Go1OriginalDistillationRunner",
  "Go1OriginalRoughDistillationRunner",
]
