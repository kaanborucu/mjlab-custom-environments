"""RSL-RL runner glue for Quad Mini teacher/student distillation."""

from pathlib import Path
from typing import Any

import torch
from rsl_rl.env import VecEnv

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.utils.os import get_checkpoint_path

from .rl_cfg import QUAD_MINI_TUNED_TEACHER_EXPERIMENT


def _get_latest_teacher_checkpoint(log_dir: str) -> Path:
  """Find the newest checkpoint in the latest local teacher run."""
  student_experiment_root = Path(log_dir).parent
  log_root = student_experiment_root.parent
  teacher_experiment_root = log_root / QUAD_MINI_TUNED_TEACHER_EXPERIMENT
  try:
    return get_checkpoint_path(
      teacher_experiment_root,
      run_dir=r".*",
      checkpoint=r"^model_\d+\.pt$",
    )
  except ValueError as exc:
    raise ValueError(
      "No Quad Mini teacher checkpoint was found. Train the teacher first or "
      "provide `--agent.teacher-checkpoint /path/to/model.pt`."
    ) from exc


class QuadMiniTunedTeacherRunner(VelocityOnPolicyRunner):
  """Restore the Teacher's DR stage before its first resumed rollout."""

  def load(
    self,
    path: str,
    load_cfg: dict[str, Any] | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    infos = super().load(
      path,
      load_cfg=load_cfg,
      strict=strict,
      map_location=map_location,
    )
    if "domain_randomization" in self.env.unwrapped.curriculum_manager.active_terms:
      # The vector wrapper performs its initial reset before load() restores
      # common_step_counter. Reset once more so the restored stage is applied.
      self.env.reset()
    return infos


class QuadMiniTunedDistillationRunner(MjlabOnPolicyRunner):
  """Keep distillation checkpoints compatible with MJLab play."""

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    teacher_checkpoint = train_cfg.pop("teacher_checkpoint", None)
    auto_selected = False
    if teacher_checkpoint is None and log_dir is not None:
      teacher_checkpoint = str(_get_latest_teacher_checkpoint(log_dir))
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
        f"[INFO] Loaded {selection} Quad Mini teacher checkpoint: {teacher_checkpoint}"
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
      load_cfg = {
        **load_cfg,
        "student": load_cfg["actor"],
      }
      load_cfg.pop("actor", None)
    return super().load(
      path,
      load_cfg=load_cfg,
      strict=strict,
      map_location=map_location,
    )
