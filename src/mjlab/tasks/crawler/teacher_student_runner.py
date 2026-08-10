"""RSL-RL runner glue for crawler teacher/student distillation."""

from typing import Any

from rsl_rl.env import VecEnv

from mjlab.rl import MjlabOnPolicyRunner


class CrawlerDistillationRunner(MjlabOnPolicyRunner):
  """Use MJLab checkpoint handling with RSL-RL's Distillation algorithm.

  The play script uses the generic ``actor`` load key. Distillation checkpoints
  call that model ``student``, so this adapter keeps play and hot-reload behavior
  compatible without changing the global runner or existing tasks.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    for key in ("student", "teacher"):
      if key in train_cfg:
        if train_cfg[key].get("cnn_cfg") is None:
          train_cfg[key].pop("cnn_cfg", None)
        if train_cfg[key].get("rnn_type") is None:
          for option in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            train_cfg[key].pop(option, None)
    super().__init__(env, train_cfg, log_dir, device)

  def load(
    self,
    path: str,
    load_cfg: dict[str, Any] | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
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
