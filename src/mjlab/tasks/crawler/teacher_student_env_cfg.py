"""Isolated teacher and student environment configurations."""

from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.tasks.crawler import mdp
from mjlab.tasks.crawler.env_cfg import crawler_env_cfg


def _privileged_observation_group(
  cfg: ManagerBasedRlEnvCfg,
  *,
  play: bool,
) -> ObservationGroupCfg:
  """Build the full-information group used by the teacher."""
  return ObservationGroupCfg(
    terms={
      "privileged_state": ObservationTermCfg(
        func=mdp.full_privileged_state,
        params={
          "command_name": "planar_velocity",
          "contact_sensor_name": "body_ground_contact",
        },
      )
    },
    concatenate_terms=True,
    history_length=10,
    flatten_history_dim=True,
    # The teacher term contains exact simulator state and has no injected noise.
    # Keep this enabled in training so the task has the same manager contract as
    # the other crawler tasks.
    enable_corruption=not play,
    nan_policy=cfg.observations["critic"].nan_policy,
  )


def crawler_teacher_env_cfg(
  play: bool = False,
  profile: Literal["foundation", "flat", "robust"] = "robust",
) -> ManagerBasedRlEnvCfg:
  """Create a privileged history-10 PPO teacher for the selected profile."""
  cfg = crawler_env_cfg(play=play, profile=profile)
  privileged_group = _privileged_observation_group(cfg, play=play)
  cfg.observations["actor"] = privileged_group
  cfg.observations["critic"] = _privileged_observation_group(cfg, play=play)
  cfg.observations["critic"].enable_corruption = False
  return cfg


def crawler_student_env_cfg(
  play: bool = False,
  profile: Literal["foundation", "flat", "robust"] = "robust",
) -> ManagerBasedRlEnvCfg:
  """Create an IMU/history-5 student with a history-10 privileged critic."""
  cfg = crawler_env_cfg(
    play=play,
    profile=profile,
    include_imu_in_actor=True,
    include_gravity_in_actor=False,
    actor_history_length=5,
    realistic_imu=True,
  )
  cfg.observations["critic"] = _privileged_observation_group(cfg, play=play)
  cfg.observations["critic"].enable_corruption = False
  return cfg
