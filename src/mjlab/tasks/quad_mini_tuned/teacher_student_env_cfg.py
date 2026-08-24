"""Teacher and student configurations for the Quad Mini Tuned task."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)

from . import mdp
from .env_cfg import PRIVILEGED_CFG, quad_mini_tuned_env_cfg

_DOMAIN_RANDOMIZATION_STAGES: list[mdp.QuadDomainRandomizationStage] = [
  {"step": 0, "light": 0.0, "physical": 0.0},
  {"step": 12_000, "light": 0.25, "physical": 0.0},
  {"step": 36_000, "light": 0.50, "physical": 0.25},
  {"step": 72_000, "light": 0.75, "physical": 0.50},
  {"step": 120_000, "light": 1.0, "physical": 1.0},
]

_DOMAIN_RANDOMIZATION_EVENT_NEUTRALS: dict[
  str, dict[str, mdp.QuadRandomizationRange]
] = {
  "floor_friction": {"ranges": (1.0, 1.0)},
  "joint_friction": {"ranges": (1.0, 1.0)},
  "joint_armature": {"ranges": (1.0, 1.0)},
  "joint_damping": {"ranges": (1.0, 1.0)},
  "default_joint_pose": {"ranges": (0.0, 0.0)},
  "pd_gains": {"kp_range": (25.0, 25.0), "kd_range": (0.5, 0.5)},
  "effort_limits": {"effort_limit_range": (1.0, 1.0)},
  "foot_radius": {"ranges": (1.0, 1.0)},
  "encoder_bias": {"bias_range": (0.0, 0.0)},
  "torso_com": {"ranges": {0: (0.0, 0.0), 1: (0.0, 0.0), 2: (0.0, 0.0)}},
  "link_mass": {"ranges": (1.0, 1.0)},
  "torso_payload": {"ranges": (0.0, 0.0)},
  "body_inertia": {"ranges": (1.0, 1.0)},
}

_LIGHT_RANDOMIZATION_EVENTS = (
  "floor_friction",
  "joint_friction",
  "joint_armature",
  "joint_damping",
  "default_joint_pose",
  "pd_gains",
  "effort_limits",
  "foot_radius",
  "encoder_bias",
)
_PHYSICAL_RANDOMIZATION_EVENTS = (
  "torso_com",
  "link_mass",
  "torso_payload",
  "body_inertia",
)


def _configure_teacher_domain_randomization_curriculum(
  cfg: ManagerBasedRlEnvCfg,
) -> None:
  """Start the Teacher nominal and preserve its full DR ranges as targets."""
  event_targets: mdp.QuadRandomizationEventTargets = {
    "light": {},
    "physical": {},
  }
  for group_name, event_names in (
    ("light", _LIGHT_RANDOMIZATION_EVENTS),
    ("physical", _PHYSICAL_RANDOMIZATION_EVENTS),
  ):
    for event_name in event_names:
      event_cfg = cfg.events[event_name]
      param_targets: mdp.QuadRandomizationParamTargets = {}
      for param_name, neutral in _DOMAIN_RANDOMIZATION_EVENT_NEUTRALS[
        event_name
      ].items():
        target = deepcopy(event_cfg.params[param_name])
        param_targets[param_name] = (deepcopy(neutral), target)
        event_cfg.params[param_name] = deepcopy(neutral)
      event_targets[group_name][event_name] = param_targets

  root_velocity_range = cfg.events["reset_root_state"].params["velocity_range"]
  root_velocity_target = deepcopy(root_velocity_range["z"])
  root_velocity_range["z"] = (0.0, 0.0)
  joint_reset_cfg = cfg.events["reset_joint_state"]
  joint_position_target = deepcopy(joint_reset_cfg.params["position_range"])
  joint_reset_cfg.params["position_range"] = (0.0, 0.0)

  reset_targets: mdp.QuadResetRandomizationTargets = {
    "root_velocity_z": ((0.0, 0.0), root_velocity_target),
    "joint_position": ((0.0, 0.0), joint_position_target),
  }
  cfg.curriculum["domain_randomization"] = CurriculumTermCfg(
    func=mdp.quad_domain_randomization_curriculum,
    params={
      "stages": deepcopy(_DOMAIN_RANDOMIZATION_STAGES),
      "event_targets": event_targets,
      "reset_targets": reset_targets,
    },
  )


def _clean_privileged_observation_group() -> ObservationGroupCfg:
  """Build the clean five-frame privileged observation for teacher policies."""
  return ObservationGroupCfg(
    terms={
      "privileged_state": ObservationTermCfg(
        func=mdp.quad_privileged_observation,
        params={
          "command_name": "twist",
          "contact_sensor_name": "feet_ground_contact",
          "asset_cfg": PRIVILEGED_CFG,
        },
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=5,
    flatten_history_dim=True,
  )


def quad_mini_tuned_teacher_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create a clean five-frame privileged Quad Mini teacher configuration."""
  cfg = quad_mini_tuned_env_cfg(play=play)
  if not play:
    _configure_teacher_domain_randomization_curriculum(cfg)
  cfg.observations["actor"] = _clean_privileged_observation_group()
  cfg.observations["critic"] = _clean_privileged_observation_group()
  return cfg


def quad_mini_tuned_student_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the noisy five-frame student configuration."""
  cfg = quad_mini_tuned_env_cfg(play=play)
  cfg.observations["actor"].history_length = 5
  cfg.observations["actor"].flatten_history_dim = True
  cfg.observations["critic"] = _clean_privileged_observation_group()
  return cfg
