"""Original forward-only bird task using a lightly adjusted original XML."""

import mujoco

from mjlab.asset_zoo.robots.bird_5dof_original.bird_constants import (
  BIRD_5DOF_ORIGINAL_INIT_STATE,
  get_bird_5dof_original_robot_cfg,
  get_original_spec,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.velocity.config.bird_5dof_original.env_cfgs import (
  bird_5dof_original_env_cfg,
)


def get_adjusted_new_xml_spec() -> mujoco.MjSpec:
  """Load the original XML, then apply only the requested scaling."""
  spec = get_original_spec()

  base = spec.body("bird")
  base.mass *= 0.8
  base.inertia = base.inertia * 0.8

  for name in ("left_wing_aero", "right_wing_aero"):
    wing = spec.geom(name)
    wing.size = wing.size * 1.1

  return spec


def bird_5dof_original_new_xml_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Use the original task with the adjusted new robot XML."""
  cfg = bird_5dof_original_env_cfg(play=play)
  robot_cfg = get_bird_5dof_original_robot_cfg()
  robot_cfg.spec_fn = get_adjusted_new_xml_spec
  robot_cfg.init_state = BIRD_5DOF_ORIGINAL_INIT_STATE
  cfg.scene.entities["robot"] = robot_cfg
  return cfg
