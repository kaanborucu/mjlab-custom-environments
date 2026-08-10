"""Configuration for the five-DoF bird-inspired flapping-wing robot."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

BIRD_5DOF_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "bird_5dof" / "xmls" / "bird_5dof_crow.xml"
)
assert BIRD_5DOF_XML.exists()


def get_spec() -> mujoco.MjSpec:
  """Load a scene-attachable copy of the bird MJCF.

  MuJoCo does not propagate option fields while attaching an entity spec. The
  velocity task applies these values to the assembled scene instead.
  """
  spec = mujoco.MjSpec.from_file(str(BIRD_5DOF_XML))
  spec.option.density = 0.0
  spec.option.viscosity = 0.0
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  return spec


BIRD_5DOF_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 8.0),
  lin_vel=(0.0, 0.0, 0.0),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

BIRD_5DOF_JOINT_NAMES = (
  "left_flap",
  "left_feather",
  "right_flap",
  "right_feather",
  "tail_pitch",
)

BIRD_5DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=BIRD_5DOF_JOINT_NAMES),),
)

# Normalized policy actions map to the XML position-actuator control ranges.
BIRD_5DOF_ACTION_SCALE = {
  "left_flap": 1.10,
  "left_feather": 1.2217304764,
  "right_flap": 1.10,
  "right_feather": 1.2217304764,
  "tail_pitch": 0.6108652382,
}


def get_bird_5dof_robot_cfg() -> EntityCfg:
  """Return a fresh robot configuration."""
  return EntityCfg(
    init_state=BIRD_5DOF_INIT_STATE,
    spec_fn=get_spec,
    articulation=BIRD_5DOF_ARTICULATION,
  )
