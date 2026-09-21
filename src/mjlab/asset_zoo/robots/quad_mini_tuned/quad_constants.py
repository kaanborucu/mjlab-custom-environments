"""Configuration for the Quad Mini Tuned quadruped."""

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

QUAD_MINI_TUNED_XML = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "quad_mini_tuned"
  / "xmls"
  / "robot_feetonly_noground.xml"
)
assert QUAD_MINI_TUNED_XML.exists()

JOINT_NAMES = (
  "RF_HAA",
  "RF_HFE",
  "RF_KFE",
  "LF_HAA",
  "LF_HFE",
  "LF_KFE",
  "RH_HAA",
  "RH_HFE",
  "RH_KFE",
  "LH_HAA",
  "LH_HFE",
  "LH_KFE",
)

FEET_SITES = ("RF_FOOT", "LF_FOOT", "RH_FOOT", "LH_FOOT")
FEET_GEOMS = tuple(f"{name}_geom" for name in FEET_SITES)
ROOT_BODY = "trunk"
IMU_SITE = "imu_sensor_site"

DEFAULT_JOINT_POS = {
  "RF_HAA": 0.0,
  "RF_HFE": 0.4,
  "RF_KFE": -1.35,
  "LF_HAA": 0.0,
  "LF_HFE": -0.4,
  "LF_KFE": 1.35,
  "RH_HAA": 0.0,
  "RH_HFE": 0.4,
  "RH_KFE": -1.35,
  "LH_HAA": 0.0,
  "LH_HFE": -0.4,
  "LH_KFE": 1.35,
}


def get_spec() -> mujoco.MjSpec:
  """Load the feet-only XML and configure its native position actuators."""
  spec = mujoco.MjSpec.from_file(str(QUAD_MINI_TUNED_XML))
  # The source XML contains unnamed jointpos/jointvel sensors. MJLab wraps every
  # XML sensor by name, so those unnamed sensors are not useful here; joint data is
  # read directly from the entity instead.
  for sensor in list(spec.sensors):
    if not sensor.name:
      spec.delete(sensor)
  for key in list(spec.keys):
    spec.delete(key)
  # Keep the XML position actuators so MuJoCo integrates their PD forces
  # implicitly.  Computing the same PD torque in Python is unstable for this
  # robot's light lower legs at the 4 ms simulation timestep.
  for actuator in spec.actuators:
    actuator.gainprm[0] = 25.0
    actuator.biasprm[1] = -25.0
    actuator.biasprm[2] = -0.25
  return spec


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.22),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos=DEFAULT_JOINT_POS,
  joint_vel={".*": 0.0},
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(XmlActuatorCfg(target_names_expr=JOINT_NAMES, command_field="position"),),
  soft_joint_pos_limit_factor=0.95,
)


def get_quad_mini_tuned_robot_cfg() -> EntityCfg:
  """Return a fresh Quad Mini Tuned entity configuration."""
  return EntityCfg(
    init_state=INIT_STATE,
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


QUAD_MINI_TUNED_ACTION_SCALE = 0.6
QUAD_MINI_TUNED_EFFORT_LIMIT = 24.0
