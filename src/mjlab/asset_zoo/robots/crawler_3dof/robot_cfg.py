"""Robot constants and MJLab entity configuration."""

import math
from pathlib import Path

import mujoco

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

PACKAGE_DIR = Path(__file__).resolve().parent
ROBOT_XML = PACKAGE_DIR / "xmls" / "crawler_3dof.xml"
SOURCE_XML = PACKAGE_DIR / "source" / "three_dof_ground_crawler_ellipsoid_3d_full.xml"

JOINT_NAMES = ("joint_1", "joint_2", "joint_3")
ACTUATOR_NAMES = ("servo_1", "servo_2", "servo_3")
JOINT_LIMIT = 1.5707963268
ACTION_LIMIT = 1.4835298642
ACTION_SCALE = ACTION_LIMIT
PHYSICS_TIMESTEP = 0.002
DECIMATION = 10
POLICY_DELAY_S = 0.010
POLICY_DELAY_PHYSICS_STEPS = round(POLICY_DELAY_S / PHYSICS_TIMESTEP)
MAX_PLANAR_SPEED = 1.5
SEGMENT_RADIUS_M = 0.030
SEGMENT_HALF_LENGTH_M = 0.070
SERVO_MASS_KG = 0.063
SERVO_STALL_TORQUE_NM_7V4 = 3.491
SERVO_SPEED_SECONDS_PER_60_DEG_7V4 = 0.11
SERVO_NO_LOAD_SPEED_RAD_S_7V4 = math.radians(60.0) / SERVO_SPEED_SECONDS_PER_60_DEG_7V4
EXPECTED_ROBOT_MASS_KG = 0.409


def get_spec() -> mujoco.MjSpec:
  """Load a fresh robot-only MuJoCo specification."""
  spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
  # MJLab supplies the torque-speed-aware actuators below. Keep the XML
  # position servos only for standalone open-loop inspection/playback.
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  return spec


CRAWLER_CFG = EntityCfg(
  spec_fn=get_spec,
  articulation=EntityArticulationInfoCfg(
    actuators=(
      DcMotorActuatorCfg(
        target_names_expr=JOINT_NAMES,
        stiffness=35.0,
        damping=1.5,
        effort_limit=SERVO_STALL_TORQUE_NM_7V4,
        saturation_effort=SERVO_STALL_TORQUE_NM_7V4,
        velocity_limit=SERVO_NO_LOAD_SPEED_RAD_S_7V4,
        delay_min_lag=POLICY_DELAY_PHYSICS_STEPS,
        delay_max_lag=POLICY_DELAY_PHYSICS_STEPS,
      ),
    ),
    soft_joint_pos_limit_factor=0.95,
  ),
  init_state=EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.031),
    rot=(1.0, 0.0, 0.0, 0.0),
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    joint_pos={name: 0.0 for name in JOINT_NAMES},
    joint_vel={name: 0.0 for name in JOINT_NAMES},
  ),
)
