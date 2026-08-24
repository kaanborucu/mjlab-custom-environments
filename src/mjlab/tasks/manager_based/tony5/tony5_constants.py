"""TONY5 V0 asset and physical constants."""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import (
  BuiltinDcMotorActuatorCfg,
  DcMotorInputMode,
  DcMotorPhysicalParams,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

TONY5_XML = Path(__file__).parent / "assets" / "tony5_v0_650.xml"

ROTOR_NAMES = ("fl", "fr", "rr", "rl")
ROTOR_BODIES = tuple(f"rotor_{name}" for name in ROTOR_NAMES)
ROTOR_JOINTS = tuple(f"rotor_{name}_joint" for name in ROTOR_NAMES)
ROTOR_SITES = tuple(f"rotor_{name}_site" for name in ROTOR_NAMES)

# The signs are the physical hinge-axis signs in the XML. Positive joint speed
# remains the controller convention for all four rotors.
ROTOR_AXIS_SIGNS = (-1.0, 1.0, -1.0, 1.0)

TOTAL_MASS = 0.285
BASE_MASS = 0.257
BASE_INERTIA = (4.2e-4, 4.4e-4, 7.9e-4)
ROTOR_MASS = 0.0070
ROTOR_INERTIA = (1.7e-6, 1.7e-6, 3.3e-6)
MOTOR_XY = 0.076014
MOTOR_RADIUS = 0.1075
PROP_RADIUS = 0.0635

# Estimated aerodynamic coefficients; replace after bench identification.
KT_THRUST = 1.00e-6
KQ_TORQUE = 1.25e-8

OMEGA_HOVER = 836.0
# Reachable V0 control ceiling with the modeled 22.2 V pack and propeller
# loading. Keep electrical voltage saturation independent of this action bound.
OMEGA_MAX = 2600.0

MOTOR_KV_RPM_PER_V = 1800.0
MOTOR_KT = 0.005305
MOTOR_KE = 0.005305
BATTERY_VOLTAGE = 22.2
MOTOR_TORQUE_LIMIT = 0.11

# Estimated / TODO: calibrate from real motor terminal-resistance measurements.
MOTOR_RESISTANCE = 0.50

# Voltage-space speed-loop gains. These values were tuned against the 1 kHz
# TONY5 motor/propeller model for a fast, well-damped speed transient.
KP_VELOCITY = 0.50
KI_VELOCITY = 0.80
KD_VELOCITY = 0.0001
INTEGRAL_LIMIT_VELOCITY = 800.0
DERIVATIVE_FILTER_TIME = 0.005

PHYSICS_DT = 0.001
DECIMATION = 10
POLICY_DT = PHYSICS_DT * DECIMATION


def get_tony5_spec() -> mujoco.MjSpec:
  """Load a fresh TONY5 XML spec and ensure MJLab owns the actuators."""
  if not TONY5_XML.exists():
    raise FileNotFoundError(f"TONY5 asset not found: {TONY5_XML}")
  spec = mujoco.MjSpec.from_file(str(TONY5_XML))
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  return spec


def get_tony5_robot_cfg() -> EntityCfg:
  """Return a fresh articulated TONY5 entity configuration."""
  motor_params = DcMotorPhysicalParams(
    kt=MOTOR_KT,
    ke=MOTOR_KE,
    resistance=MOTOR_RESISTANCE,
  )
  articulation = EntityArticulationInfoCfg(
    actuators=(
      BuiltinDcMotorActuatorCfg(
        target_names_expr=ROTOR_JOINTS,
        # The speed PID is computed in Tony5RotorSpeedAction. MuJoCo voltage
        # mode then applies the requested drive voltage through the physical
        # DC motor model, including back-EMF and torque saturation.
        mode=DcMotorInputMode.VOLTAGE,
        motor_params=motor_params,
        voltage_limit=BATTERY_VOLTAGE,
        effort_limit=MOTOR_TORQUE_LIMIT,
      ),
    ),
  )
  return EntityCfg(
    spec_fn=get_tony5_spec,
    articulation=articulation,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 1.5),
      rot=(1.0, 0.0, 0.0, 0.0),
      lin_vel=(0.0, 0.0, 0.0),
      ang_vel=(0.0, 0.0, 0.0),
      joint_pos={name: 0.0 for name in ROTOR_JOINTS},
      joint_vel={name: OMEGA_HOVER for name in ROTOR_JOINTS},
    ),
  )


__all__ = [
  "BASE_INERTIA",
  "BASE_MASS",
  "BATTERY_VOLTAGE",
  "DECIMATION",
  "DERIVATIVE_FILTER_TIME",
  "INTEGRAL_LIMIT_VELOCITY",
  "KD_VELOCITY",
  "KI_VELOCITY",
  "KP_VELOCITY",
  "KQ_TORQUE",
  "KT_THRUST",
  "MOTOR_KT",
  "MOTOR_KE",
  "MOTOR_KV_RPM_PER_V",
  "MOTOR_RESISTANCE",
  "MOTOR_TORQUE_LIMIT",
  "MOTOR_XY",
  "OMEGA_HOVER",
  "OMEGA_MAX",
  "PHYSICS_DT",
  "POLICY_DT",
  "PROP_RADIUS",
  "ROTOR_AXIS_SIGNS",
  "ROTOR_BODIES",
  "ROTOR_INERTIA",
  "ROTOR_JOINTS",
  "ROTOR_MASS",
  "ROTOR_NAMES",
  "ROTOR_SITES",
  "TOTAL_MASS",
  "TONY5_XML",
  "get_tony5_robot_cfg",
  "get_tony5_spec",
]
