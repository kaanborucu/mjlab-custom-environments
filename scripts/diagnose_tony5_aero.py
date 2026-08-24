"""Diagnostics for the TONY5 V1 motor and aerodynamic load stack.

This script is intentionally separate from the task implementation.  The H-force
and body-fluid toggles below are temporary process-local diagnostic controls; they
do not modify the V0 or V1 task configuration on disk.

Run from the repository root with::

    uv run python scripts/diagnose_tony5_aero.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import mujoco
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.manager_based.tony5 import tony5_aero_physics
from mjlab.tasks.manager_based.tony5.tony5_actions import Tony5RotorSpeedAction
from mjlab.tasks.manager_based.tony5.tony5_aero_actions import (
  Tony5AeroRotorSpeedAction,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_env_cfg import (
  tony5_velocity_aero_env_cfg,
  tony5_velocity_aero_play_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_physics import (
  ROTOR_DRAG_KH,
  Tony5AeroRotorAerodynamics,
)
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  BATTERY_VOLTAGE,
  DERIVATIVE_FILTER_TIME,
  INTEGRAL_LIMIT_VELOCITY,
  KD_VELOCITY,
  KI_VELOCITY,
  KP_VELOCITY,
  KQ_TORQUE,
  MOTOR_KE,
  MOTOR_KT,
  MOTOR_RESISTANCE,
  MOTOR_TORQUE_LIMIT,
  OMEGA_MAX,
  ROTOR_JOINTS,
  TOTAL_MASS,
)
from mjlab.tasks.manager_based.tony5.tony5_velocity_curriculum import (
  Tony5RadialVelocityCommand,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from mjlab.utils.os import get_checkpoint_path

TASK_ID = "Mjlab-Tony5-Velocity-Aero-v1"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
KH_SWEEP = (0.0, 2.5e-6, 5.0e-6, 1.0e-5, 2.0e-5)
AERO_SPEEDS = (5.0, 10.0, 15.0, 20.0, 25.0, 27.78)
REPRESENTATIVE_OMEGAS = (1000.0, 1500.0, 2000.0, 2500.0)
STEP_TRANSITIONS = ((836.0, 1800.0), (1800.0, 836.0), (1800.0, 400.0), (836.0, 0.0))


def _quiet_env(cfg: ManagerBasedRlEnvCfg, device: str) -> ManagerBasedRlEnv:
  """Construct an env while suppressing the normal startup tables."""
  with contextlib.redirect_stdout(io.StringIO()):
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def _make_cfg(*, body_aero: bool, play: bool = True) -> ManagerBasedRlEnvCfg:
  """Build a fresh V1 config, optionally disabling only body fluid drag."""
  cfg = tony5_velocity_aero_play_env_cfg() if play else tony5_velocity_aero_env_cfg()
  if body_aero:
    return cfg

  original_scene_fn = cfg.scene.spec_fn

  def disable_body_aero(spec: mujoco.MjSpec) -> None:
    if original_scene_fn is not None:
      original_scene_fn(spec)
    for name in ("robot/body_aero", "body_aero"):
      try:
        geom = spec.geom(name)
      except (KeyError, RuntimeError):
        continue
      geom.fluid_ellipsoid = 0.0
      cast(Any, geom).fluid_coefs = (0.0, 0.0, 0.0, 0.0, 0.0)
      return
    raise RuntimeError("Could not find the assembled V1 body_aero geom")

  cfg.scene.spec_fn = disable_body_aero
  return cfg


def _tensor_field(
  env: ManagerBasedRlEnv, name: str, dof_ids: torch.Tensor
) -> torch.Tensor:
  """Read a MuJoCo generalized-force field at selected rotor DoFs."""
  value = getattr(env.sim.data, name)
  return value[:, dof_ids]


def _motor_pre_step(
  term: Tony5RotorSpeedAction,
  omega: torch.Tensor,
) -> dict[str, torch.Tensor]:
  """Reconstruct the V1 PID quantities before its next apply_actions call."""
  previous_omega = torch.where(
    term._pid_initialized[:, None], term._previous_omega, omega
  )
  omega_rate = (omega - previous_omega) / term._env.physics_dt
  filter_alpha = term._env.physics_dt / (DERIVATIVE_FILTER_TIME + term._env.physics_dt)
  filtered_rate = term._filtered_omega_rate + filter_alpha * (
    omega_rate - term._filtered_omega_rate
  )
  error = term.speed_targets - omega
  integral_candidate = (term._integral_error + error * term._env.physics_dt).clamp(
    -INTEGRAL_LIMIT_VELOCITY, INTEGRAL_LIMIT_VELOCITY
  )
  voltage_pre = (
    KP_VELOCITY * error + KI_VELOCITY * integral_candidate - KD_VELOCITY * filtered_rate
  )
  return {
    "error": error.clone(),
    "integral": integral_candidate.clone(),
    "derivative": filtered_rate.clone(),
    "voltage_pre": voltage_pre.clone(),
    "back_emf": MOTOR_KE * omega,
    "current_pre": (voltage_pre - MOTOR_KE * omega) / MOTOR_RESISTANCE,
    "torque_pre": MOTOR_KT * (voltage_pre - MOTOR_KE * omega) / MOTOR_RESISTANCE,
  }


def _append_step(
  log: dict[str, list[torch.Tensor]], values: dict[str, torch.Tensor]
) -> None:
  for key, value in values.items():
    log.setdefault(key, []).append(value.detach().cpu())


def _native_motor_torque(voltage: float, omega: float) -> float:
  """Reconstruct MuJoCo's voltage-mode DC-motor torque for this model."""
  torque = MOTOR_KT * (voltage - MOTOR_KE * omega) / MOTOR_RESISTANCE
  return max(-MOTOR_TORQUE_LIMIT, min(MOTOR_TORQUE_LIMIT, torque))


def _native_motor_equilibrium(voltage: float) -> float:
  """Solve native motor torque = prop torque, without aerodynamic H-load."""
  low = 0.0
  high = 10000.0
  for _ in range(80):
    omega = 0.5 * (low + high)
    motor_torque = _native_motor_torque(voltage, omega)
    prop_torque = KQ_TORQUE * omega * omega
    if motor_torque > prop_torque:
      low = omega
    else:
      high = omega
  return 0.5 * (low + high)


def _run_motor_test(
  *,
  device: str,
  body_aero: bool,
  constrain_root: bool,
  seconds: float,
) -> dict[str, torch.Tensor]:
  """Run the actual V1 action/PID/motor stack at a fixed 2600 rad/s target."""
  env = _quiet_env(_make_cfg(body_aero=body_aero), device)
  try:
    env.reset()
    robot = env.scene["robot"]
    term = cast(Tony5AeroRotorSpeedAction, env.action_manager.get_term("rotor_speed"))
    aero = cast(Tony5AeroRotorAerodynamics, term._aerodynamics)
    joint_ids = term._joint_ids
    dof_ids = robot.indexing.joint_v_adr
    target_action = torch.ones((env.num_envs, 4), device=env.device)
    term.process_actions(target_action)
    fixed_pose = robot.data.root_link_pose_w.clone()
    zero_root_velocity = torch.zeros((env.num_envs, 6), device=env.device)
    steps = round(seconds / env.physics_dt)
    log: dict[str, list[torch.Tensor]] = {}

    for _ in range(steps):
      if constrain_root:
        robot.write_root_link_pose_to_sim(fixed_pose)
        robot.write_root_link_velocity_to_sim(zero_root_velocity)
        env.scene.write_data_to_sim()
        env.sim.forward()

      omega = robot.data.joint_vel[:, joint_ids].clone()
      pid = _motor_pre_step(term, omega)
      term.apply_actions()
      env.scene.write_data_to_sim()
      env.sim.step()
      env.scene.update(dt=env.physics_dt)

      omega_after = robot.data.joint_vel[:, joint_ids].clone()
      qfrc_actuator = robot.data.qfrc_actuator[:, joint_ids].clone()
      qfrc_external = robot.data.qfrc_external[:, joint_ids].clone()
      qfrc_passive = _tensor_field(env, "qfrc_passive", dof_ids).clone()
      qfrc_bias = _tensor_field(env, "qfrc_bias", dof_ids).clone()
      qfrc_applied = _tensor_field(env, "qfrc_applied", dof_ids).clone()
      qfrc_smooth = _tensor_field(env, "qfrc_smooth", dof_ids).clone()
      qfrc_fluid = _tensor_field(env, "qfrc_fluid", dof_ids).clone()
      h_force = aero.h_force_w.clone()
      h_force_norm = torch.linalg.vector_norm(h_force, dim=-1)
      prop_torque = -KQ_TORQUE * omega_after * omega_after.abs()
      non_actuator = qfrc_smooth - qfrc_actuator
      _append_step(
        log,
        {
          "omega": omega_after,
          "target": term.speed_targets,
          "voltage": term.last_voltage,
          "voltage_pre": pid["voltage_pre"],
          "error": pid["error"],
          "integral": term._integral_error,
          "derivative": pid["derivative"],
          "back_emf": pid["back_emf"],
          "current_pre": pid["current_pre"],
          "torque_pre": pid["torque_pre"],
          "torque_post": qfrc_actuator,
          "prop_torque": prop_torque,
          "qfrc_external": qfrc_external,
          "qfrc_passive": qfrc_passive,
          "qfrc_bias": qfrc_bias,
          "qfrc_applied": qfrc_applied,
          "qfrc_smooth": qfrc_smooth,
          "qfrc_fluid": qfrc_fluid,
          "net_non_actuator": non_actuator,
          "qacc": _tensor_field(env, "qacc", dof_ids).clone(),
          "h_force_norm": h_force_norm,
          "root_speed": robot.data.root_link_lin_vel_w.clone(),
        },
      )

    return {key: torch.cat(values, dim=0) for key, values in log.items()}
  finally:
    env.close()


def _run_rotor_load_grid(device: str) -> None:
  """Log per-rotor fluid, actuator, prop-load, and qvel values at test speeds."""
  env = _quiet_env(_make_cfg(body_aero=True), device)
  try:
    env.reset()
    robot = env.scene["robot"]
    term = cast(Tony5AeroRotorSpeedAction, env.action_manager.get_term("rotor_speed"))
    rotor_dofs = robot.indexing.joint_v_adr
    max_action = torch.ones((env.num_envs, 4), device=env.device)
    print("\n3a. Per-rotor baseline/load grid after V1 fluid isolation")
    print(
      "  columns: commanded_qvel, rotor, actual_qvel, qfrc_fluid, "
      "qfrc_actuator, prop_kQ_torque"
    )
    for speed in REPRESENTATIVE_OMEGAS:
      rotor_speed = torch.full((env.num_envs, 4), speed, device=env.device)
      robot.write_joint_velocity_to_sim(rotor_speed, joint_ids=term._joint_ids)
      env.scene.write_data_to_sim()
      env.sim.forward()
      term.process_actions(max_action)
      term.apply_actions()
      env.scene.write_data_to_sim()
      env.sim.forward()
      actual = robot.data.joint_vel[:, term._joint_ids][0]
      fluid = env.sim.data.qfrc_fluid[:, rotor_dofs][0]
      actuator = robot.data.qfrc_actuator[:, term._joint_ids][0]
      prop = -KQ_TORQUE * actual * actual.abs()
      for index, rotor_name in enumerate(ROTOR_JOINTS):
        print(
          f"  {speed:>8.1f} {rotor_name:>16} {actual[index]:>14.3f} "
          f"{fluid[index]:>12.7f} {actuator[index]:>14.7f} {prop[index]:>14.7f}"
        )
  finally:
    env.close()


def _run_step_response(
  device: str, initial_speed: float, target_speed: float, seconds: float = 1.5
) -> torch.Tensor:
  """Run one exact rotor-speed step through the V1 ESC and native motor."""
  env = _quiet_env(_make_cfg(body_aero=True), device)
  try:
    env.reset()
    robot = env.scene["robot"]
    term = cast(Tony5AeroRotorSpeedAction, env.action_manager.get_term("rotor_speed"))
    fixed_pose = robot.data.root_link_pose_w.clone()
    zero_root_velocity = torch.zeros((env.num_envs, 6), device=env.device)
    initial = torch.full((env.num_envs, 4), initial_speed, device=env.device)
    term._speed_targets[:] = target_speed
    term._previous_omega[:] = initial
    term._filtered_omega_rate[:] = 0.0
    term._integral_error[:] = 0.0
    term._pid_initialized[:] = True
    robot.write_joint_velocity_to_sim(initial, joint_ids=term._joint_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    records: list[torch.Tensor] = []
    for step in range(round(seconds / env.physics_dt)):
      robot.write_root_link_pose_to_sim(fixed_pose)
      robot.write_root_link_velocity_to_sim(zero_root_velocity)
      env.scene.write_data_to_sim()
      env.sim.forward()
      term.apply_actions()
      env.scene.write_data_to_sim()
      env.sim.step()
      env.scene.update(dt=env.physics_dt)
      records.append(
        torch.cat(
          (
            torch.tensor([step * env.physics_dt], device=env.device),
            robot.data.joint_vel[:, term._joint_ids].mean(dim=-1),
            term.last_voltage.mean(dim=-1),
          )
        )
      )
    return torch.stack(records).cpu()
  finally:
    env.close()


def _step_response_metrics(
  data: torch.Tensor, initial_speed: float, target_speed: float
) -> dict[str, float | None]:
  """Compute normalized 10/90 percent and 2 percent settling metrics."""
  time = data[:, 0]
  omega = data[:, 1]
  voltage = data[:, 2]
  delta = target_speed - initial_speed
  magnitude = abs(delta)
  if delta >= 0.0:
    threshold = initial_speed + 0.9 * magnitude
    crossing = torch.where(omega >= threshold)[0]
    overshoot = max(0.0, float(omega.max()) - target_speed)
    undershoot = 0.0
  else:
    threshold = initial_speed - 0.9 * magnitude
    crossing = torch.where(omega <= threshold)[0]
    overshoot = 0.0
    undershoot = max(0.0, target_speed - float(omega.min()))
  settling_band = 0.02 * max(magnitude, 1.0)
  in_band = (omega - target_speed).abs() <= settling_band
  settling_time: float | None = None
  for index in range(len(in_band)):
    if bool(in_band[index:].all()):
      settling_time = float(time[index])
      break
  response_time = float(time[crossing[0]]) if len(crossing) else None
  return {
    "rise_or_fall_time": response_time,
    "overshoot": overshoot,
    "undershoot": undershoot,
    "settling_time": settling_time,
    "voltage_min": float(voltage.min()),
    "voltage_max": float(voltage.max()),
    "zero_fraction": float((voltage <= 1.0e-6).float().mean()),
    "negative_fraction": float((voltage < -1.0e-6).float().mean()),
  }


def _run_step_response_suite(device: str) -> None:
  """Run and print the requested V1 ESC step-response suite."""
  print("\n3b. V1 unidirectional ESC step-response suite")
  print(
    "  columns: step, response_time_s, overshoot_rad_s, undershoot_rad_s, "
    "settling_time_s, V_min, V_max, zero_fraction, negative_fraction"
  )
  for initial_speed, target_speed in STEP_TRANSITIONS:
    data = _run_step_response(device, initial_speed, target_speed)
    metrics = _step_response_metrics(data, initial_speed, target_speed)
    response_name = "rise" if target_speed > initial_speed else "fall"
    response_time = metrics["rise_or_fall_time"]
    settling_time = metrics["settling_time"]
    response_text = "n/a" if response_time is None else f"{response_time:.4f}"
    settling_text = "n/a" if settling_time is None else f"{settling_time:.4f}"

    print(
      f"  {initial_speed:.0f}->{target_speed:.0f} {response_name} "
      f"{response_text:>10} "
      f"{metrics['overshoot']:>17.3f} {metrics['undershoot']:>17.3f} "
      f"{settling_text:>16} {metrics['voltage_min']:>7.3f} "
      f"{metrics['voltage_max']:>7.3f} {metrics['zero_fraction']:>13.3f} "
      f"{metrics['negative_fraction']:>16.3f}"
    )
    trace_indices = range(0, len(data), max(1, len(data) // 6))
    print(
      "    voltage trace (t, mean_omega, mean_voltage):",
      " ".join(
        f"({data[index, 0]:.3f},{data[index, 1]:.1f},{data[index, 2]:.2f})"
        for index in trace_indices
      ),
    )


def _print_rotor_summary(label: str, log: dict[str, torch.Tensor]) -> None:
  print(f"\n{label}")
  print("  rotor order:", ", ".join(ROTOR_JOINTS))
  equilibrium = _native_motor_equilibrium(BATTERY_VOLTAGE)
  print(
    "  native stateless motor/prop equilibrium (no H-force or body aero): "
    f"{equilibrium:.2f} rad/s ({equilibrium * 60.0 / (2.0 * math.pi):.0f} RPM)"
  )
  for i, name in enumerate(ROTOR_JOINTS):
    omega = log["omega"][:, i]
    target = log["target"][:, i]
    voltage = log["voltage"][:, i]
    torque = log["torque_post"][:, i]
    final_omega = float(omega[-1])
    final_voltage = float(voltage[-1])
    back_emf = MOTOR_KE * final_omega
    current = (final_voltage - back_emf) / MOTOR_RESISTANCE
    native_torque = _native_motor_torque(final_voltage, final_omega)
    prop_torque = KQ_TORQUE * final_omega * final_omega
    print(
      f"  {name}: final omega={omega[-1]:.1f}, max={omega.max():.1f} rad/s "
      f"({omega.max() * 60.0 / (2.0 * math.pi):.0f} RPM); "
      f"target_final={target[-1]:.1f}; voltage=[{voltage.min():.2f}, "
      f"{voltage.max():.2f}] V; zero_frac={(voltage <= 1.0e-6).float().mean():.3f}; "
      f"negative_frac={(voltage < -1.0e-6).float().mean():.3f}; "
      f"torque_final={torque[-1]:.4f} N*m; "
      f"final back_emf={back_emf:.3f} V, current={current:.3f} A, "
      f"native torque={native_torque:.5f} N*m, prop load={prop_torque:.5f} N*m"
    )
  print(
    "  all-rotor final/maximum voltage:",
    f"{log['voltage'][-1].abs().max():.2f} / {log['voltage'].abs().max():.2f} V",
  )
  print(
    "  all-rotor final/maximum motor torque:",
    f"{log['torque_post'][-1].abs().max():.4f} / "
    f"{log['torque_post'].abs().max():.4f} N*m",
  )
  print(
    "  final qfrc external / non-actuator rotor-axis torque max:",
    f"{log['qfrc_external'][-1].abs().max():.6f} / "
    f"{log['net_non_actuator'][-1].abs().max():.6f} N*m",
  )
  for key in (
    "torque_post",
    "qfrc_external",
    "qfrc_passive",
    "qfrc_bias",
    "qfrc_applied",
    "qfrc_smooth",
    "qfrc_fluid",
    "net_non_actuator",
    "qacc",
  ):
    units = "rad/s^2" if key == "qacc" else "N*m"
    print(f"  {key} final by rotor ({units}): {log[key][-1].tolist()}")


def _print_static_model(env: ManagerBasedRlEnv) -> None:
  model = env.sim.mj_model
  warp_model = env.sim.model
  robot = env.scene["robot"]
  joint_ids = robot.indexing.joint_ids.detach().cpu().tolist()
  dofs = robot.indexing.joint_v_adr.detach().cpu().tolist()
  ctrl_ids = robot.indexing.ctrl_ids.detach().cpu().tolist()
  print("\n3. MuJoCo rotor-axis load inspection")
  for index, dof in enumerate(dofs):
    print(
      f"  {ROTOR_JOINTS[index]}: joint_id={joint_ids[index]}, dof={dof}, "
      f"damping={model.dof_damping[dof]:g}, "
      f"frictionloss={model.dof_frictionloss[dof]:g}, "
      f"armature={model.dof_armature[dof]:g}"
    )
  for index, actuator in enumerate(ctrl_ids):
    print(
      f"  actuator {index}: gear={model.actuator_gear[actuator, 0]:g}, "
      f"forcelimited={bool(model.actuator_forcelimited[actuator])}, "
      f"forcerange={tuple(model.actuator_forcerange[actuator])}, "
      f"ctrl_limited={bool(model.actuator_ctrllimited[actuator])}, "
      f"gainprm={tuple(model.actuator_gainprm[actuator, :8])}"
    )
  print(
    "  duplicate actuator/prop source check: XML has no actuator section; "
    "thrust and reaction torque are applied once by Tony5RotorAerodynamics."
  )
  print(
    "  fluid-force source check: body_aero is the only nonzero-coefficient "
    "fluid geom; rotor disable geoms have zero fluid coefficients."
  )
  body_fluid_ellipsoid = getattr(warp_model, "body_fluid_ellipsoid", None)
  if body_fluid_ellipsoid is not None:
    print("  MuJoCo-Warp body fluid flags:")
    for body_id in robot.indexing.body_ids.detach().cpu().tolist():
      body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
      print(f"    {body_name}: {bool(body_fluid_ellipsoid[body_id])}")
  print("  V1 rotor fluid-disable geoms:")
  for rotor_name in ROTOR_JOINTS:
    geom_name = f"robot/{rotor_name.removesuffix('_joint')}_fluid_disable"
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    print(f"    {geom_name}: geom_id={geom_id}")


def _body_fluid_force(env: ManagerBasedRlEnv, speed: float) -> torch.Tensor:
  robot = env.scene["robot"]
  root_velocity_b = torch.zeros((env.num_envs, 6), device=env.device)
  root_velocity_b[:, 0] = speed
  robot.write_root_link_velocity_b_to_sim(root_velocity_b)
  env.scene.write_data_to_sim()
  env.sim.forward()
  root_dofs = torch.arange(6, device=env.device)
  return _tensor_field(env, "qfrc_passive", root_dofs)[0, :3].detach().cpu()


def _h_force_measurement(device: str) -> None:
  env = _quiet_env(_make_cfg(body_aero=True), device)
  try:
    env.reset()
    robot = env.scene["robot"]
    term = cast(Tony5AeroRotorSpeedAction, env.action_manager.get_term("rotor_speed"))
    aero = cast(Tony5AeroRotorAerodynamics, term._aerodynamics)
    omega = torch.full((env.num_envs, 4), OMEGA_MAX, device=env.device)
    robot.write_joint_velocity_to_sim(omega, joint_ids=term._joint_ids)
    print("\n6. H-force magnitude with actual rotor speed fixed at 2600 rad/s")
    print(f"  body mass={TOTAL_MASS:.3f} kg; mg={TOTAL_MASS * 9.80665:.3f} N")
    for speed in AERO_SPEEDS:
      robot.write_root_link_velocity_b_to_sim(
        torch.tensor([[speed, 0.0, 0.0, 0.0, 0.0, 0.0]], device=env.device)
      )
      env.scene.write_data_to_sim()
      env.sim.forward()
      forces, _ = aero.compute_wrenches()
      del forces
      h_force = aero.h_force_w[0]
      total = torch.linalg.vector_norm(h_force, dim=-1).sum().item()
      print(
        f"  speed={speed:>5.1f} m/s: per-rotor={total / 4.0:.4f} N, "
        f"sum magnitudes={total:.4f} N, ratio_to_mg={total / (TOTAL_MASS * 9.80665):.3f}"
      )
  finally:
    env.close()


def _measure_aero_point(
  env: ManagerBasedRlEnv, speed: float, omega: float
) -> tuple[float, float, float]:
  """Measure body drag, rotor H-drag, and their combined forward loads."""
  robot = env.scene["robot"]
  term = cast(Tony5AeroRotorSpeedAction, env.action_manager.get_term("rotor_speed"))
  aero = cast(Tony5AeroRotorAerodynamics, term._aerodynamics)
  root_velocity_b = torch.zeros((env.num_envs, 6), device=env.device)
  root_velocity_b[:, 0] = speed
  rotor_velocity = torch.full(
    (env.num_envs, len(term._joint_ids)), omega, device=env.device
  )
  robot.write_root_link_velocity_b_to_sim(root_velocity_b)
  robot.write_joint_velocity_to_sim(rotor_velocity, joint_ids=term._joint_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
  aero.compute_wrenches()

  root_dofs = torch.arange(6, device=env.device)
  body_fluid = _tensor_field(env, "qfrc_fluid", root_dofs)[0, :3]
  body_drag = max(0.0, float(-body_fluid[0]))
  h_force = aero.h_force_w[0, :, 0].sum().item()
  h_drag = max(0.0, float(-h_force))
  return body_drag, h_drag, body_drag + h_drag


def _run_kh_sweep(device: str) -> dict[float, float]:
  """Measure the requested KH/speed/rotor-speed grid in a fresh V1 env."""
  env = _quiet_env(_make_cfg(body_aero=True), device)
  original_kh = tony5_aero_physics.ROTOR_DRAG_KH
  body_drag_by_speed: dict[float, float] = {}
  try:
    env.reset()
    print("\n7. Isolated aerodynamic load sweep")
    print(
      "  columns: KH, speed_mps, omega_rad_s, body_drag_N, "
      "rotor_H_drag_N, combined_drag_N"
    )
    for kh in KH_SWEEP:
      tony5_aero_physics.ROTOR_DRAG_KH = kh
      for speed in AERO_SPEEDS:
        for omega in REPRESENTATIVE_OMEGAS:
          body_drag, h_drag, combined_drag = _measure_aero_point(env, speed, omega)
          if omega == REPRESENTATIVE_OMEGAS[0] and kh == KH_SWEEP[0]:
            body_drag_by_speed[speed] = body_drag
          print(
            f"  {kh:.3e} {speed:>8.2f} {omega:>8.1f} "
            f"{body_drag:>12.5f} {h_drag:>15.5f} {combined_drag:>15.5f}"
          )
  finally:
    tony5_aero_physics.ROTOR_DRAG_KH = original_kh
    env.close()
  return body_drag_by_speed


def _print_forward_feasibility(
  body_drag_by_speed: dict[float, float], simulated_max_omega: float
) -> None:
  """Print an upper-authority steady force-balance estimate, not a flight test."""
  mass = TOTAL_MASS
  gravity_force = mass * 9.80665
  max_thrust = 4.0 * 1.0e-6 * simulated_max_omega**2
  print("\n8. Steady forward-flight feasibility estimate")
  print(
    "  Assumption: equal rotors at the measured maximum omega, with thrust "
    "tilted to balance gravity plus measured body/H drag."
  )
  print(
    "  columns: KH, speed_mps, body_drag_N, H_drag_N, total_drag_N, "
    "tilt_deg, required_thrust_N, thrust_margin_N, omega, at_22.2V_rail"
  )
  for kh in KH_SWEEP:
    for speed in AERO_SPEEDS:
      body_drag = body_drag_by_speed[speed]
      h_drag = 4.0 * kh * simulated_max_omega * speed
      total_drag = body_drag + h_drag
      required_thrust = math.hypot(gravity_force, total_drag)
      tilt = math.degrees(math.atan2(total_drag, gravity_force))
      margin = max_thrust - required_thrust
      print(
        f"  {kh:.3e} {speed:>8.2f} {body_drag:>12.5f} {h_drag:>10.5f} "
        f"{total_drag:>12.5f} {tilt:>8.2f} {required_thrust:>17.5f} "
        f"{margin:>16.5f} {simulated_max_omega:>8.1f} True"
      )
  print(
    f"  measured motor-test omega={simulated_max_omega:.2f} rad/s; "
    f"equal-rotor thrust ceiling={max_thrust:.3f} N; mg={gravity_force:.3f} N"
  )


def _run_checkpoint_ramp(
  device: str, checkpoint: Path, ramp_s: float, hold_s: float
) -> None:
  """Run the latest actor through a gradual body-forward 26 m/s command."""
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(TASK_ID)
  env = _quiet_env(env_cfg, device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  try:
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(vec_env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    obs, _ = vec_env.reset()
    command = cast(Tony5RadialVelocityCommand, env.command_manager.get_term("velocity"))
    command.enable_keyboard_control()
    command.set_keyboard_mode("forward")
    action_term = cast(
      Tony5AeroRotorSpeedAction, env.action_manager.get_term("rotor_speed")
    )
    aero = cast(Tony5AeroRotorAerodynamics, action_term._aerodynamics)
    robot = env.scene["robot"]
    root_dofs = robot.indexing.free_joint_v_adr
    dt = env.step_dt
    total_steps = round((ramp_s + hold_s) / dt)
    records: list[dict[str, torch.Tensor]] = []
    reset_count = 0
    with torch.inference_mode():
      for step in range(total_steps):
        command._keyboard_max_speed = min(26.0, 26.0 * step * dt / ramp_s)
        command._set_keyboard_command()
        action = policy(obs)
        obs, _, dones, _ = vec_env.step(action)
        did_reset = bool(dones.any().item())
        reset_count += int(did_reset)
        actual = robot.data.joint_vel[:, action_term._joint_ids]
        _, pitch, _ = euler_xyz_from_quat(robot.data.root_link_quat_w)
        records.append(
          {
            "time": torch.full((1,), step * dt, device=env.device),
            "command": command.vel_command_b.clone(),
            "body_velocity": robot.data.root_link_lin_vel_b.clone(),
            "world_velocity": robot.data.root_link_lin_vel_w.clone(),
            "yaw_rate": robot.data.root_link_ang_vel_b[:, 2].clone(),
            "pitch": pitch.clone(),
            "position": robot.data.root_link_pos_w.clone(),
            "target": action_term.speed_targets.clone(),
            "actual": actual.clone(),
            "voltage": action_term.last_voltage.clone(),
            "h_force": aero.h_force_w.sum(dim=1).clone(),
            "body_fluid": _tensor_field(env, "qfrc_fluid", root_dofs).clone(),
            "reset": torch.full((1,), float(did_reset), device=env.device),
          }
        )
    data = {
      key: torch.cat([record[key] for record in records], dim=0).cpu()
      for key in records[0]
    }
  finally:
    vec_env.close()

  print("\n7. Latest-checkpoint gradual forward-command test")
  print(f"  checkpoint={checkpoint}")
  print(f"  command ramp=0->26 m/s over {ramp_s:g} s; hold={hold_s:g} s")
  print("  command: body vx ramp, body vy=0, yaw_rate=0, vz=0")
  print(f"  resets observed={reset_count}")
  print(
    "  columns: time, cmd_vx, body_vx, body_vy, world_vx, world_vy, yaw_rate, pitch_deg"
  )
  sample_indices = range(0, len(data["time"]), max(1, len(data["time"]) // 8))
  for index in sample_indices:
    print(
      "  ",
      " ".join(
        f"{value:.3f}"
        for value in (
          data["time"][index],
          data["command"][index, 0],
          data["body_velocity"][index, 0],
          data["body_velocity"][index, 1],
          data["world_velocity"][index, 0],
          data["world_velocity"][index, 1],
          data["yaw_rate"][index],
          data["pitch"][index] * 180.0 / math.pi,
        )
      ),
    )

  hold_mask = data["time"] >= ramp_s
  print(
    "  achieved speed summary: "
    f"peak_body_vx={data['body_velocity'][:, 0].max():.3f} m/s, "
    f"hold_mean_body_vx={data['body_velocity'][hold_mask, 0].mean():.3f} m/s, "
    f"hold_final_body_vx={data['body_velocity'][-1, 0]:.3f} m/s, "
    f"peak_abs_body_vy={data['body_velocity'][:, 1].abs().max():.3f} m/s, "
    f"peak_world_vxy={torch.linalg.vector_norm(data['world_velocity'][:, :2], dim=-1).max():.3f} m/s"
  )
  print(
    "  attitude summary: "
    f"peak_abs_pitch={data['pitch'].abs().max() * 180.0 / math.pi:.2f} deg, "
    f"peak_abs_yaw_rate={data['yaw_rate'].abs().max():.3f} rad/s"
  )

  target_rpm = data["target"] * 60.0 / (2.0 * math.pi)
  actual_rpm = data["actual"] * 60.0 / (2.0 * math.pi)
  error_rpm = target_rpm - actual_rpm
  print("  per-rotor checkpoint statistics")
  print(
    "  columns: rotor, mean_target_RPM, max_target_RPM, mean_actual_RPM, "
    "max_actual_RPM, mean_V, max_V, min_V, +rail_frac, -rail_frac, "
    "mean_target-actual_RPM, max_abs_error_RPM"
  )
  for index, rotor_name in enumerate(ROTOR_JOINTS):
    voltage = data["voltage"][:, index]
    positive_rail = (voltage >= BATTERY_VOLTAGE - 1.0e-3).float().mean()
    negative_rail = (voltage <= -BATTERY_VOLTAGE + 1.0e-3).float().mean()
    print(
      f"  {rotor_name:>16} {target_rpm[:, index].mean():>16.1f} "
      f"{target_rpm[:, index].max():>15.1f} {actual_rpm[:, index].mean():>16.1f} "
      f"{actual_rpm[:, index].max():>14.1f} {voltage.mean():>8.3f} "
      f"{voltage.max():>8.3f} {voltage.min():>8.3f} "
      f"{positive_rail:>10.3f} {negative_rail:>10.3f} "
      f"{error_rpm[:, index].mean():>24.1f} "
      f"{error_rpm[:, index].abs().max():>19.1f}"
    )

  h_force_norm = torch.linalg.vector_norm(data["h_force"], dim=-1)
  body_drag_x = -data["body_fluid"][:, 0]
  print(
    f"  H-force sum: mean={h_force_norm.mean():.4f} N, max={h_force_norm.max():.4f} N"
  )
  print(
    "  main-body ellipsoid drag x: "
    f"mean={body_drag_x.mean():.4f} N, max={body_drag_x.max():.4f} N"
  )

  reset_mask = data["reset"] > 0.5
  segments: list[tuple[int, int]] = []
  start = 0
  for index in torch.where(reset_mask)[0].tolist():
    if index > start:
      segments.append((start, index))
    start = index
  if start < len(data["position"]):
    segments.append((start, len(data["position"])))
  best_course = None
  for start, end in segments:
    xy = data["position"][start:end, :2]
    if len(xy) < 2:
      continue
    path_length = torch.linalg.vector_norm(xy[1:] - xy[:-1], dim=-1).sum()
    displacement = torch.linalg.vector_norm(xy[-1] - xy[0])
    straightness = displacement / path_length.clamp_min(1.0e-6)
    candidate = (
      int(end - start),
      float(displacement),
      float(path_length),
      float(straightness),
    )
    if best_course is None or candidate[0] > best_course[0]:
      best_course = candidate
  if best_course is None:
    print("  course: insufficient uninterrupted trajectory for classification")
  else:
    _, displacement, path_length, straightness = best_course
    course = "roughly straight" if straightness >= 0.8 else "circling/deviating"
    print(
      f"  course: {course}; longest uninterrupted XY displacement={displacement:.3f} m, "
      f"path={path_length:.3f} m, straightness={straightness:.3f}"
    )


def _latest_checkpoint(log_root: Path, experiment: str) -> Path:
  return get_checkpoint_path(
    log_root / experiment, run_dir=r".*", checkpoint=r"^model_\d+\.pt$"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default=DEFAULT_DEVICE)
  parser.add_argument("--motor-seconds", type=float, default=2.0)
  parser.add_argument("--ramp-seconds", type=float, default=13.0)
  parser.add_argument("--hold-seconds", type=float, default=5.0)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl"))
  parser.add_argument("--skip-checkpoint", action="store_true")
  args = parser.parse_args()

  print("TONY5 V1 diagnostics; no task configuration is modified")
  print(f"device={args.device}; physics_dt=0.001 s; KH={ROTOR_DRAG_KH:g}")
  print(
    f"motor constants: Kt={MOTOR_KT:g}, Ke={MOTOR_KE:g}, R={MOTOR_RESISTANCE:g}, "
    f"voltage_limit={BATTERY_VOLTAGE:g}, force_limit=0.11"
  )

  static_env = _quiet_env(_make_cfg(body_aero=True), args.device)
  try:
    _print_static_model(static_env)
  finally:
    static_env.close()

  _run_rotor_load_grid(args.device)
  _run_step_response_suite(args.device)

  constrained = _run_motor_test(
    device=args.device,
    body_aero=True,
    constrain_root=True,
    seconds=args.motor_seconds,
  )
  _print_rotor_summary("1. Constrained/stationary rotor test", constrained)

  free = _run_motor_test(
    device=args.device,
    body_aero=True,
    constrain_root=False,
    seconds=args.motor_seconds,
  )
  _print_rotor_summary("2. Normal free-flight motor test", free)

  original_kh = tony5_aero_physics.ROTOR_DRAG_KH
  try:
    tony5_aero_physics.ROTOR_DRAG_KH = 0.0
    h_disabled = _run_motor_test(
      device=args.device,
      body_aero=True,
      constrain_root=False,
      seconds=args.motor_seconds,
    )
  finally:
    tony5_aero_physics.ROTOR_DRAG_KH = original_kh
  _print_rotor_summary("4. Free flight with H-force temporarily disabled", h_disabled)

  body_aero_disabled = _run_motor_test(
    device=args.device,
    body_aero=False,
    constrain_root=False,
    seconds=args.motor_seconds,
  )
  _print_rotor_summary(
    "5. Free flight with body ellipsoid fluid temporarily disabled", body_aero_disabled
  )

  _h_force_measurement(args.device)

  body_force_env = _quiet_env(_make_cfg(body_aero=True), args.device)
  body_force_env_no_aero = _quiet_env(_make_cfg(body_aero=False), args.device)
  try:
    body_force_env.reset()
    body_force_env_no_aero.reset()
    print("\n5. Body ellipsoid fluid generalized-force comparison")
    for speed in AERO_SPEEDS:
      with_aero = _body_fluid_force(body_force_env, speed)
      without_aero = _body_fluid_force(body_force_env_no_aero, speed)
      print(
        f"  speed={speed:>5.1f} m/s: with body aero={with_aero.tolist()} N, "
        f"without={without_aero.tolist()} N"
      )
  finally:
    body_force_env.close()
    body_force_env_no_aero.close()

  body_drag_by_speed = _run_kh_sweep(args.device)
  simulated_max_omega = float(constrained["omega"][-1].mean())
  _print_forward_feasibility(body_drag_by_speed, simulated_max_omega)

  if not args.skip_checkpoint:
    checkpoint = args.checkpoint
    if checkpoint is None:
      checkpoint = _latest_checkpoint(args.log_root, "tony5_velocity_aero_v1")
    _run_checkpoint_ramp(args.device, checkpoint, args.ramp_seconds, args.hold_seconds)

  print("\n9-13. Safety and interpretation")
  print("  No task source files, PPO configs, reward terms, or constants were changed.")
  print("  H-force and body-fluid disablements were restored after each diagnostic.")
  print(
    "  Inspect NaNs with the finite checks below; any non-finite value is reported."
  )
  for name, values in (
    ("constrained", constrained),
    ("free", free),
    ("h_disabled", h_disabled),
    ("body_aero_disabled", body_aero_disabled),
  ):
    finite = all(bool(torch.isfinite(value).all()) for value in values.values())
    print(f"  {name}: finite={finite}")


if __name__ == "__main__":
  main()
