"""Controlled V3 prop, battery, and H-force diagnostics.

The script changes only process-local diagnostic objects. It never edits the
production V3 configuration or the V0 task.

Run from the repository root with::

    uv run python scripts/diagnose_tony5_omni_v3_physics.py --device cuda:0
"""

from __future__ import annotations

import argparse
import contextlib
import io
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.tasks.manager_based.tony5 import tony5_omni_v3_physics
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  OMEGA_HOVER,
  OMEGA_MAX,
  TOTAL_MASS,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_actions import (
  Tony5OmniV3RotorSpeedAction,
  Tony5OmniV3RotorSpeedActionCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_battery import (
  Tony5OmniV3BatteryCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_env_cfg import (
  tony5_velocity_aero_omni_v3_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_physics import (
  Tony5OmniV3PhysicsCfg,
  Tony5OmniV3RotorAerodynamics,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_prop import (
  Tony5OmniV3PropCfg,
  static_cq0,
  static_ct0,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_wind import (
  Tony5OmniV3WindCfg,
)

KH_SWEEP = (2.5e-6, 5.0e-6, 7.5e-6, 1.0e-5, 1.25e-5)
HORIZONTAL_SPEEDS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 27.78)
ROTOR_SPEEDS = (OMEGA_HOVER, 1800.0, 2200.0, OMEGA_MAX)
G = 9.80665


def _quiet_env(cfg: ManagerBasedRlEnvCfg, device: str) -> ManagerBasedRlEnv:
  with contextlib.redirect_stdout(io.StringIO()):
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def _make_env(
  device: str,
  *,
  num_envs: int = 1,
  wind: Tony5OmniV3WindCfg | None = None,
  physics: Tony5OmniV3PhysicsCfg | None = None,
  battery: Tony5OmniV3BatteryCfg | None = None,
) -> ManagerBasedRlEnv:
  cfg = tony5_velocity_aero_omni_v3_env_cfg()
  cfg.scene.num_envs = num_envs
  action_cfg = cast(Tony5OmniV3RotorSpeedActionCfg, cfg.actions["rotor_speed"])
  if wind is not None:
    action_cfg.wind = wind
  if physics is not None:
    action_cfg.physics = physics
  if battery is not None:
    action_cfg.battery = battery
  return _quiet_env(cfg, device)


def _set_state(
  env: ManagerBasedRlEnv,
  action: Tony5OmniV3RotorSpeedAction,
  speed: float,
  omega: float,
) -> None:
  robot = env.scene["robot"]
  root_velocity_b = torch.zeros((env.num_envs, 6), device=env.device)
  root_velocity_b[:, 0] = speed
  rotor_velocity = torch.full(
    (env.num_envs, len(action._joint_ids)),
    omega,
    device=env.device,
  )
  robot.write_root_link_velocity_b_to_sim(root_velocity_b)
  robot.write_joint_velocity_to_sim(rotor_velocity, joint_ids=action._joint_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()


def _h_force_sweep(device: str) -> None:
  env = _make_env(device)
  original_kh = tony5_omni_v3_physics.ROTOR_DRAG_KH
  try:
    env.reset()
    action = cast(
      Tony5OmniV3RotorSpeedAction,
      env.action_manager.get_term("rotor_speed"),
    )
    aero = cast(Tony5OmniV3RotorAerodynamics, action._aerodynamics)
    robot = env.scene["robot"]
    root_dofs = robot.indexing.free_joint_v_adr
    weight = TOTAL_MASS * G
    print("\nH-force calibration sweep (process-local KH override)")
    print(f"static_CT0={static_ct0():.9g} static_CQ0={static_cq0():.9g}")
    print(
      "KH speed omega H_total_N H_per_rotor_N body_drag_N "
      "total_aero_N H_over_total F_over_mg"
    )
    for kh in KH_SWEEP:
      tony5_omni_v3_physics.ROTOR_DRAG_KH = kh
      for speed in HORIZONTAL_SPEEDS:
        for omega in ROTOR_SPEEDS:
          _set_state(env, action, speed, omega)
          aero.compute_wrenches()
          h_force_xy = aero.h_force_w[0, :, :2]
          h_total = torch.linalg.vector_norm(h_force_xy.sum(dim=0)).item()
          h_per_rotor = torch.linalg.vector_norm(h_force_xy, dim=-1).mean().item()
          body_force_xy = env.sim.data.qfrc_fluid[0, root_dofs[:2]]
          body_drag = torch.linalg.vector_norm(body_force_xy).item()
          total_aero = torch.linalg.vector_norm(
            body_force_xy + h_force_xy.sum(dim=0)
          ).item()
          ratio = h_total / total_aero if total_aero > 1.0e-12 else 0.0
          print(
            f"{kh:.3e} {speed:5.2f} {omega:7.1f} {h_total:10.6f} "
            f"{h_per_rotor:14.6f} {body_drag:12.6f} {total_aero:12.6f} "
            f"{ratio:12.6f} {total_aero / weight:10.6f}"
          )
  finally:
    tony5_omni_v3_physics.ROTOR_DRAG_KH = original_kh
    env.close()


def _normalized_action_for_speed(omega: float) -> float:
  if omega <= OMEGA_HOVER:
    return omega / OMEGA_HOVER - 1.0
  return (omega - OMEGA_HOVER) / (OMEGA_MAX - OMEGA_HOVER)


def _run_battery_test(device: str, enabled: bool) -> None:
  cfg = Tony5OmniV3BatteryCfg(
    enable_battery_sag=enabled,
    battery_resistance=0.02,
  )
  env = _make_env(device, battery=cfg)
  try:
    env.reset()
    robot = env.scene["robot"]
    action = cast(
      Tony5OmniV3RotorSpeedAction,
      env.action_manager.get_term("rotor_speed"),
    )
    fixed_pose = robot.data.root_link_pose_w.clone()
    zero_velocity = torch.zeros((env.num_envs, 6), device=env.device)
    print(f"\nBattery sag {'ON' if enabled else 'OFF'}")
    print("target actual voltage current_total loaded_voltage sag")
    for target in (OMEGA_HOVER, 1800.0, OMEGA_MAX):
      env.reset()
      action_value = _normalized_action_for_speed(target)
      action.process_actions(
        torch.full((env.num_envs, 4), action_value, device=env.device)
      )
      steps = round(1.0 / env.physics_dt)
      for _ in range(steps):
        robot.write_root_link_pose_to_sim(fixed_pose)
        robot.write_root_link_velocity_to_sim(zero_velocity)
        env.scene.write_data_to_sim()
        env.sim.forward()
        action.apply_actions()
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
      action.battery.update_voltage_limit()
      actual = robot.data.joint_vel[:, action._joint_ids].mean().item()
      voltage = action.last_voltage.mean().item()
      current = action.battery.total_current.mean().item()
      loaded = action.battery.loaded_voltage.mean().item()
      sag = action.battery.voltage_sag.mean().item()
      print(
        f"{target:6.1f} {actual:7.1f} {voltage:7.3f} {current:12.3f} "
        f"{loaded:13.3f} {sag:7.3f}"
      )
  finally:
    env.close()


def _feature_smoke(device: str) -> None:
  prop = Tony5OmniV3PropCfg(
    enable_advanced_prop_model=True,
    ct_curve_j=(-1.0, 0.0, 1.0),
    ct_curve_values=(0.35, 0.30, 0.25),
    cq_curve_j=(-1.0, 0.0, 1.0),
    cq_curve_values=(0.07, 0.06, 0.05),
  )
  physics = Tony5OmniV3PhysicsCfg(
    prop=prop,
    enable_blade_flapping=True,
  )
  env = _make_env(
    device,
    num_envs=64,
    wind=Tony5OmniV3WindCfg(
      enable_wind=True,
      wind_x=2.0,
      enable_gusts=True,
      gust_sigma=0.2,
      gust_tau=1.0,
      max_gust_speed=1.0,
    ),
    physics=physics,
    battery=Tony5OmniV3BatteryCfg(enable_battery_sag=True),
  )
  try:
    env.reset()
    action = cast(
      Tony5OmniV3RotorSpeedAction,
      env.action_manager.get_term("rotor_speed"),
    )
    action_tensor = torch.zeros((env.num_envs, 4), device=env.device)
    for _ in range(20):
      observations, rewards, terminated, truncated, _ = env.step(action_tensor)
      del rewards, terminated, truncated
      aero = cast(Tony5OmniV3RotorAerodynamics, action._aerodynamics)
      actor_observation = observations["actor"]
      if isinstance(actor_observation, dict):
        observation_tensors = tuple(actor_observation.values())
      else:
        observation_tensors = (actor_observation,)
      tensors = (
        *observation_tensors,
        aero.advance_ratio,
        aero.ct,
        aero.cq,
        aero.flapping_moment_w,
        action.battery.motor_current,
        action.battery.loaded_voltage,
      )
      if not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("Combined V3 feature smoke test produced non-finite data")
    print("\nCombined feature smoke: 64 environments, 20 control steps, finite")
  finally:
    env.close()


def _rollout_smoke(device: str, enable_wind: bool) -> None:
  wind = Tony5OmniV3WindCfg(
    enable_wind=enable_wind,
    wind_x=2.0,
    enable_gusts=enable_wind,
    gust_sigma=0.5,
    gust_tau=1.0,
    max_gust_speed=2.0,
  )
  env = _make_env(device, num_envs=4096, wind=wind)
  try:
    env.reset()
    action = cast(
      Tony5OmniV3RotorSpeedAction,
      env.action_manager.get_term("rotor_speed"),
    )
    action_tensor = torch.zeros((env.num_envs, 4), device=env.device)
    for _ in range(20):
      observations, rewards, terminated, truncated, _ = env.step(action_tensor)
      actor_observation = observations["actor"]
      observation_tensors = (
        tuple(actor_observation.values())
        if isinstance(actor_observation, dict)
        else (actor_observation,)
      )
      tensors = (*observation_tensors, rewards, terminated, truncated)
      if not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("4096-environment rollout produced non-finite data")
    robot = env.scene["robot"]
    rotor_dofs = robot.indexing.joint_v_adr[action._joint_ids]
    rotor_fluid = env.sim.data.qfrc_fluid[:, rotor_dofs]
    if torch.max(torch.abs(rotor_fluid)).item() > 1.0e-5:
      raise RuntimeError("Rotor native fluid isolation failed in 4096 rollout")
    print(
      f"4096-environment rollout wind={'on' if enable_wind else 'off'}: "
      "20 control steps, finite, rotor qfrc_fluid≈0"
    )
  finally:
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
  )
  parser.add_argument("--skip-h-sweep", action="store_true")
  parser.add_argument("--only-h-sweep", action="store_true")
  parser.add_argument("--skip-rollout", action="store_true")
  args = parser.parse_args()
  print(f"device={args.device}")
  if not args.only_h_sweep:
    _run_battery_test(args.device, enabled=False)
    _run_battery_test(args.device, enabled=True)
    _feature_smoke(args.device)
    if not args.skip_rollout:
      _rollout_smoke(args.device, enable_wind=False)
      _rollout_smoke(args.device, enable_wind=True)
  if args.only_h_sweep or not args.skip_h_sweep:
    _h_force_sweep(args.device)


if __name__ == "__main__":
  main()
