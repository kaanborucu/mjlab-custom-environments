"""A/B diagnostic for TONY5 V1 rotor-body generic fluid loading.

This script changes only the temporary in-memory MJCF for case B. It does not
modify the registered V1 task or any production configuration.
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import mujoco
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from mjlab.sim.sim import Simulation
from mjlab.tasks.manager_based.tony5.tony5_aero_actions import (
  Tony5AeroRotorSpeedAction,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_env_cfg import (
  tony5_velocity_aero_play_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_physics import ROTOR_DRAG_KH
from mjlab.tasks.manager_based.tony5.tony5_aero_safety import (
  BODY_ANGULAR_SPEED_LIMIT,
  QACC_LIMIT,
  ROOT_SPEED_LIMIT,
  ROTOR_QVEL_LIMIT,
)
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  KQ_TORQUE,
  OMEGA_MAX,
)
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Tony5-Velocity-Aero-v1"
NUM_ENVS = 8192
SEED = 42
PHYSICS_DT = 0.001
INTEGRATOR = "implicitfast"
PHYSICS_STEPS = 5760
TRAINING_ITERATIONS = 24


def _delete_rotor_fluid_disable_geoms(cfg: Any) -> None:
  """Remove only V1's four temporary rotor fluid-disable geoms."""
  original_scene_fn = cfg.scene.spec_fn

  def spec_fn(spec: mujoco.MjSpec) -> None:
    if original_scene_fn is not None:
      original_scene_fn(spec)
    for geom in list(spec.geoms):
      if geom.name.endswith("_fluid_disable"):
        spec.delete(geom)

  cfg.scene.spec_fn = spec_fn


def _state_limits(
  sim: Simulation,
  action: Tony5AeroRotorSpeedAction,
) -> torch.Tensor:
  """Return per-environment numerical-safety failures without mutating state."""
  data = sim.data
  monitor = action.safety_monitor
  free_v_adr = monitor._free_v_adr
  rotor_v_adr = action._entity.indexing.joint_v_adr[action._joint_ids]
  qvel = data.qvel
  qacc = data.qacc
  root_speed = torch.linalg.vector_norm(qvel[:, free_v_adr[:3]], dim=-1)
  body_omega = torch.linalg.vector_norm(qvel[:, free_v_adr[3:6]], dim=-1)
  rotor_speed = qvel[:, rotor_v_adr].abs().amax(dim=-1)
  max_abs_qacc = qacc.abs().amax(dim=-1)
  nonfinite = ~(
    torch.isfinite(data.qpos).all(dim=-1)
    & torch.isfinite(qvel).all(dim=-1)
    & torch.isfinite(qacc).all(dim=-1)
  )
  return (
    nonfinite
    | (torch.isfinite(root_speed) & (root_speed > ROOT_SPEED_LIMIT))
    | (torch.isfinite(body_omega) & (body_omega > BODY_ANGULAR_SPEED_LIMIT))
    | (torch.isfinite(rotor_speed) & (rotor_speed > ROTOR_QVEL_LIMIT))
    | (torch.isfinite(max_abs_qacc) & (max_abs_qacc > QACC_LIMIT))
  )


def _model_fluid_report(env: ManagerBasedRlEnv) -> dict[str, int]:
  model = env.sim.mj_model
  total = 0
  disable = 0
  for geom_id in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    if name.endswith("_fluid_disable"):
      disable += 1
    if bool(torch.as_tensor(model.geom_fluid[geom_id]).abs().any()):
      total += 1
  return {"nonzero_fluid_geoms": total, "rotor_disable_geoms": disable}


def _run_ab(case: str, device: str) -> dict[str, Any]:
  cfg = TrainConfig.from_task(TASK_ID)
  cfg.env.scene.num_envs = NUM_ENVS
  cfg.env.seed = SEED
  cfg.agent.seed = SEED
  cfg.agent.max_iterations = TRAINING_ITERATIONS
  cfg.env.sim.nan_guard.enabled = False
  cfg.env.sim.mujoco.integrator = INTEGRATOR
  if case == "B_old_generic_rotor_fluid":
    _delete_rotor_fluid_disable_geoms(cfg.env)

  env: ManagerBasedRlEnv | None = None
  vec_env: RslRlVecEnvWrapper | None = None
  action: Tony5AeroRotorSpeedAction | None = None
  original_step = Simulation.step
  first_safety_step: int | None = None
  first_safety_env: int | None = None
  safety_step = 0
  nan_count = 0
  max_values = {
    "body_angular_speed": 0.0,
    "root_speed": 0.0,
    "rotor_speed": 0.0,
  }
  result: dict[str, Any] = {
    "case": case,
    "num_envs": NUM_ENVS,
    "seed": SEED,
    "physics_dt": PHYSICS_DT,
    "integrator": INTEGRATOR,
    "requested_physics_steps": PHYSICS_STEPS,
    "requested_training_iterations": TRAINING_ITERATIONS,
    "rotor_drag_kh": ROTOR_DRAG_KH,
    "model": None,
  }

  def hooked_step(sim: Simulation) -> None:
    nonlocal first_safety_env, first_safety_step, nan_count, safety_step
    original_step(sim)
    safety_step += 1
    assert action is not None
    failed = _state_limits(sim, action)
    if bool(failed.any()):
      if first_safety_step is None:
        first_safety_step = safety_step
        first_safety_env = int(torch.nonzero(failed, as_tuple=False)[0, 0].item())
      nan_count += int((~torch.isfinite(sim.data.qpos).all(dim=-1)).sum().item())
      nan_count += int((~torch.isfinite(sim.data.qvel).all(dim=-1)).sum().item())
      nan_count += int((~torch.isfinite(sim.data.qacc).all(dim=-1)).sum().item())

    data = sim.data
    monitor = action.safety_monitor
    free_v_adr = monitor._free_v_adr
    rotor_v_adr = action._entity.indexing.joint_v_adr[action._joint_ids]
    root_speed = torch.linalg.vector_norm(data.qvel[:, free_v_adr[:3]], dim=-1)
    body_omega = torch.linalg.vector_norm(data.qvel[:, free_v_adr[3:6]], dim=-1)
    rotor_speed = data.qvel[:, rotor_v_adr].abs().amax(dim=-1)
    for name, value in (
      ("root_speed", root_speed),
      ("body_angular_speed", body_omega),
      ("rotor_speed", rotor_speed),
    ):
      finite = value[torch.isfinite(value)]
      if finite.numel() > 0:
        max_values[name] = max(max_values[name], float(finite.max().item()))

  Simulation.step = hooked_step  # type: ignore[method-assign]
  try:
    with contextlib.redirect_stdout(io.StringIO()):
      env = ManagerBasedRlEnv(cfg=cfg.env, device=device)
      action = cast(
        Tony5AeroRotorSpeedAction,
        env.action_manager.get_term("rotor_speed"),
      )
      result["model"] = _model_fluid_report(env)
      vec_env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
      runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
      runner = runner_cls(vec_env, asdict(cfg.agent), device=device)
      runner.learn(
        num_learning_iterations=TRAINING_ITERATIONS,
        init_at_random_ep_len=True,
      )
      result["status"] = "completed"
  except Exception as exc:
    result["status"] = "stopped_with_error"
    result["error"] = f"{type(exc).__name__}: {exc}"
  finally:
    result["observed_physics_steps"] = safety_step
    result["first_safety_trip_step"] = first_safety_step
    result["first_safety_trip_env"] = first_safety_env
    result["nan_count"] = nan_count
    result["max_values"] = max_values
    if action is not None:
      monitor = action.safety_monitor
      result["safety_trip_count"] = int(monitor.total_failure_count.item())
      result["safety_totals"] = {
        "root_speed": int(monitor.total_root_speed_count.item()),
        "body_angular_speed": int(monitor.total_body_omega_count.item()),
        "rotor_speed": int(monitor.total_rotor_speed_count.item()),
        "qacc": int(monitor.total_qacc_count.item()),
        "nonfinite": int(monitor.total_nonfinite_count.item()),
      }
    Simulation.step = original_step
    if vec_env is not None:
      vec_env.close()
    elif env is not None:
      env.close()
  return result


def _run_motor(case: str, device: str) -> dict[str, Any]:
  cfg = tony5_velocity_aero_play_env_cfg()
  cfg.scene.num_envs = 1
  if case == "B_old_generic_rotor_fluid":
    _delete_rotor_fluid_disable_geoms(cfg)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    robot = env.scene["robot"]
    action = cast(
      Tony5AeroRotorSpeedAction,
      env.action_manager.get_term("rotor_speed"),
    )
    fixed_pose = robot.data.root_link_pose_w.clone()
    zero_velocity = torch.zeros((1, 6), device=device)
    action.process_actions(torch.ones((1, 4), device=device))
    for _ in range(round(2.0 / env.physics_dt)):
      robot.write_root_link_pose_to_sim(fixed_pose)
      robot.write_root_link_velocity_to_sim(zero_velocity)
      env.scene.write_data_to_sim()
      env.sim.forward()
      action.apply_actions()
      env.scene.write_data_to_sim()
      env.sim.step()
      env.scene.update(dt=env.physics_dt)

    omega = robot.data.joint_vel[:, action._joint_ids][0]
    rotor_dofs = robot.indexing.joint_v_adr[action._joint_ids]
    qfrc_fluid = env.sim.data.qfrc_fluid[:, rotor_dofs][0]
    kq_torque = -KQ_TORQUE * omega * omega.abs()
    return {
      "case": case,
      "target_rad_s": OMEGA_MAX,
      "final_rad_s": omega.detach().cpu().tolist(),
      "final_rpm": (omega * 60.0 / (2.0 * torch.pi)).detach().cpu().tolist(),
      "qfrc_fluid_rotor_axis": qfrc_fluid.detach().cpu().tolist(),
      "custom_kq_torque": kq_torque.detach().cpu().tolist(),
      "model": _model_fluid_report(env),
    }
  finally:
    env.close()


def main() -> None:
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  reports = {
    "runtime": {
      "device": device,
      "mujoco": mujoco.__version__,
    },
    "cases": [
      _run_ab("A_current_isolated_rotor_fluid", device),
      _run_ab("B_old_generic_rotor_fluid", device),
    ],
    "motor_tests": [
      _run_motor("A_current_isolated_rotor_fluid", device),
      _run_motor("B_old_generic_rotor_fluid", device),
    ],
  }
  output = Path("/tmp/mjlab/tony5_rotor_fluid_ab.json")
  output.parent.mkdir(parents=True, exist_ok=True)
  output.write_text(json.dumps(reports, indent=2) + "\n")
  print(json.dumps(reports, indent=2))
  print(f"report={output}")


if __name__ == "__main__":
  main()
