"""Reproduce and ablate the TONY5 V1 numerical runaway.

This is a diagnostic-only runner.  It changes no registered task defaults and
disables the V1 NaN dump guard for these runs so the first-divergence hook can
stop at the exact physics substep.  Run from the repository root with::

    uv run python scripts/diagnose_tony5_nan.py --case all
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import mujoco
import mujoco_warp
import torch
import warp as wp

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from mjlab.sim import sim as sim_module
from mjlab.sim.sim import Simulation
from mjlab.tasks.manager_based.tony5 import tony5_aero_physics, tony5_physics
from mjlab.tasks.manager_based.tony5.tony5_aero_actions import (
  Tony5AeroRotorSpeedAction,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_physics import (
  Tony5AeroRotorAerodynamics,
)
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.torch import configure_torch_backends

TASK_ID = "Mjlab-Tony5-Velocity-Aero-v1"
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_ITERATIONS = 24
NUM_ENVS = 8192
SEED = 42
PHYSICS_DT = 0.001
DECIMATION = 10
CONTROL_STEPS_PER_ITERATION = 24
OUTPUT_DIR = Path("/tmp/mjlab/tony5_nan_diagnostics")

THRESHOLDS = (
  ("root_speed_50_mps", "root_speed", 50.0),
  ("root_speed_100_mps", "root_speed", 100.0),
  ("body_angular_speed_30_rad_s", "body_angular_speed", 30.0),
  ("body_angular_speed_50_rad_s", "body_angular_speed", 50.0),
  ("rotor_qvel_3500_rad_s", "rotor_qvel", 3500.0),
  ("rotor_qvel_5000_rad_s", "rotor_qvel", 5000.0),
  ("qacc_1e5", "qacc", 1.0e5),
  ("qacc_1e7", "qacc", 1.0e7),
  ("qfrc_fluid_100_n", "qfrc_fluid", 100.0),
  ("qfrc_fluid_1000_n", "qfrc_fluid", 1000.0),
)


class FirstNonfinite(RuntimeError):
  """Internal stop used after the first invalid physics state."""


def _clean_json(value: Any) -> Any:
  if isinstance(value, list):
    return [_clean_json(item) for item in value]
  if isinstance(value, tuple):
    return [_clean_json(item) for item in value]
  if isinstance(value, float):
    return value if math.isfinite(value) else None
  return value


def _tensor_list(value: torch.Tensor) -> Any:
  return _clean_json(value.detach().cpu().tolist())


def _scalar(value: torch.Tensor) -> float:
  result = float(value.detach().cpu().item())
  return result if math.isfinite(result) else math.nan


@dataclass
class DivergenceTracker:
  """Capture threshold crossings and the first nonfinite physics state."""

  env: ManagerBasedRlEnv
  action: Tony5AeroRotorSpeedAction
  physics_step: int = 0
  max_values: dict[str, float] = field(default_factory=dict)
  first_crossings: dict[str, dict[str, Any]] = field(default_factory=dict)
  first_nonfinite: dict[str, Any] | None = None
  history: deque[dict[str, torch.Tensor]] = field(
    default_factory=lambda: deque(maxlen=6)
  )

  def __post_init__(self) -> None:
    robot = self.env.scene["robot"]
    self._free_q_adr = robot.indexing.free_joint_q_adr
    self._free_v_adr = robot.indexing.free_joint_v_adr
    self._rotor_q_adr = robot.indexing.joint_q_adr[self.action._joint_ids]
    self._rotor_v_adr = robot.indexing.joint_v_adr[self.action._joint_ids]
    self._aero = cast(Tony5AeroRotorAerodynamics, self.action._aerodynamics)

  def _snapshot(
    self,
    env_id: int,
    qpos: torch.Tensor,
    qvel: torch.Tensor,
    qacc: torch.Tensor,
    qfrc_actuator: torch.Tensor,
    qfrc_fluid: torch.Tensor,
    qfrc_applied: torch.Tensor,
    applied_omega: torch.Tensor,
  ) -> dict[str, Any]:
    data = self.env.sim.data
    h_force = self._aero.h_force_w
    thrust = 1.0e-6 * applied_omega.abs().square()
    kq_torque = -1.25e-8 * applied_omega * applied_omega.abs()
    free_q = qpos[env_id, self._free_q_adr]
    free_v = qvel[env_id, self._free_v_adr]
    rotor_q = qpos[env_id, self._rotor_q_adr]
    rotor_v = qvel[env_id, self._rotor_v_adr]
    fluid_generalized = qfrc_fluid[env_id]
    return {
      "physics_step": self.physics_step,
      "env_id": env_id,
      "root_position": _tensor_list(free_q[:3]),
      "root_quaternion": _tensor_list(free_q[3:7]),
      "root_linear_velocity": _tensor_list(free_v[:3]),
      "body_angular_velocity": _tensor_list(free_v[3:6]),
      "rotor_qpos": _tensor_list(rotor_q),
      "rotor_qvel": _tensor_list(rotor_v),
      "policy_action": _tensor_list(self.action.raw_action[env_id]),
      "rotor_target_speed": _tensor_list(self.action.speed_targets[env_id]),
      "esc_voltage": _tensor_list(self.action.last_voltage[env_id]),
      "thrust_per_rotor": _tensor_list(thrust[env_id]),
      "kq_torque_per_rotor": _tensor_list(kq_torque[env_id]),
      "h_force_per_rotor": _tensor_list(h_force[env_id]),
      "main_body_fluid_generalized": _tensor_list(fluid_generalized[self._free_v_adr]),
      "qfrc_actuator": _tensor_list(qfrc_actuator[env_id]),
      "qfrc_fluid": _tensor_list(qfrc_fluid[env_id]),
      "qfrc_applied": _tensor_list(qfrc_applied[env_id]),
      "qacc": _tensor_list(qacc[env_id]),
      "sensordata": _tensor_list(data.sensordata[env_id]),
    }

  def _preceding(self, env_id: int) -> list[dict[str, Any]]:
    result = []
    for item in self.history:
      result.append(
        {
          "physics_step": int(item["physics_step"][0].item()),
          "root_position": _tensor_list(item["qpos"][env_id, self._free_q_adr][:3]),
          "root_linear_velocity": _tensor_list(
            item["qvel"][env_id, self._free_v_adr][:3]
          ),
          "body_angular_velocity": _tensor_list(
            item["qvel"][env_id, self._free_v_adr][3:6]
          ),
          "rotor_qvel": _tensor_list(item["qvel"][env_id, self._rotor_v_adr]),
          "qacc": _tensor_list(item["qacc"][env_id]),
        }
      )
    return result

  def after_physics_step(self, sim: Simulation) -> None:
    self.physics_step += 1
    data = sim.data
    qpos = data.qpos
    qvel = data.qvel
    qacc = data.qacc
    qfrc_actuator = data.qfrc_actuator
    qfrc_fluid = data.qfrc_fluid
    qfrc_applied = data.qfrc_applied
    applied_omega = qvel[:, self._rotor_v_adr]

    root_speed = torch.linalg.vector_norm(qvel[:, self._free_v_adr[:3]], dim=-1)
    body_angular_speed = torch.linalg.vector_norm(
      qvel[:, self._free_v_adr[3:6]], dim=-1
    )
    rotor_qvel = qvel[:, self._rotor_v_adr].abs().max(dim=-1).values
    qacc_max = qacc.abs().max(dim=-1).values
    qfrc_fluid_max = qfrc_fluid.abs().max(dim=-1).values
    h_force_max = (
      torch.linalg.vector_norm(self._aero.h_force_w, dim=-1).max(dim=-1).values
    )
    series = {
      "root_speed": root_speed,
      "body_angular_speed": body_angular_speed,
      "rotor_qvel": rotor_qvel,
      "qacc": qacc_max,
      "qfrc_fluid": qfrc_fluid_max,
      "h_force": h_force_max,
    }
    for name, values in series.items():
      finite = torch.isfinite(values)
      if bool(finite.any()):
        current = values[finite].max()
        self.max_values[name] = max(self.max_values.get(name, 0.0), _scalar(current))

    history_item = {
      "physics_step": torch.tensor([self.physics_step], device=qpos.device),
      "qpos": qpos.detach().clone(),
      "qvel": qvel.detach().clone(),
      "qacc": qacc.detach().clone(),
    }
    self.history.append(history_item)
    for label, metric, threshold in THRESHOLDS:
      if label in self.first_crossings:
        continue
      values = series[metric]
      mask = torch.isfinite(values) & (values > threshold)
      if bool(mask.any()):
        env_id = int(torch.where(mask)[0][0].item())
        self.first_crossings[label] = {
          "physics_step": self.physics_step,
          "control_step": (self.physics_step + DECIMATION - 1) // DECIMATION,
          "collection_iteration": (
            (self.physics_step - 1) // (DECIMATION * CONTROL_STEPS_PER_ITERATION)
          ),
          "env_id": env_id,
          "value": _scalar(values[env_id]),
          "threshold": threshold,
          "metric": metric,
          "snapshot": self._snapshot(
            env_id,
            qpos,
            qvel,
            qacc,
            qfrc_actuator,
            qfrc_fluid,
            qfrc_applied,
            applied_omega,
          ),
          "preceding": self._preceding(env_id),
        }

    nonfinite_fields = {}
    for name in ("qpos", "qvel", "qacc", "qacc_warmstart", "sensordata"):
      value = getattr(data, name)
      mask = ~torch.isfinite(value).all(dim=-1)
      if bool(mask.any()):
        nonfinite_fields[name] = torch.where(mask)[0].cpu().tolist()
    if nonfinite_fields and self.first_nonfinite is None:
      env_id = int(next(iter(nonfinite_fields.values()))[0])
      self.first_nonfinite = {
        "physics_step": self.physics_step,
        "control_step": (self.physics_step + DECIMATION - 1) // DECIMATION,
        "collection_iteration": (
          (self.physics_step - 1) // (DECIMATION * CONTROL_STEPS_PER_ITERATION)
        ),
        "env_id": env_id,
        "fields": nonfinite_fields,
        "snapshot": self._snapshot(
          env_id,
          qpos,
          qvel,
          qacc,
          qfrc_actuator,
          qfrc_fluid,
          qfrc_applied,
          applied_omega,
        ),
        "preceding": self._preceding(env_id),
      }
      raise FirstNonfinite(
        f"first nonfinite state at physics step {self.physics_step}, env {env_id}"
      )


def _disable_body_aero(cfg: Any) -> None:
  original = cfg.scene.spec_fn

  def spec_fn(spec: mujoco.MjSpec) -> None:
    if original is not None:
      original(spec)
    geom = spec.geom("body_aero")
    if geom is None:
      geom = spec.geom("robot/body_aero")
    if geom is None:
      raise RuntimeError("Could not find assembled body_aero geom")
    geom.fluid_ellipsoid = 0.0
    cast(Any, geom).fluid_coefs = (0.0, 0.0, 0.0, 0.0, 0.0)

  cfg.scene.spec_fn = spec_fn


def _disable_global_fluid_density(cfg: Any) -> None:
  original = cfg.scene.spec_fn

  def spec_fn(spec: mujoco.MjSpec) -> None:
    if original is not None:
      original(spec)
    spec.option.density = 0.0

  cfg.scene.spec_fn = spec_fn


def _model_report(env: ManagerBasedRlEnv) -> dict[str, Any]:
  model = env.sim.mj_model
  geoms = []
  for geom_id in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    fluid = model.geom_fluid[geom_id]
    if name is not None and bool(torch.as_tensor(fluid).abs().any()):
      geoms.append(
        {
          "name": name,
          "size": model.geom_size[geom_id].tolist(),
          "position": model.geom_pos[geom_id].tolist(),
          "fluid": fluid.tolist(),
        }
      )
  body_id = next(
    body_id
    for body_id in range(model.nbody)
    if (
      (body_name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
      is not None
      and body_name.rsplit("/", 1)[-1] == "quad_base"
    )
  )
  return {
    "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
    "timestep": float(model.opt.timestep),
    "density": float(model.opt.density),
    "viscosity": float(model.opt.viscosity),
    "body_mass": float(model.body_mass[body_id]),
    "body_inertia": model.body_inertia[body_id].tolist(),
    "nonzero_fluid_geoms": geoms,
  }


def _run_case(
  case: str,
  integrator: str,
  device: str,
  iterations: int,
  disable_global_fluid: bool = False,
  disable_thrust: bool = False,
  disable_reaction_torque: bool = False,
) -> dict:
  args = TrainConfig.from_task(TASK_ID)
  args.env.scene.num_envs = NUM_ENVS
  args.env.seed = SEED
  args.agent.seed = SEED
  args.agent.max_iterations = iterations
  args.env.sim.nan_guard.enabled = False
  args.env.sim.mujoco.integrator = cast(Any, integrator)
  if case in ("B", "D"):
    _disable_body_aero(args.env)
  if disable_global_fluid:
    _disable_global_fluid_density(args.env)

  original_kh = tony5_aero_physics.ROTOR_DRAG_KH
  original_kt = tony5_physics.KT_THRUST
  original_kq = tony5_physics.KQ_TORQUE
  if case in ("C", "D"):
    tony5_aero_physics.ROTOR_DRAG_KH = 0.0
  if disable_thrust:
    tony5_physics.KT_THRUST = 0.0
  if disable_reaction_torque:
    tony5_physics.KQ_TORQUE = 0.0

  original_integrator = sim_module._INTEGRATOR_MAP.get("implicit")
  sim_module._INTEGRATOR_MAP["implicit"] = mujoco.mjtIntegrator.mjINT_IMPLICIT
  tracker: DivergenceTracker | None = None
  env: ManagerBasedRlEnv | None = None
  vec_env: RslRlVecEnvWrapper | None = None
  safety_monitor: Any | None = None
  original_step = Simulation.step

  def hooked_step(sim: Simulation) -> None:
    original_step(sim)
    assert tracker is not None
    tracker.after_physics_step(sim)

  Simulation.step = hooked_step  # type: ignore[method-assign]
  result: dict[str, Any] = {
    "case": case,
    "integrator_requested": integrator,
    "num_envs": NUM_ENVS,
    "seed": SEED,
    "physics_dt": PHYSICS_DT,
    "decimation": DECIMATION,
    "body_aero": case in ("A", "C"),
    "h_force": case in ("A", "B"),
    "global_fluid": not disable_global_fluid,
    "thrust": not disable_thrust,
    "reaction_torque": not disable_reaction_torque,
  }
  try:
    with contextlib.redirect_stdout(io.StringIO()):
      env = ManagerBasedRlEnv(cfg=args.env, device=device)
      action = cast(
        Tony5AeroRotorSpeedAction,
        env.action_manager.get_term("rotor_speed"),
      )
      tracker = DivergenceTracker(env, action)
      safety_monitor = action.safety_monitor
      result["model"] = _model_report(env)
      vec_env = RslRlVecEnvWrapper(env, clip_actions=args.agent.clip_actions)
      runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
      runner = runner_cls(vec_env, asdict(args.agent), device=device)
      runner.learn(
        num_learning_iterations=iterations,
        init_at_random_ep_len=True,
      )
    result["status"] = "stable_through_requested_iterations"
  except FirstNonfinite as exc:
    result["status"] = "nonfinite_state"
    result["stop_reason"] = str(exc)
  except Exception as exc:
    result["status"] = "diagnostic_error"
    result["stop_reason"] = f"{type(exc).__name__}: {exc}"
  finally:
    if tracker is not None:
      result["physics_steps_observed"] = tracker.physics_step
      result["max_values"] = tracker.max_values
      result["first_crossings"] = tracker.first_crossings
      result["first_nonfinite"] = tracker.first_nonfinite
    if safety_monitor is not None:
      result["safety_totals"] = {
        "failure": int(safety_monitor.total_failure_count.item()),
        "root_speed": int(safety_monitor.total_root_speed_count.item()),
        "body_omega": int(safety_monitor.total_body_omega_count.item()),
        "rotor_speed": int(safety_monitor.total_rotor_speed_count.item()),
        "qacc": int(safety_monitor.total_qacc_count.item()),
        "nonfinite": int(safety_monitor.total_nonfinite_count.item()),
      }
    Simulation.step = original_step
    if vec_env is not None:
      vec_env.close()
    elif env is not None:
      env.close()
    tony5_aero_physics.ROTOR_DRAG_KH = original_kh
    tony5_physics.KT_THRUST = original_kt
    tony5_physics.KQ_TORQUE = original_kq
    if original_integrator is None:
      sim_module._INTEGRATOR_MAP.pop("implicit", None)
    else:
      sim_module._INTEGRATOR_MAP["implicit"] = original_integrator
  return _clean_json(result)


def _runtime_report() -> dict[str, str]:
  return {
    "mujoco": mujoco.__version__,
    "mujoco_warp": getattr(mujoco_warp, "__version__", "unknown"),
    "mjlab": version("mjlab"),
    "warp": wp.config.version,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--case", choices=("A", "B", "C", "D", "all"), default="all")
  parser.add_argument(
    "--integrator", choices=("implicitfast", "implicit"), default="implicitfast"
  )
  parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
  parser.add_argument(
    "--disable-global-fluid",
    action="store_true",
    help="Diagnostic-only: set assembled MuJoCo fluid density to zero.",
  )
  parser.add_argument(
    "--disable-thrust",
    action="store_true",
    help="Diagnostic-only: disable custom rotor thrust.",
  )
  parser.add_argument(
    "--disable-reaction-torque",
    action="store_true",
    help="Diagnostic-only: disable custom propeller reaction torque.",
  )
  parser.add_argument("--device", default=DEFAULT_DEVICE)
  parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
  args = parser.parse_args()
  configure_torch_backends()
  cases = ("A", "B", "C", "D") if args.case == "all" else (args.case,)
  reports: dict[str, Any] = {
    "runtime": _runtime_report(),
    "cases": [],
  }
  for case in cases:
    report = _run_case(
      case,
      args.integrator,
      args.device,
      args.iterations,
      disable_global_fluid=args.disable_global_fluid,
      disable_thrust=args.disable_thrust,
      disable_reaction_torque=args.disable_reaction_torque,
    )
    reports["cases"].append(report)
    print(
      case,
      report["status"],
      "physics_steps=",
      report.get("physics_steps_observed"),
      "first_nonfinite=",
      report.get("first_nonfinite"),
      flush=True,
    )
  args.output_dir.mkdir(parents=True, exist_ok=True)
  output = args.output_dir / f"report_{args.case}_{args.integrator}.json"
  output.write_text(json.dumps(_clean_json(reports), indent=2) + "\n")
  print(f"report={output}")
  print(json.dumps(_clean_json(reports["runtime"]), sort_keys=True))


if __name__ == "__main__":
  main()
