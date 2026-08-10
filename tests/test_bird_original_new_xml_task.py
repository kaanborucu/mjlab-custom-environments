"""Tests for the original bird task running its adjusted XML copy."""

import mujoco
import pytest

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.bird_5dof_original_new_xml.env_cfgs import (
  get_adjusted_new_xml_spec,
)
from mjlab.tasks.velocity.config.bird_5dof_original_new_xml.rl_cfg import (
  bird_5dof_original_new_xml_ppo_runner_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

TASK_ID = "Mjlab-Velocity-Bird-5DoF-Original-NewXML"


def test_original_setup_uses_adjusted_original_xml() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert load_rl_cfg(TASK_ID) is not None
  agent_cfg = bird_5dof_original_new_xml_ppo_runner_cfg()
  robot_cfg = cfg.scene.entities["robot"]
  command = cfg.commands["twist"]

  assert robot_cfg.spec_fn is get_adjusted_new_xml_spec
  assert robot_cfg.init_state.pos == (0.0, 0.0, 8.0)
  assert robot_cfg.init_state.lin_vel == (3.0, 0.0, 0.0)
  assert cfg.scene.num_envs == 4096
  assert cfg.sim.mujoco.timestep == pytest.approx(0.002)
  assert cfg.decimation == 10
  assert cfg.episode_length_s == pytest.approx(10.0)
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (2.5, 4.0)
  assert command.ranges.lin_vel_y == (0.0, 0.0)
  assert command.ranges.lin_vel_z is None
  assert command.ranges.ang_vel_z == (0.0, 0.0)
  assert set(cfg.events) == {"reset_scene_to_default"}
  assert set(cfg.rewards) == {
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "alive",
    "action_rate",
    "joint_torques",
  }
  assert set(cfg.terminations) == {
    "time_out",
    "low_altitude",
    "bad_orientation",
    "non_finite_state",
  }
  assert agent_cfg.experiment_name == "bird_5dof_velocity_original_new_xml"


def test_new_xml_task_applies_only_requested_physics_scaling() -> None:
  spec = get_adjusted_new_xml_spec()
  model = spec.compile()

  base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bird")
  left_wing_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wing_aero")
  right_wing_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_wing_aero")

  assert model.body_mass[base_id] == pytest.approx(0.260 * 0.8)
  assert model.body_inertia[base_id] == pytest.approx(
    [0.00072222 * 0.8, 0.00308148 * 0.8, 0.00322593 * 0.8]
  )
  assert model.body_mass.sum() == pytest.approx(0.258)
  expected_wing_size = pytest.approx([0.108225 * 1.1, 0.22375 * 1.1, 0.012 * 1.1])
  assert model.geom_size[left_wing_id] == expected_wing_size
  assert model.geom_size[right_wing_id] == expected_wing_size
