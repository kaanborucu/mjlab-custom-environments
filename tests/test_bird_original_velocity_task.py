"""Regression tests for the frozen original bird velocity task."""

import hashlib
import math

import mujoco
import pytest

from mjlab.asset_zoo.robots import (
  BIRD_5DOF_ORIGINAL_ACTION_SCALE,
  get_bird_5dof_original_robot_cfg,
)
from mjlab.asset_zoo.robots.bird_5dof_original.bird_constants import (
  BIRD_5DOF_ORIGINAL_XML,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.bird_5dof_original.rl_cfg import (
  bird_5dof_original_ppo_runner_cfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

TASK_ID = "Mjlab-Velocity-Bird-5DoF-Original"
ORIGINAL_XML_SHA256 = "8504eb213d1681c94dec420b7aa96d0b912280b8c6dc45004ff6806e02646961"


def test_original_bird_model_is_frozen() -> None:
  xml_digest = hashlib.sha256(BIRD_5DOF_ORIGINAL_XML.read_bytes()).hexdigest()
  model = get_bird_5dof_original_robot_cfg().build().compile()
  ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

  assert xml_digest == ORIGINAL_XML_SHA256
  assert model.nu == 5
  assert model.nq == 12
  assert model.nv == 11
  assert model.body_mass.sum() == pytest.approx(0.31)
  assert ground_id >= 0
  assert model.geom_contype[ground_id] == 1
  assert model.geom_conaffinity[ground_id] == 1


def test_original_bird_environment_config() -> None:
  cfg = load_env_cfg(TASK_ID)
  action = cfg.actions["joint_pos"]
  command = cfg.commands["twist"]

  assert isinstance(action, JointPositionActionCfg)
  assert isinstance(command, UniformVelocityCommandCfg)
  assert cfg.scene.num_envs == 4096
  assert cfg.scene.entities["robot"].init_state.pos == (0.0, 0.0, 8.0)
  assert cfg.scene.entities["robot"].init_state.lin_vel == (3.0, 0.0, 0.0)
  assert cfg.sim.mujoco.timestep == pytest.approx(0.002)
  assert cfg.decimation == 10
  assert cfg.episode_length_s == pytest.approx(10.0)
  assert action.scale == BIRD_5DOF_ORIGINAL_ACTION_SCALE
  assert command.ranges.lin_vel_x == (2.5, 4.0)
  assert command.ranges.lin_vel_y == (0.0, 0.0)
  assert command.ranges.lin_vel_z is None
  assert command.ranges.ang_vel_z == (0.0, 0.0)
  assert command.resampling_time_range == (10.0, 10.0)
  assert set(cfg.events) == {"reset_scene_to_default"}
  assert set(cfg.rewards) == {
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "alive",
    "action_rate",
    "joint_torques",
  }
  assert cfg.rewards["track_linear_velocity"].weight == pytest.approx(2.0)
  assert cfg.rewards["track_angular_velocity"].weight == pytest.approx(0.5)
  assert cfg.rewards["upright"].weight == pytest.approx(0.5)
  assert cfg.rewards["alive"].weight == pytest.approx(0.2)
  assert cfg.rewards["action_rate"].weight == pytest.approx(-0.01)
  assert cfg.rewards["joint_torques"].weight == pytest.approx(-1.0e-4)
  assert set(cfg.terminations) == {
    "time_out",
    "low_altitude",
    "bad_orientation",
    "non_finite_state",
  }
  assert cfg.terminations["low_altitude"].params["minimum_height"] == pytest.approx(
    0.25
  )
  assert cfg.terminations["bad_orientation"].params["limit_angle"] == pytest.approx(
    math.radians(80.0)
  )


def test_original_bird_ppo_config() -> None:
  assert load_rl_cfg(TASK_ID) is not None
  cfg = bird_5dof_original_ppo_runner_cfg()

  assert cfg.actor.hidden_dims == (256, 128, 64)
  assert cfg.critic.hidden_dims == (256, 128, 64)
  assert cfg.actor.distribution_cfg is not None
  assert cfg.actor.distribution_cfg["init_std"] == pytest.approx(0.6)
  assert cfg.algorithm.learning_rate == pytest.approx(3.0e-4)
  assert cfg.algorithm.entropy_coef == pytest.approx(0.005)
  assert cfg.num_steps_per_env == 24
  assert cfg.save_interval == 100
  assert cfg.max_iterations == 3000
  assert cfg.experiment_name == "bird_5dof_velocity_original"
