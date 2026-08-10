"""Tests for the dedicated bird forward-flight environment."""

import math
from unittest.mock import MagicMock

import mujoco
import pytest
import torch

from mjlab.asset_zoo.robots.bird_5dof_forward import (
  BIRD_5DOF_FORWARD_XML,
  get_forward_spec,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.bird_5dof_forward.env_cfgs import (
  FORWARD_DIRECTION_MIN_SPEED,
  FORWARD_DIRECTION_REWARD_STD,
  FORWARD_REWARD_WEIGHTS,
  FORWARD_ROLL_FULL_SPEED,
  FORWARD_ROLL_MIN_HORIZONTAL,
  FORWARD_ROLL_REWARD_STD,
  FORWARD_ROLL_ZERO_SPEED,
  FORWARD_TILT_LIMIT_DEG,
)
from mjlab.tasks.velocity.config.bird_5dof_forward.events import (
  reset_heading_forward_velocity,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.rewards import (
  base_roll_pitch_angular_velocity_l2,
  level_roll_alignment,
  low_speed_world_up_alignment,
  track_commanded_velocity_direction,
)

FORWARD_TASK_ID = "Mjlab-Velocity-Bird-Forward-3D"
OMNIDIRECTIONAL_TASK_ID = "Mjlab-Velocity-Bird-5DoF"


def test_forward_task_tracks_forward_biased_3d_velocity() -> None:
  cfg = load_env_cfg(FORWARD_TASK_ID)
  command = cfg.commands["twist"]

  assert cfg.scene.entities["robot"].spec_fn is get_forward_spec
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (0.0, 10.0)
  assert command.ranges.lin_vel_y == (-4.0, 4.0)
  assert command.ranges.lin_vel_z == (-6.0, 6.0)
  assert command.ranges.ang_vel_z == (0.0, 0.0)
  assert command.linear_velocity_frame == "heading"
  assert command.yaw_velocity_frame == "world"
  assert command.resampling_time_range == (2.0, 4.0)
  assert command.rel_standing_envs == 0.0
  assert command.rel_world_envs == 1.0
  assert command.rel_forward_envs == 0.0
  assert command.yaw_command_speed_threshold is None
  assert cfg.curriculum == {}
  assert cfg.sim.mujoco.timestep == 0.002
  assert cfg.decimation == 10
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)
  assert cfg.events["reset_forward_velocity"].func is reset_heading_forward_velocity
  assert cfg.events["reset_forward_velocity"].params["speed_range"] == (0.75, 1.5)
  assert "upright" not in cfg.rewards
  assert "track_angular_velocity" not in cfg.rewards
  assert "uncommanded_yaw_rate" not in cfg.rewards
  direction_reward = cfg.rewards["velocity_direction_alignment"]
  assert direction_reward.func is track_commanded_velocity_direction
  assert direction_reward.weight == pytest.approx(
    FORWARD_REWARD_WEIGHTS["velocity_direction_alignment"]
  )
  assert direction_reward.params == {
    "command_name": "twist",
    "std": FORWARD_DIRECTION_REWARD_STD,
    "min_speed": FORWARD_DIRECTION_MIN_SPEED,
  }
  assert cfg.rewards["low_speed_world_up"].func is low_speed_world_up_alignment
  roll_reward = cfg.rewards["level_roll"]
  assert roll_reward.func is level_roll_alignment
  assert roll_reward.weight == pytest.approx(FORWARD_REWARD_WEIGHTS["level_roll"])
  assert roll_reward.params == {
    "std": FORWARD_ROLL_REWARD_STD,
    "min_horizontal_forward": FORWARD_ROLL_MIN_HORIZONTAL,
    "command_name": "twist",
    "full_speed": FORWARD_ROLL_FULL_SPEED,
    "zero_speed": FORWARD_ROLL_ZERO_SPEED,
  }
  roll_pitch_penalty = cfg.rewards["roll_pitch_angular_velocity"]
  assert roll_pitch_penalty.func is base_roll_pitch_angular_velocity_l2
  assert roll_pitch_penalty.weight == pytest.approx(
    FORWARD_REWARD_WEIGHTS["roll_pitch_angular_velocity"]
  )
  assert cfg.terminations["excessive_tilt"].params["limit_angle"] == pytest.approx(
    math.radians(FORWARD_TILT_LIMIT_DEG)
  )


def test_forward_task_uses_adjusted_original_model() -> None:
  cfg = load_env_cfg(FORWARD_TASK_ID)
  assert BIRD_5DOF_FORWARD_XML.exists()
  model = cfg.scene.entities["robot"].spec_fn().compile()
  base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bird")
  wing_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wing_aero")
  ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
  world_x_axis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "world_x_axis")
  world_y_axis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "world_y_axis")

  assert model.body_mass[base_id] == pytest.approx(0.208)
  assert model.body_mass.sum() == pytest.approx(0.258)
  assert model.geom_size[wing_id] == pytest.approx(
    [0.108225 * 1.1, 0.22375 * 1.1, 0.012 * 1.1]
  )
  assert model.geom_size[ground_id] == pytest.approx([80.0, 80.0, 0.1])
  assert model.geom_contype[ground_id] == 0
  assert model.geom_conaffinity[ground_id] == 0
  assert model.geom_contype[[world_x_axis_id, world_y_axis_id]].tolist() == [0, 0]
  assert model.geom_conaffinity[[world_x_axis_id, world_y_axis_id]].tolist() == [
    0,
    0,
  ]


def test_forward_task_has_separate_play_and_rl_configs() -> None:
  play_cfg = load_env_cfg(FORWARD_TASK_ID, play=True)
  command = play_cfg.commands["twist"]
  rl_cfg = load_rl_cfg(FORWARD_TASK_ID)

  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == (0.0, 10.0)
  assert command.ranges.lin_vel_y == (-4.0, 4.0)
  assert command.ranges.lin_vel_z == (-6.0, 6.0)
  assert command.ranges.ang_vel_z == (0.0, 0.0)
  assert command.debug_vis
  assert play_cfg.curriculum == {}
  assert rl_cfg.experiment_name == "bird_5dof_forward_3d"
  assert rl_cfg.max_iterations == 1000


def test_forward_task_does_not_mutate_omnidirectional_task() -> None:
  forward_cfg = load_env_cfg(FORWARD_TASK_ID)
  original_cfg = load_env_cfg(OMNIDIRECTIONAL_TASK_ID)
  original_command = original_cfg.commands["twist"]
  forward_command = forward_cfg.commands["twist"]

  assert isinstance(original_command, UniformVelocityCommandCfg)
  assert isinstance(forward_command, UniformVelocityCommandCfg)
  assert original_command.ranges == forward_command.ranges
  assert original_command.resampling_time_range == forward_command.resampling_time_range
  assert original_command.rel_standing_envs == forward_command.rel_standing_envs
  assert original_command.rel_world_envs == forward_command.rel_world_envs
  assert original_command.rel_forward_envs == forward_command.rel_forward_envs
  assert set(original_cfg.rewards) == set(forward_cfg.rewards)
  for name in original_cfg.rewards:
    assert original_cfg.rewards[name].func is forward_cfg.rewards[name].func
    assert original_cfg.rewards[name].weight == forward_cfg.rewards[name].weight
    assert original_cfg.rewards[name].params == forward_cfg.rewards[name].params
  assert original_cfg.curriculum
  assert original_cfg.terminations["excessive_tilt"].params[
    "limit_angle"
  ] == pytest.approx(math.radians(FORWARD_TILT_LIMIT_DEG))

  forward_command.ranges.lin_vel_x = (9.0, 10.0)
  fresh_original_command = load_env_cfg(OMNIDIRECTIONAL_TASK_ID).commands["twist"]
  assert isinstance(fresh_original_command, UniformVelocityCommandCfg)
  assert fresh_original_command.ranges.lin_vel_x == (0.0, 10.0)
  assert "reset_forward_velocity" in original_cfg.events


def test_forward_reset_speed_follows_each_birds_heading() -> None:
  robot = MagicMock()
  robot.data.root_link_vel_w = torch.zeros(2, 6)
  robot.data.heading_w = torch.tensor([0.0, torch.pi / 2])
  env = MagicMock()
  env.num_envs = 2
  env.device = "cpu"
  env.scene.__getitem__.return_value = robot

  reset_heading_forward_velocity(
    env,
    torch.arange(2),
    speed_range=(1.0, 1.0),
  )

  written_velocity = robot.write_root_link_velocity_to_sim.call_args.args[0]
  torch.testing.assert_close(
    written_velocity[:, :2],
    torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    atol=1.0e-6,
    rtol=0.0,
  )
