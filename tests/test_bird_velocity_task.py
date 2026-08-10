"""Tests for the five-DoF bird velocity task."""

import math

import mujoco
import pytest

from mjlab.asset_zoo.robots import (
  BIRD_5DOF_ACTION_SCALE,
  get_bird_5dof_robot_cfg,
)
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.observations import base_lin_vel_heading
from mjlab.envs.mdp.terminations import bad_orientation
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.config.bird_5dof.env_cfgs import (
  REWARD_WEIGHTS,
  VELOCITY_CURRICULUM_STAGES,
)
from mjlab.tasks.velocity.config.bird_5dof.events import (
  reset_heading_forward_velocity,
)
from mjlab.tasks.velocity.config.bird_5dof.flight_profile import (
  ROLL_FULL_SPEED,
  ROLL_ZERO_SPEED,
  TILT_LIMIT_DEG,
)
from mjlab.tasks.velocity.config.bird_5dof.rl_cfg import bird_5dof_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.rewards import (
  base_roll_pitch_angular_velocity_l2,
  level_roll_alignment,
  low_speed_world_up_alignment,
  track_commanded_velocity_direction,
  track_linear_velocity_heading,
)

TASK_ID = "Mjlab-Velocity-Bird-5DoF"
EXPECTED_JOINTS = [
  "left_flap",
  "left_feather",
  "right_flap",
  "right_feather",
  "tail_pitch",
]


def test_bird_model_structure() -> None:
  model = get_bird_5dof_robot_cfg().build().compile()
  ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
  world_x_axis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "world_x_axis")
  world_y_axis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "world_y_axis")
  left_wing_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_wing_aero")
  right_wing_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_wing_aero")

  assert model.nu == 5
  assert model.nq == 12
  assert model.nv == 11
  assert model.body_mass.sum() == pytest.approx(0.258)
  assert model.geom_size[left_wing_id].tolist() == pytest.approx(
    [0.1190475, 0.246125, 0.0132]
  )
  assert model.geom_size[right_wing_id].tolist() == pytest.approx(
    [0.1190475, 0.246125, 0.0132]
  )
  assert ground_id >= 0
  assert model.geom_size[ground_id].tolist() == pytest.approx([80.0, 80.0, 0.1])
  assert model.geom_contype[ground_id] == 0
  assert model.geom_conaffinity[ground_id] == 0
  assert model.geom_contype[[world_x_axis_id, world_y_axis_id]].tolist() == [0, 0]
  assert model.geom_conaffinity[[world_x_axis_id, world_y_axis_id]].tolist() == [
    0,
    0,
  ]
  assert model.vis.map.fogstart == pytest.approx(60.0)
  assert model.vis.map.fogend == pytest.approx(200.0)
  assert [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
  ] == [
    "left_flap_act",
    "left_feather_act",
    "right_flap_act",
    "right_feather_act",
    "tail_pitch_act",
  ]
  assert model.actuator_gainprm[:, 0].tolist() == pytest.approx(
    [6.0, 1.5, 6.0, 1.5, 1.5]
  )
  assert model.actuator_biasprm[:, 2].tolist() == pytest.approx(
    [-0.12, -0.015, -0.12, -0.015, -0.02]
  )
  assert model.actuator_forcerange.flatten().tolist() == pytest.approx(
    [-5.0, 5.0, -2.0, 2.0, -5.0, 5.0, -2.0, 2.0, -1.5, 1.5]
  )


def test_bird_velocity_training_config() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert load_rl_cfg(TASK_ID) is not None
  agent_cfg = bird_5dof_ppo_runner_cfg()
  action = cfg.actions["joint_pos"]
  command = cfg.commands["twist"]

  assert agent_cfg.actor.hidden_dims == (512, 256, 128)
  assert agent_cfg.critic.hidden_dims == (512, 256, 128)
  assert agent_cfg.actor.distribution_cfg is not None
  assert agent_cfg.actor.distribution_cfg["init_std"] == pytest.approx(0.6)
  assert agent_cfg.actor.distribution_cfg["std_range"] == (0.5, 1.0)
  assert agent_cfg.algorithm.entropy_coef == pytest.approx(0.001)
  assert agent_cfg.max_iterations == 10000
  assert isinstance(action, JointPositionActionCfg)
  assert isinstance(command, UniformVelocityCommandCfg)
  assert cfg.scene.num_envs > 1
  assert cfg.sim.mujoco.timestep == 0.002
  assert cfg.decimation == 10
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)
  assert cfg.sim.mujoco.integrator == "implicitfast"
  assert cfg.episode_length_s == 20.0
  assert action.scale == BIRD_5DOF_ACTION_SCALE
  assert command.ranges.lin_vel_x == (0.0, 10.0)
  assert command.ranges.lin_vel_y == (-4.0, 4.0)
  assert command.ranges.lin_vel_z == (-6.0, 6.0)
  assert command.ranges.ang_vel_z == (0.0, 0.0)
  assert command.resampling_time_range == (2.0, 4.0)
  assert command.rel_standing_envs == pytest.approx(0.0)
  assert command.rel_world_envs == pytest.approx(1.0)
  assert command.rel_forward_envs == pytest.approx(0.0)
  assert command.linear_velocity_frame == "heading"
  assert command.yaw_velocity_frame == "world"
  assert command.linear_velocity_sampling == "ellipsoid"
  assert command.yaw_command_speed_threshold is None
  assert cfg.scene.entities["robot"].init_state.lin_vel == (0.0, 0.0, 0.0)
  assert cfg.observations["actor"].terms["base_lin_vel"].func is base_lin_vel_heading
  pose_range = cfg.events["reset_base"].params["pose_range"]
  assert pose_range["roll"] == (-0.1, 0.1)
  assert pose_range["pitch"] == (-0.1, 0.1)
  assert pose_range["yaw"] == pytest.approx((-math.pi, math.pi))
  assert set(cfg.events) == {
    "reset_base",
    "reset_flap_joints",
    "reset_feather_joints",
    "reset_tail_joint",
    "reset_forward_velocity",
  }
  assert cfg.events["reset_forward_velocity"].func is reset_heading_forward_velocity
  assert set(cfg.terminations) == {
    "time_out",
    "non_finite_state",
    "excessive_tilt",
  }
  assert cfg.terminations["excessive_tilt"].func is bad_orientation
  assert cfg.terminations["excessive_tilt"].params["limit_angle"] == pytest.approx(
    math.radians(TILT_LIMIT_DEG)
  )
  assert cfg.rewards["track_linear_velocity"].func is track_linear_velocity_heading
  assert (
    cfg.rewards["velocity_direction_alignment"].func
    is track_commanded_velocity_direction
  )
  assert cfg.rewards["velocity_direction_alignment"].weight == pytest.approx(
    REWARD_WEIGHTS["velocity_direction_alignment"]
  )
  assert cfg.rewards["level_roll"].func is level_roll_alignment
  assert cfg.rewards["level_roll"].weight == pytest.approx(REWARD_WEIGHTS["level_roll"])
  assert cfg.rewards["level_roll"].params["command_name"] == "twist"
  assert cfg.rewards["level_roll"].params["full_speed"] == pytest.approx(
    ROLL_FULL_SPEED
  )
  assert cfg.rewards["level_roll"].params["zero_speed"] == pytest.approx(
    ROLL_ZERO_SPEED
  )
  assert (
    cfg.rewards["roll_pitch_angular_velocity"].func
    is base_roll_pitch_angular_velocity_l2
  )
  assert cfg.rewards["low_speed_world_up"].func is low_speed_world_up_alignment
  assert "track_angular_velocity" not in cfg.rewards
  assert "uncommanded_yaw_rate" not in cfg.rewards
  assert "upright" not in cfg.rewards
  stages = cfg.curriculum["command_vel"].params["velocity_stages"]
  assert stages == VELOCITY_CURRICULUM_STAGES
  assert [stage["step"] for stage in stages] == [
    0,
    200 * 24,
    400 * 24,
    600 * 24,
    800 * 24,
    1000 * 24,
  ]
  assert [stage["name"] for stage in stages] == [
    "proven_forward_3d",
    "hover_and_stall_recovery",
    "full_world_directions",
    "rapid_redirection",
    "moderate_disturbances",
    "robustness",
  ]
  assert [stage["rel_standing_envs"] for stage in stages] == [
    0.0,
    0.15,
    0.1,
    0.1,
    0.1,
    0.1,
  ]
  assert [stage["lin_vel_x"] for stage in stages] == [
    (0.0, 10.0),
    (0.0, 10.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
    (-10.0, 10.0),
  ]
  assert [stage["rel_forward_envs"] for stage in stages] == [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
  ]
  assert [stage["resampling_time_range"] for stage in stages] == [
    (2.0, 4.0),
    (2.0, 4.0),
    (2.0, 4.0),
    (1.5, 3.0),
    (1.5, 3.0),
    (1.5, 3.0),
  ]
  assert [stage["lin_vel_z"] for stage in stages] == [
    (-6.0, 6.0),
    (-6.0, 6.0),
    (-6.0, 6.0),
    (-6.0, 6.0),
    (-6.0, 6.0),
    (-6.0, 6.0),
  ]
  assert [stage["ang_vel_z"] for stage in stages] == [
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
  ]
  assert stages[0]["joint_position_ranges"] == {
    "reset_flap_joints": (-0.2, 0.2),
    "reset_feather_joints": (-0.2, 0.2),
    "reset_tail_joint": (-0.1, 0.1),
  }
  assert stages[-1]["reset_velocity_range"]["yaw"] == (-1.5, 1.5)


def test_bird_scene_preserves_atmosphere() -> None:
  cfg = load_env_cfg(TASK_ID, play=True)
  command = cfg.commands["twist"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert cfg.curriculum == {}
  assert command.ranges.lin_vel_x == VELOCITY_CURRICULUM_STAGES[-1]["lin_vel_x"]
  assert command.ranges.lin_vel_y == VELOCITY_CURRICULUM_STAGES[-1]["lin_vel_y"]
  assert command.ranges.lin_vel_z == VELOCITY_CURRICULUM_STAGES[-1]["lin_vel_z"]
  assert command.ranges.ang_vel_z == VELOCITY_CURRICULUM_STAGES[-1]["ang_vel_z"]
  assert (
    command.rel_standing_envs == VELOCITY_CURRICULUM_STAGES[-1]["rel_standing_envs"]
  )
  assert (
    command.resampling_time_range
    == VELOCITY_CURRICULUM_STAGES[-1]["resampling_time_range"]
  )
  scene = Scene(cfg.scene, device="cpu")
  model = scene.compile()

  assert model.opt.density == pytest.approx(1.225)
  assert model.opt.viscosity == pytest.approx(1.81e-5)
  assert model.opt.wind.tolist() == pytest.approx([0.0, 0.0, 0.0])
  assert [
    name.removeprefix("robot/") for name in scene.entities["robot"].joint_names
  ] == EXPECTED_JOINTS
