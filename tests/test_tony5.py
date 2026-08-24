"""Focused validation for the TONY5 V0 task."""

from typing import cast
from unittest.mock import Mock

import mujoco
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manager_based.tony5.tony5_actions import (
  Tony5RotorSpeedAction,
  map_normalized_action_to_speed,
)
from mjlab.tasks.manager_based.tony5.tony5_commands import (
  Tony5PositionYawCommand,
  Tony5PositionYawCommandCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  BATTERY_VOLTAGE,
  INTEGRAL_LIMIT_VELOCITY,
  KD_VELOCITY,
  KI_VELOCITY,
  KP_VELOCITY,
  KT_THRUST,
  OMEGA_HOVER,
  OMEGA_MAX,
  TOTAL_MASS,
)
from mjlab.tasks.manager_based.tony5.tony5_env_cfg import (
  tony5_position_yaw_env_cfg,
  tony5_position_yaw_play_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_velocity_env_cfg import (
  tony5_velocity_env_cfg,
  tony5_velocity_play_env_cfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.viewer.debug_visualizer import DebugVisualizer

POSITION_YAW_TASK_ID = "Mjlab-Tony5-PositionYaw-v0"
VELOCITY_TASK_ID = "Mjlab-Tony5-Velocity-v0"


@pytest.fixture(scope="module")
def device() -> str:
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_env(device: str, num_envs: int = 1) -> ManagerBasedRlEnv:
  cfg = tony5_position_yaw_env_cfg()
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def _rotor_action_term(env: ManagerBasedRlEnv) -> Tony5RotorSpeedAction:
  return cast(Tony5RotorSpeedAction, env.action_manager.get_term("rotor_speed"))


def _make_velocity_env(device: str, num_envs: int = 1) -> ManagerBasedRlEnv:
  cfg = tony5_velocity_env_cfg()
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def _position_command_term(env: ManagerBasedRlEnv) -> Tony5PositionYawCommand:
  return cast(Tony5PositionYawCommand, env.command_manager.get_term("position_yaw"))


def test_tony5_task_is_registered_and_loadable() -> None:
  import mjlab.tasks  # noqa: F401

  assert POSITION_YAW_TASK_ID in list_tasks()
  assert VELOCITY_TASK_ID in list_tasks()
  cfg = load_env_cfg(POSITION_YAW_TASK_ID)
  assert cfg.scene.entities["robot"] is not None
  assert cfg.actions["rotor_speed"] is not None

  velocity_cfg = load_env_cfg(VELOCITY_TASK_ID)
  assert velocity_cfg.commands["velocity"] is not None
  assert load_rl_cfg(VELOCITY_TASK_ID).experiment_name == "tony5_velocity_v0_history5"


def test_tony5_play_resamples_commands() -> None:
  cfg = tony5_position_yaw_play_env_cfg()
  command_cfg = cfg.commands["position_yaw"]
  assert isinstance(command_cfg, Tony5PositionYawCommandCfg)
  assert not command_cfg.fixed


def test_tony5_velocity_task_uses_body_twist_commands() -> None:
  cfg = tony5_velocity_play_env_cfg()
  command_cfg = cfg.commands["velocity"]
  assert isinstance(command_cfg, UniformVelocityCommandCfg)
  assert command_cfg.linear_velocity_frame == "body"
  assert command_cfg.yaw_velocity_frame == "body"
  assert command_cfg.ranges.lin_vel_z is not None
  assert command_cfg.rel_standing_envs == pytest.approx(0.2)
  assert cfg.observations["actor"].history_length == 5
  assert cfg.observations["actor"].flatten_history_dim
  assert cfg.observations["critic"].history_length == 5
  assert cfg.observations["critic"].flatten_history_dim
  assert cfg.rewards["linear_velocity_tracking"].params["std"] == pytest.approx(0.8)
  assert cfg.rewards["angular_velocity_tracking"].params["std"] == pytest.approx(0.9)
  assert cfg.rewards["action_rate"].weight == pytest.approx(-0.05)
  assert "ground" not in cfg.terminations
  assert cfg.scene.num_envs == 1
  assert cfg.episode_length_s > 1e8


def test_tony5_velocity_env_shape_and_debug_visualization(device: str) -> None:
  env = _make_velocity_env(device)
  try:
    obs, _ = env.reset()
    assert isinstance(obs["actor"], torch.Tensor)
    assert obs["actor"].shape == (1, 110)
    assert isinstance(obs["critic"], torch.Tensor)
    assert obs["critic"].shape == (1, 110)
    command = cast(UniformVelocityCommand, env.command_manager.get_term("velocity"))
    assert command.command.shape == (1, 4)
    ground_id = mujoco.mj_name2id(
      env.sim.mj_model,
      mujoco.mjtObj.mjOBJ_GEOM,
      "terrain",
    )
    assert ground_id >= 0
    assert env.sim.mj_model.geom_contype[ground_id] == 0
    assert env.sim.mj_model.geom_conaffinity[ground_id] == 0
    assert "ground" not in env.termination_manager.active_terms

    visualizer = Mock()
    visualizer.get_env_indices.return_value = [0]
    command._debug_vis_impl(cast(DebugVisualizer, visualizer))
    assert visualizer.add_arrow.call_count == 4

    obs, reward, terminated, truncated, _ = env.step(torch.zeros((1, 4), device=device))
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert actor_obs.shape == (1, 110)
    assert isinstance(obs["critic"], torch.Tensor)
    assert obs["critic"].shape == (1, 110)
    assert torch.isfinite(reward).all()
    assert not torch.isnan(terminated.float()).any()
    assert not torch.isnan(truncated.float()).any()
  finally:
    env.close()


def test_zero_action_maps_to_hover_speed() -> None:
  actions = torch.tensor([[-1.0, 0.0, 1.0, 0.25]])
  speeds = map_normalized_action_to_speed(actions)
  expected = torch.tensor(
    [[0.0, OMEGA_HOVER, OMEGA_MAX, 836.0 + 0.25 * (OMEGA_MAX - 836.0)]]
  )
  assert torch.allclose(speeds, expected)


def test_hover_thrust_matches_weight() -> None:
  total_thrust = 4.0 * KT_THRUST * OMEGA_HOVER**2
  weight = TOTAL_MASS * 9.80665
  assert total_thrust == pytest.approx(weight, rel=0.01)


def test_tony5_entity_and_observation_shape(device: str) -> None:
  env = _make_env(device)
  try:
    actor_obs = env.reset()[0]["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert actor_obs.shape == (1, 22)
    robot = env.scene["robot"]
    assert robot.num_actuators == 4
    actuator_gainprm = env.sim.model.actuator_gainprm[0]
    assert actuator_gainprm.shape[0] == 4
    assert torch.allclose(
      actuator_gainprm[:, 4],
      torch.zeros(4, device=device),
    )
    assert torch.allclose(
      actuator_gainprm[:, 5],
      torch.zeros(4, device=device),
    )
    assert torch.allclose(
      actuator_gainprm[:, 6],
      torch.zeros(4, device=device),
    )
    assert torch.allclose(
      actuator_gainprm[:, 7],
      torch.full((4,), BATTERY_VOLTAGE, device=device),
    )
    assert torch.allclose(
      actuator_gainprm[:, 8],
      torch.zeros(4, device=device),
    )
    assert env.cfg.commands["position_yaw"].debug_vis
    assert KP_VELOCITY > 0.0
    assert KI_VELOCITY > 0.0
    assert KD_VELOCITY > 0.0
    assert INTEGRAL_LIMIT_VELOCITY > 0.0

    term = _rotor_action_term(env)
    env.action_manager.process_action(torch.full((1, 4), 0.5, device=device))
    term.apply_actions()
    assert torch.any(term.last_voltage != 0.0)
    assert torch.all(term.last_voltage.abs() <= BATTERY_VOLTAGE)
    assert tuple(robot.find_joints("rotor_.*_joint", preserve_order=True)[1]) == (
      "rotor_fl_joint",
      "rotor_fr_joint",
      "rotor_rr_joint",
      "rotor_rl_joint",
    )
    assert tuple(robot.find_sites("rotor_.*_site", preserve_order=True)[1]) == (
      "rotor_fl_site",
      "rotor_fr_site",
      "rotor_rr_site",
      "rotor_rl_site",
    )
    assert env.sim.model.body_mass[0].sum().item() == pytest.approx(TOTAL_MASS)
    assert env.physics_dt == pytest.approx(0.001)
    assert env.step_dt == pytest.approx(0.01)
  finally:
    env.close()


def test_tony5_target_debug_visualization(device: str) -> None:
  env = _make_env(device)
  try:
    env.reset()
    visualizer = Mock()
    visualizer.get_env_indices.return_value = [0]
    _position_command_term(env)._debug_vis_impl(cast(DebugVisualizer, visualizer))
    visualizer.add_sphere.assert_called_once()
    visualizer.add_arrow.assert_called_once()
  finally:
    env.close()


def test_thrust_uses_actual_qvel_not_target(device: str) -> None:
  env = _make_env(device)
  try:
    env.reset()
    term = _rotor_action_term(env)
    env.action_manager.process_action(torch.zeros((1, 4), device=device))
    target_wrench = term._aerodynamics.compute_wrenches()[0].clone()

    robot = env.scene["robot"]
    robot.write_joint_velocity_to_sim(
      torch.full((1, 4), OMEGA_MAX, device=device),
      joint_ids=term._aerodynamics.joint_ids,
    )
    env.sim.forward()
    actual_wrench = term._aerodynamics.compute_wrenches()[0]
    actual_magnitude = torch.linalg.vector_norm(actual_wrench, dim=-1)
    target_magnitude = torch.linalg.vector_norm(target_wrench, dim=-1)
    assert torch.all(actual_magnitude > target_magnitude)
  finally:
    env.close()


def test_collective_force_and_yaw_torque_cancellation(device: str) -> None:
  env = _make_env(device)
  try:
    env.reset()
    term = _rotor_action_term(env)
    robot = env.scene["robot"]
    robot.write_root_link_pose_to_sim(
      torch.tensor([[0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0]], device=device)
    )
    robot.write_joint_velocity_to_sim(
      torch.full((1, 4), OMEGA_HOVER, device=device),
      joint_ids=term._aerodynamics.joint_ids,
    )
    env.sim.forward()
    forces, torques = term._aerodynamics.compute_wrenches()
    assert torch.allclose(forces[0, :, :2], torch.zeros((4, 2), device=device))
    assert torch.allclose(torques.sum(dim=1), torch.zeros((1, 3), device=device))
    assert forces[0, :, 2].sum().item() == pytest.approx(
      4.0 * KT_THRUST * OMEGA_HOVER**2,
      rel=1e-5,
    )
  finally:
    env.close()


def test_collective_differential_force_moment_signs(device: str) -> None:
  env = _make_env(device)
  try:
    env.reset()
    term = _rotor_action_term(env)
    robot = env.scene["robot"]
    robot.write_root_link_pose_to_sim(
      torch.tensor([[0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0]], device=device)
    )
    robot.write_joint_velocity_to_sim(
      torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device),
      joint_ids=term._aerodynamics.joint_ids,
    )
    env.sim.forward()
    body_ids = term._aerodynamics.body_ids
    global_body_ids = robot.data.indexing.body_ids[body_ids]
    root_id = robot.data.indexing.root_body_id
    root_pos = env.sim.data.xpos[:, root_id]
    rotor_pos = env.sim.data.xpos[:, global_body_ids]
    rel_pos = rotor_pos - root_pos[:, None, :]

    for speeds, expected_axis, description in (
      ([1.1, 0.9, 0.9, 1.1], 0, "roll"),
      ([1.1, 1.1, 0.9, 0.9], 1, "pitch"),
    ):
      robot.write_joint_velocity_to_sim(
        torch.tensor([speeds], device=device) * OMEGA_HOVER,
        joint_ids=term._aerodynamics.joint_ids,
      )
      env.sim.forward()
      forces, _ = term._aerodynamics.compute_wrenches()
      moment = torch.cross(rel_pos, forces, dim=-1).sum(dim=1)[0]
      assert moment[expected_axis] != 0.0, description
      assert (
        moment[expected_axis] > 0.0
        if description == "roll"
        else moment[expected_axis] < 0.0
      )
  finally:
    env.close()


@pytest.mark.slow
def test_tony5_batched_rollout(device: str) -> None:
  env = _make_env(device, num_envs=64)
  try:
    obs, _ = env.reset()
    actor_obs = obs["actor"]
    assert isinstance(actor_obs, torch.Tensor)
    assert actor_obs.shape == (64, 22)
    for _ in range(300):
      action = 2.0 * torch.rand((64, 4), device=device) - 1.0
      obs, _, terminated, truncated, _ = env.step(action)
      actor_obs = obs["actor"]
      assert isinstance(actor_obs, torch.Tensor)
      assert torch.isfinite(actor_obs).all()
      assert torch.isfinite(env.sim.data.qvel).all()
      assert torch.isfinite(env.scene["robot"].data.root_link_quat_w).all()
      assert not torch.isnan(terminated.float()).any()
      assert not torch.isnan(truncated.float()).any()
  finally:
    env.close()
