"""Focused validation for the TONY5 V1 aerodynamic task."""

from typing import cast

import mujoco
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manager_based.tony5 import (
  VELOCITY_AERO_OMNI_TASK_ID,
  VELOCITY_AERO_TASK_ID,
  VELOCITY_TASK_ID,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_actions import (
  Tony5AeroRotorSpeedAction,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_env_cfg import (
  tony5_velocity_aero_env_cfg,
  tony5_velocity_aero_play_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_physics import (
  BODY_AERO_FLUID_COEFS,
  BODY_AERO_POS,
  BODY_AERO_SIZE,
  BODY_AIR_DENSITY,
  BODY_AIR_VISCOSITY,
  ROTOR_DRAG_KH,
  ROTOR_FLUID_DISABLE_SIZE,
  Tony5AeroRotorAerodynamics,
  get_tony5_aero_spec,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_rewards import (
  downward_velocity_l2,
  high_speed_roll_uprightness_reward,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_safety import (
  BODY_ANGULAR_SPEED_LIMIT,
  QACC_LIMIT,
  ROOT_SPEED_LIMIT,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_terminations import (
  TONY5_AERO_DOWNWARD_VELOCITY_LIMIT,
  numerical_safety_failure,
  rapid_descent_termination,
)
from mjlab.tasks.manager_based.tony5.tony5_constants import (
  BATTERY_VOLTAGE,
  KQ_TORQUE,
  KT_THRUST,
  OMEGA_HOVER,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_command import (
  Tony5OmniVelocityCommandCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_env_cfg import (
  TONY5_OMNI_COMMAND_TRANSITION_TIME_S,
  TONY5_OMNI_MAX_HORIZONTAL_SPEED,
  tony5_velocity_aero_omni_env_cfg,
  tony5_velocity_aero_omni_play_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_rewards import (
  TONY5_OMNI_REWARD_WEIGHTS,
)
from mjlab.tasks.manager_based.tony5.tony5_velocity_curriculum import (
  Tony5RadialVelocityCommand,
)
from mjlab.tasks.manager_based.tony5.tony5_velocity_env_cfg import (
  tony5_velocity_env_cfg,
)
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz


@pytest.fixture(scope="module")
def device() -> str:
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_aero_env(device: str, num_envs: int = 1) -> ManagerBasedRlEnv:
  cfg = tony5_velocity_aero_env_cfg()
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def _aero_action(env: ManagerBasedRlEnv) -> Tony5AeroRotorSpeedAction:
  return cast(
    Tony5AeroRotorSpeedAction,
    env.action_manager.get_term("rotor_speed"),
  )


def _aero_physics(env: ManagerBasedRlEnv) -> Tony5AeroRotorAerodynamics:
  return cast(Tony5AeroRotorAerodynamics, _aero_action(env)._aerodynamics)


def test_tony5_aero_task_is_registered_and_preserves_v0_task() -> None:
  import mjlab.tasks  # noqa: F401

  assert VELOCITY_TASK_ID in list_tasks()
  assert VELOCITY_AERO_TASK_ID in list_tasks()
  assert VELOCITY_AERO_OMNI_TASK_ID in list_tasks()
  assert load_rl_cfg(VELOCITY_AERO_TASK_ID).experiment_name == "tony5_velocity_aero_v1"
  assert (
    load_rl_cfg(VELOCITY_AERO_OMNI_TASK_ID).experiment_name
    == "tony5_velocity_aero_omni_v0"
  )

  v0_cfg = tony5_velocity_env_cfg()
  v1_cfg = tony5_velocity_aero_env_cfg()
  assert not v0_cfg.sim.nan_guard.enabled
  assert v1_cfg.sim.nan_guard.enabled
  assert v0_cfg.commands.keys() == v1_cfg.commands.keys()
  assert set(v1_cfg.rewards) == (set(v0_cfg.rewards) - {"uprightness"}) | {
    "body_angular_acceleration",
    "downward_velocity",
    "uprightness",
  }
  assert "attitude" in v0_cfg.terminations
  assert "attitude" not in v1_cfg.terminations
  assert v0_cfg.terminations["time_out"].log
  assert not v1_cfg.terminations["time_out"].log
  assert set(v1_cfg.terminations) == {
    "time_out",
    "rapid_descent",
    "numerical_safety_failure",
  }
  assert "downward_velocity" in v1_cfg.rewards
  assert not v1_cfg.rewards["body_angular_acceleration"].log
  assert not v1_cfg.rewards["uprightness"].log
  assert (
    v1_cfg.scene.entities["robot"].spec_fn is not v0_cfg.scene.entities["robot"].spec_fn
  )


def test_tony5_aero_omni_task_has_fixed_radial_10_mps_commands() -> None:
  cfg = tony5_velocity_aero_omni_env_cfg()
  play_cfg = tony5_velocity_aero_omni_play_env_cfg()
  command = cfg.commands["velocity"]
  assert isinstance(command, Tony5OmniVelocityCommandCfg)
  assert command.stage_speeds == (TONY5_OMNI_MAX_HORIZONTAL_SPEED,)
  assert command.curriculum_switch_iterations == ()
  assert command.previous_speed_fraction == 0.0
  assert command.current_speed_fraction == 1.0
  assert command.high_speed_fraction == 0.0
  assert command.ranges.lin_vel_x == (-10.0, 10.0)
  assert command.ranges.lin_vel_y == (-10.0, 10.0)
  assert command.rel_standing_envs == 0.2
  assert command.command_transition_time_s == pytest.approx(
    TONY5_OMNI_COMMAND_TRANSITION_TIME_S
  )
  assert play_cfg.scene.num_envs == 1
  assert "curriculum_stage" not in cfg.metrics
  assert "current_vmax" not in cfg.metrics
  assert set(cfg.rewards) == set(TONY5_OMNI_REWARD_WEIGHTS)
  for name, weight in TONY5_OMNI_REWARD_WEIGHTS.items():
    assert cfg.rewards[name].weight == pytest.approx(weight)


def test_tony5_aero_omni_sampling_stays_within_10_mps(device: str) -> None:
  cfg = tony5_velocity_aero_omni_env_cfg()
  cfg.scene.num_envs = 4096
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    command = cast(
      Tony5RadialVelocityCommand,
      env.command_manager.get_term("velocity"),
    )
    env_ids = torch.arange(env.num_envs, device=device)
    command._resample_command(env_ids)
    horizontal_speed = torch.linalg.vector_norm(command.target_command[:, :2], dim=-1)
    assert torch.all(horizontal_speed <= 10.0 + 1.0e-5)
    assert torch.mean(command.is_standing_env.float()).item() == pytest.approx(
      0.2, abs=0.03
    )
    moving = ~command.is_standing_env
    assert torch.any(command.target_command[moving, 0] > 0.0)
    assert torch.any(command.target_command[moving, 0] < 0.0)
    assert torch.any(command.target_command[moving, 1] > 0.0)
    assert torch.any(command.target_command[moving, 1] < 0.0)
    command.compute(0.01)
    assert torch.any(
      torch.linalg.vector_norm(command.command[moving, :2], dim=-1)
      < horizontal_speed[moving]
    )
  finally:
    env.close()


def test_tony5_aero_downward_velocity_penalty_uses_actual_velocity_only(
  device: str,
) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    command = cast(
      Tony5RadialVelocityCommand,
      env.command_manager.get_term("velocity"),
    )

    command.vel_command_b[0, 3] = -0.75
    robot.write_root_link_velocity_b_to_sim(torch.zeros((1, 6), device=device))
    env.sim.forward()
    assert downward_velocity_l2(env).item() == 0.0

    command.vel_command_b[0, 3] = 0.75
    robot.write_root_link_velocity_b_to_sim(
      torch.tensor([[0.0, 0.0, -3.1, 0.0, 0.0, 0.0]], device=device)
    )
    env.sim.forward()
    assert not rapid_descent_termination(env).item()
    assert downward_velocity_l2(env).item() == pytest.approx(0.01, abs=1.0e-6)
    robot.write_root_link_velocity_b_to_sim(
      torch.tensor([[0.0, 0.0, -5.1, 0.0, 0.0, 0.0]], device=device)
    )
    env.sim.forward()
    assert rapid_descent_termination(env).item()
    assert TONY5_AERO_DOWNWARD_VELOCITY_LIMIT == -5.0
  finally:
    env.close()


def test_tony5_aero_uprightness_reward_only_applies_above_8_mps(
  device: str,
) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    command = cast(
      Tony5RadialVelocityCommand,
      env.command_manager.get_term("velocity"),
    )
    command.vel_command_b.zero_()
    assert high_speed_roll_uprightness_reward(env).item() == 0.0

    command.vel_command_b[0, 0] = 8.0
    assert high_speed_roll_uprightness_reward(env).item() == 0.0

    command.vel_command_b[0, 0] = 8.01
    robot = env.scene["robot"]
    zero = torch.zeros(1, device=device)
    pitch = torch.full((1,), 0.5, device=device)
    pitch_quat = quat_from_euler_xyz(zero, pitch, zero)
    robot.write_root_link_pose_to_sim(
      torch.cat(
        (torch.tensor([[0.0, 0.0, 1.5]], device=device), pitch_quat),
        dim=-1,
      )
    )
    env.sim.forward()
    assert high_speed_roll_uprightness_reward(env).item() == pytest.approx(1.0)

    roll = torch.full((1,), 0.5, device=device)
    roll_quat = quat_from_euler_xyz(roll, zero, zero)
    robot.write_root_link_pose_to_sim(
      torch.cat(
        (torch.tensor([[0.0, 0.0, 1.5]], device=device), roll_quat),
        dim=-1,
      )
    )
    env.sim.forward()
    assert high_speed_roll_uprightness_reward(env).item() < 1.0
  finally:
    env.close()


def test_tony5_aero_spec_has_expected_fluid_geom() -> None:
  spec = get_tony5_aero_spec()
  body_aero = spec.geom("body_aero")
  assert body_aero.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
  assert tuple(body_aero.size) == BODY_AERO_SIZE
  assert tuple(body_aero.pos) == BODY_AERO_POS
  assert body_aero.contype == 0
  assert body_aero.conaffinity == 0
  assert body_aero.density == 0.0
  assert body_aero.fluid_ellipsoid == 1.0
  assert tuple(body_aero.fluid_coefs) == BODY_AERO_FLUID_COEFS
  assert body_aero.rgba[3] == 0.0
  assert spec.option.density == BODY_AIR_DENSITY
  assert spec.option.viscosity == BODY_AIR_VISCOSITY

  model = spec.compile()
  geom_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "body_aero",
  )
  assert geom_id >= 0
  assert model.geom_contype[geom_id] == 0
  assert model.geom_conaffinity[geom_id] == 0
  for name in ("top_frame", "front_led", "rear_led"):
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) == -1
  for rotor_name in ("rotor_fl", "rotor_fr", "rotor_rr", "rotor_rl"):
    fluid_disable = spec.geom(f"{rotor_name}_fluid_disable")
    assert fluid_disable.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID
    assert tuple(fluid_disable.size) == ROTOR_FLUID_DISABLE_SIZE
    assert fluid_disable.density == 0.0
    assert fluid_disable.contype == 0
    assert fluid_disable.conaffinity == 0
    assert fluid_disable.fluid_ellipsoid == 1.0
    assert tuple(fluid_disable.fluid_coefs) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert fluid_disable.rgba[3] == 0.0


def test_tony5_aero_zero_airflow_has_zero_h_force(device: str) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    aero = _aero_physics(env)
    aero.compute_wrenches()
    assert torch.allclose(
      aero.local_air_velocity_b,
      torch.zeros_like(aero.local_air_velocity_b),
      atol=1e-6,
    )
    assert torch.allclose(
      aero.h_force_w,
      torch.zeros_like(aero.h_force_w),
      atol=1e-6,
    )

    fluid = env.sim.data.qfrc_fluid
    root_dof_ids = env.scene["robot"].indexing.free_joint_v_adr.to(device)
    assert torch.allclose(
      fluid[:, root_dof_ids[:3]],
      torch.zeros((1, 3), device=device),
      atol=1e-6,
    )
  finally:
    env.close()


def test_tony5_aero_rotor_fluid_load_is_disabled(device: str) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    action = _aero_action(env)
    rotor_dofs = robot.indexing.joint_v_adr
    for speed in (1000.0, 1500.0, 2000.0, 2500.0):
      rotor_speed = torch.full((1, 4), speed, device=device)
      robot.write_joint_velocity_to_sim(rotor_speed, joint_ids=action._joint_ids)
      env.scene.write_data_to_sim()
      env.sim.forward()
      rotor_fluid = env.sim.data.qfrc_fluid[:, rotor_dofs]
      assert torch.allclose(rotor_fluid, torch.zeros_like(rotor_fluid), atol=1e-7)
  finally:
    env.close()


def test_tony5_aero_esc_voltage_is_unidirectional(device: str) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    action = _aero_action(env)
    high_speed = torch.full((1, 4), 1800.0, device=device)
    robot.write_joint_velocity_to_sim(high_speed, joint_ids=action._joint_ids)
    env.scene.write_data_to_sim()
    env.sim.forward()

    action._speed_targets.zero_()
    action.apply_actions()
    assert torch.all(action.last_voltage >= 0.0)
    assert torch.all(action.last_voltage <= BATTERY_VOLTAGE)
    assert torch.allclose(action.last_voltage, torch.zeros_like(action.last_voltage))

    action.process_actions(torch.ones((1, 4), device=device))
    action.apply_actions()
    assert torch.all(action.last_voltage >= 0.0)
    assert torch.all(action.last_voltage <= BATTERY_VOLTAGE)
  finally:
    env.close()


def test_tony5_aero_forward_airflow_opposes_motion(device: str) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    robot.write_root_link_velocity_b_to_sim(
      torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    )
    env.sim.forward()

    action = _aero_action(env)
    aero = _aero_physics(env)
    aero.compute_wrenches()
    local_airflow = aero.local_air_velocity_b
    h_force = aero.h_force_w
    assert torch.allclose(local_airflow[..., 0], torch.ones((1, 4), device=device))
    assert torch.allclose(local_airflow[..., 1:], torch.zeros((1, 4, 2), device=device))
    assert torch.all(h_force[..., 0] < 0.0)
    assert torch.allclose(h_force[..., 1:], torch.zeros((1, 4, 2), device=device))
    assert torch.allclose(
      h_force[..., 0],
      -ROTOR_DRAG_KH * robot.data.joint_vel[:, action._joint_ids].abs(),
      atol=1e-6,
    )

    root_dof_ids = robot.indexing.free_joint_v_adr.to(device)
    fluid = env.sim.data.qfrc_fluid[:, root_dof_ids]
    assert torch.all(fluid[:, 0] < 0.0)
  finally:
    env.close()


def test_tony5_aero_rotational_airflow_differs_by_rotor(device: str) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    robot.write_root_link_velocity_b_to_sim(
      torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device)
    )
    env.sim.forward()

    aero = _aero_physics(env)
    aero.compute_wrenches()
    values = aero.local_air_velocity_b[0]
    assert torch.linalg.vector_norm(values[1:] - values[0], dim=-1).max() > 1e-5
    assert torch.allclose(values[:, 2], torch.zeros(4, device=device), atol=1e-6)
  finally:
    env.close()


def test_tony5_aero_equal_rotors_have_no_static_attitude_bias(device: str) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    action = _aero_action(env)
    speeds = torch.full((1, 4), OMEGA_HOVER, device=device)
    robot.write_joint_velocity_to_sim(speeds, joint_ids=action._joint_ids)
    env.sim.forward()
    forces, torques = action._aerodynamics.compute_wrenches()

    relative_positions = robot.data.body_link_pos_w[
      :, action._aerodynamics.body_ids
    ] - robot.data.root_link_pos_w.unsqueeze(1)
    total_moment = torch.cross(relative_positions, forces, dim=-1).sum(dim=1)
    assert torch.allclose(
      total_moment[:, :2], torch.zeros((1, 2), device=device), atol=1e-5
    )
    assert torch.allclose(
      torques.sum(dim=1), torch.zeros((1, 3), device=device), atol=1e-6
    )
  finally:
    env.close()


def test_tony5_aero_reaction_torque_opposes_physical_spin(device: str) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    aero = _aero_physics(env)
    signed_speed = torch.tensor([[1000.0, -1500.0, 2000.0, -2500.0]], device=device)
    robot.write_joint_velocity_to_sim(signed_speed, joint_ids=aero.joint_ids)
    env.sim.forward()

    _, torques = aero.compute_wrenches()
    global_body_ids = robot.data.indexing.body_ids[aero.body_ids]
    body_xmat = env.sim.data.xmat[:, global_body_ids].reshape(1, 4, 3, 3)
    physical_axis_w = body_xmat[..., :, 2] * aero._axis_signs.view(1, 4, 1)
    omega_vec_w = signed_speed.unsqueeze(-1) * physical_axis_w
    assert torch.all(torch.sum(omega_vec_w * torques, dim=-1) < 0.0)

    expected = (
      -KQ_TORQUE * (signed_speed * signed_speed.abs()).unsqueeze(-1) * physical_axis_w
    )
    assert torch.allclose(torques, expected, atol=1.0e-7)
  finally:
    env.close()


def test_tony5_aero_single_rotor_force_and_torque_signs(device: str) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    aero = _aero_physics(env)
    rotor_positions = robot.data.body_link_pos_w[:, aero.body_ids]
    root_position = robot.data.root_link_pos_w.unsqueeze(1)
    relative_positions = rotor_positions - root_position
    expected_roll_sign = (1.0, -1.0, -1.0, 1.0)
    expected_pitch_sign = (-1.0, -1.0, 1.0, 1.0)
    expected_yaw_sign = (1.0, -1.0, 1.0, -1.0)

    for rotor_id in range(4):
      speed = torch.zeros((1, 4), device=device)
      speed[0, rotor_id] = 1000.0
      robot.write_joint_velocity_to_sim(speed, joint_ids=aero.joint_ids)
      env.sim.forward()
      forces, torques = aero.compute_wrenches()
      moment = torch.cross(relative_positions, forces, dim=-1).sum(dim=1)
      assert forces[0, rotor_id, 2] > 0.0
      assert torch.sign(moment[0, 0]) == expected_roll_sign[rotor_id]
      assert torch.sign(moment[0, 1]) == expected_pitch_sign[rotor_id]
      assert torch.sign(torques[0, rotor_id, 2]) == expected_yaw_sign[rotor_id]
  finally:
    env.close()


def test_tony5_aero_world_wrench_buffer_overwrites_each_update(
  device: str,
) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    aero = _aero_physics(env)
    zero = torch.zeros((1,), device=device)
    roll = torch.full((1,), 20.0 * torch.pi / 180.0, device=device)
    pitch = torch.full((1,), -25.0 * torch.pi / 180.0, device=device)
    pose = torch.cat(
      (
        torch.tensor([[0.0, 0.0, 1.5]], device=device),
        quat_from_euler_xyz(roll, pitch, zero),
      ),
      dim=-1,
    )
    robot.write_root_link_pose_to_sim(pose)
    speed = torch.full((1, 4), 1500.0, device=device)
    robot.write_joint_velocity_to_sim(speed, joint_ids=aero.joint_ids)
    env.sim.forward()

    forces, torques = aero.compute_wrenches()
    global_body_ids = robot.data.indexing.body_ids[aero.body_ids]
    body_xmat = env.sim.data.xmat[:, global_body_ids].reshape(1, 4, 3, 3)
    expected_force = body_xmat[..., :, 2] * (KT_THRUST * speed.square()).unsqueeze(-1)
    expected_torque = (
      body_xmat[..., :, 2]
      * aero._axis_signs.view(1, 4, 1)
      * (-KQ_TORQUE * speed * speed.abs()).unsqueeze(-1)
    )
    assert torch.allclose(forces, expected_force, atol=1.0e-6)
    assert torch.allclose(torques, expected_torque, atol=1.0e-6)

    aero.apply()
    first_buffer = env.sim.data.xfrc_applied[:, global_body_ids].clone()
    assert torch.allclose(
      first_buffer,
      torch.cat((forces, torques), dim=-1),
      atol=1.0e-6,
    )

    lower_speed = torch.full((1, 4), 1000.0, device=device)
    robot.write_joint_velocity_to_sim(lower_speed, joint_ids=aero.joint_ids)
    env.sim.forward()
    lower_forces, lower_torques = aero.compute_wrenches()
    aero.apply()
    second_buffer = env.sim.data.xfrc_applied[:, global_body_ids]
    assert torch.allclose(
      second_buffer,
      torch.cat((lower_forces, lower_torques), dim=-1),
      atol=1.0e-6,
    )
    assert not torch.allclose(second_buffer, first_buffer, atol=1.0e-6)
  finally:
    env.close()


def test_tony5_aero_tilted_body_rotates_world_thrust(device: str) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    aero = _aero_physics(env)
    zero = torch.zeros((1,), device=device)
    speed = torch.full((1, 4), 1500.0, device=device)
    global_body_ids = robot.data.indexing.body_ids[aero.body_ids]

    for roll_deg, pitch_deg in ((30.0, 0.0), (0.0, 30.0), (20.0, -25.0)):
      roll = torch.full((1,), roll_deg * torch.pi / 180.0, device=device)
      pitch = torch.full((1,), pitch_deg * torch.pi / 180.0, device=device)
      pose = torch.cat(
        (
          torch.tensor([[0.0, 0.0, 1.5]], device=device),
          quat_from_euler_xyz(roll, pitch, zero),
        ),
        dim=-1,
      )
      robot.write_root_link_pose_to_sim(pose)
      robot.write_joint_velocity_to_sim(speed, joint_ids=aero.joint_ids)
      env.sim.forward()
      forces, torques = aero.compute_wrenches()
      body_xmat = env.sim.data.xmat[:, global_body_ids].reshape(1, 4, 3, 3)
      expected_force = body_xmat[..., :, 2] * (KT_THRUST * speed.square()).unsqueeze(-1)
      assert torch.allclose(forces, expected_force, atol=1.0e-6)
      assert torch.allclose(forces.sum(dim=1), expected_force.sum(dim=1), atol=1.0e-6)
      assert torch.allclose(
        torques.sum(dim=1), torch.zeros((1, 3), device=device), atol=1.0e-6
      )
  finally:
    env.close()


def test_tony5_aero_numerical_safety_keeps_wrenches_after_failure(
  device: str,
) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    action = _aero_action(env)
    monitor = action.safety_monitor
    robot.write_root_link_velocity_to_sim(
      torch.tensor([[151.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    )
    env.sim.forward()

    assert torch.all(numerical_safety_failure(env))
    assert torch.all(monitor.failed)
    assert torch.all(monitor.root_speed_count == 1.0)
    assert torch.all(monitor.max_root_speed >= 151.0)

    action.apply_actions()
    global_body_ids = robot.data.indexing.body_ids[action._aerodynamics.body_ids]
    assert torch.any(env.sim.data.xfrc_applied[:, global_body_ids] != 0.0)

    action.reset(torch.tensor([0], device=device))
    assert not torch.any(monitor.failed)
  finally:
    env.close()


def test_tony5_aero_numerical_safety_limits_are_relaxed_twofold() -> None:
  assert ROOT_SPEED_LIMIT == 150.0
  assert BODY_ANGULAR_SPEED_LIMIT == 200.0
  assert QACC_LIMIT == 4.0e5


def test_tony5_aero_numerical_safety_ignores_rotor_qvel_limit(device: str) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    action = _aero_action(env)
    robot.write_joint_velocity_to_sim(
      torch.full((1, 4), 8000.0, device=device),
      joint_ids=action._joint_ids,
    )
    env.sim.forward()
    assert not torch.any(numerical_safety_failure(env))
    assert torch.all(action.safety_monitor.rotor_speed_count == 0.0)
  finally:
    env.close()


def test_tony5_aero_numerical_safety_detects_nonfinite_state(device: str) -> None:
  env = _make_aero_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    monitor = _aero_action(env).safety_monitor
    robot.write_root_link_velocity_to_sim(
      torch.tensor([[float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    )
    env.sim.forward()
    assert torch.all(numerical_safety_failure(env))
    assert torch.all(monitor.nonfinite_count == 1.0)
  finally:
    env.close()


@pytest.mark.slow
def test_tony5_aero_batched_rollout_is_finite(device: str) -> None:
  env = _make_aero_env(device, num_envs=64)
  try:
    obs, _ = env.reset()
    for _ in range(300):
      action = 2.0 * torch.rand((64, 4), device=device) - 1.0
      obs, reward, terminated, truncated, _ = env.step(action)
      aero = _aero_physics(env)
      actor_obs = obs["actor"]
      assert isinstance(actor_obs, torch.Tensor)
      assert torch.isfinite(actor_obs).all()
      assert torch.isfinite(reward).all()
      assert torch.isfinite(aero.local_air_velocity_b).all()
      assert torch.isfinite(aero.h_force_w).all()
      assert not torch.isnan(terminated.float()).any()
      assert not torch.isnan(truncated.float()).any()
  finally:
    env.close()


def test_tony5_aero_play_defaults_to_one_environment() -> None:
  cfg = tony5_velocity_aero_play_env_cfg()
  assert cfg.scene.num_envs == 1
  assert cfg.episode_length_s > 1e8
