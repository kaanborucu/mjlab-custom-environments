"""Focused validation for the TONY5 Omni V3 global-wind task."""

import math
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manager_based.tony5 import (
  VELOCITY_AERO_OMNI_TASK_ID,
  VELOCITY_AERO_OMNI_V3_TASK_ID,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_actions import (
  Tony5AeroRotorSpeedActionCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_command import (
  Tony5OmniVelocityCommand,
  Tony5OmniVelocityCommandCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_env_cfg import (
  tony5_velocity_aero_omni_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_actions import (
  Tony5OmniV3RotorSpeedAction,
  Tony5OmniV3RotorSpeedActionCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_battery import (
  Tony5OmniV3Battery,
  Tony5OmniV3BatteryCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_env_cfg import (
  tony5_velocity_aero_omni_v3_env_cfg,
  tony5_velocity_aero_omni_v3_play_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_observations import (
  heading_sin_cos,
  world_velocity_command,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_physics import (
  Tony5OmniV3PhysicsCfg,
  Tony5OmniV3RotorAerodynamics,
  compute_blade_flapping_moment,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_prop import (
  Tony5OmniV3PropCfg,
  Tony5OmniV3PropModel,
  static_cq0,
  static_ct0,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_rewards import (
  TONY5_OMNI_V3_MOTOR_TORQUE_PENALTY_WEIGHT,
  TONY5_OMNI_V3_REWARD_WEIGHTS,
  rotor_torque_l2,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_wind import (
  Tony5OmniV3WindCfg,
)
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab.viewer.debug_visualizer import DebugVisualizer


@pytest.fixture(scope="module")
def device() -> str:
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_v3_env(
  device: str,
  wind_cfg: Tony5OmniV3WindCfg | None = None,
  physics_cfg: Tony5OmniV3PhysicsCfg | None = None,
  num_envs: int = 1,
) -> ManagerBasedRlEnv:
  cfg = tony5_velocity_aero_omni_v3_env_cfg()
  cfg.scene.num_envs = num_envs
  action_cfg = cast(
    Tony5OmniV3RotorSpeedActionCfg,
    cfg.actions["rotor_speed"],
  )
  if wind_cfg is not None:
    action_cfg.wind = wind_cfg
  if physics_cfg is not None:
    action_cfg.physics = physics_cfg
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def test_v3_registers_without_changing_omni_v0() -> None:
  import mjlab.tasks  # noqa: F401

  assert VELOCITY_AERO_OMNI_TASK_ID in list_tasks()
  assert VELOCITY_AERO_OMNI_V3_TASK_ID in list_tasks()
  v0_cfg = tony5_velocity_aero_omni_env_cfg()
  v3_cfg = tony5_velocity_aero_omni_v3_env_cfg()
  assert isinstance(v0_cfg.actions["rotor_speed"], Tony5AeroRotorSpeedActionCfg)
  assert not isinstance(v0_cfg.actions["rotor_speed"], Tony5OmniV3RotorSpeedActionCfg)
  assert isinstance(v3_cfg.actions["rotor_speed"], Tony5OmniV3RotorSpeedActionCfg)
  v0_command = cast(Tony5OmniVelocityCommandCfg, v0_cfg.commands["velocity"])
  v3_command = cast(Tony5OmniVelocityCommandCfg, v3_cfg.commands["velocity"])
  assert v0_command.linear_velocity_frame == "body"
  assert v0_command.yaw_velocity_frame == "body"
  assert v0_command.resampling_time_range == (3.0, 8.0)
  assert v0_command.enable_command_smoothing
  assert v3_command.linear_velocity_frame == "world"
  assert v3_command.yaw_velocity_frame == "body"
  assert v3_command.resampling_time_range == (1.0, 10.0)
  assert not v3_command.enable_command_smoothing
  assert v3_command.command_transition_time_s == 0.0
  assert load_rl_cfg(VELOCITY_AERO_OMNI_TASK_ID).experiment_name == (
    "tony5_velocity_aero_omni_v0"
  )
  assert load_rl_cfg(VELOCITY_AERO_OMNI_V3_TASK_ID).experiment_name == (
    "tony5_velocity_aero_omni_v3"
  )
  assert load_rl_cfg(VELOCITY_AERO_OMNI_V3_TASK_ID).load_run == (
    r"\d{4}-\d{2}-\d{2}_.*"
  )


def test_v3_play_enables_background_wind_and_gusts() -> None:
  cfg = tony5_velocity_aero_omni_v3_play_env_cfg()
  action_cfg = cast(
    Tony5OmniV3RotorSpeedActionCfg,
    cfg.actions["rotor_speed"],
  )
  assert action_cfg.wind.enable_wind
  assert action_cfg.wind.wind_x == 2.0
  assert action_cfg.wind.wind_y == 0.0
  assert action_cfg.wind.wind_z == 0.0
  assert action_cfg.wind.randomize_background_wind
  assert action_cfg.wind.background_horizontal_speed_range == (0.0, 10.0)
  assert action_cfg.wind.background_vertical_speed_range == (-1.0, 1.0)
  assert action_cfg.wind.enable_gusts
  assert action_cfg.wind.gust_sigma == 1.0
  assert action_cfg.wind.gust_tau == 1.0
  assert action_cfg.wind.max_gust_speed == 5.0
  assert action_cfg.wind.reset_enable_probability == 0.5
  assert action_cfg.physics.prop.enable_advanced_prop_model
  assert action_cfg.physics.prop.ct_curve_j == (0.0,)
  assert action_cfg.physics.prop.cq_curve_j == (0.0,)
  assert action_cfg.physics.enable_blade_flapping
  assert action_cfg.battery.enable_battery_sag


def test_v3_training_enables_optional_physics_features() -> None:
  cfg = tony5_velocity_aero_omni_v3_env_cfg()
  action_cfg = cast(Tony5OmniV3RotorSpeedActionCfg, cfg.actions["rotor_speed"])
  assert action_cfg.physics.prop.enable_advanced_prop_model
  assert action_cfg.physics.prop.ct_curve_j == (0.0,)
  assert action_cfg.physics.prop.cq_curve_j == (0.0,)
  assert action_cfg.physics.enable_blade_flapping
  assert action_cfg.battery.enable_battery_sag
  assert action_cfg.wind.enable_wind
  assert action_cfg.wind.enable_gusts
  assert action_cfg.wind.wind_x == 2.0
  assert action_cfg.wind.randomize_background_wind
  assert action_cfg.wind.background_horizontal_speed_range == (0.0, 10.0)
  assert action_cfg.wind.background_vertical_speed_range == (-1.0, 1.0)
  assert action_cfg.wind.gust_sigma == 1.0
  assert action_cfg.wind.gust_tau == 1.0
  assert action_cfg.wind.max_gust_speed == 5.0
  assert action_cfg.wind.reset_enable_probability == 0.5
  assert set(cfg.rewards) == set(TONY5_OMNI_V3_REWARD_WEIGHTS)
  for name, weight in TONY5_OMNI_V3_REWARD_WEIGHTS.items():
    assert cfg.rewards[name].weight == pytest.approx(weight)
  assert not cfg.rewards["downward_velocity"].log
  assert cfg.rewards["motor_torque"].func is rotor_torque_l2
  assert cfg.rewards["motor_torque"].weight == TONY5_OMNI_V3_MOTOR_TORQUE_PENALTY_WEIGHT
  assert set(cfg.metrics) == {
    "numerical_safety_failure_rate",
    "max_root_speed",
    "max_body_angular_speed",
    "max_rotor_speed",
    "max_abs_qacc",
    "wind_speed",
    "voltage_sag",
  }
  play_cfg = tony5_velocity_aero_omni_v3_play_env_cfg()
  assert set(play_cfg.rewards) == {
    "linear_velocity_tracking",
    "angular_velocity_tracking",
    "action_rate",
    "crash",
    "motor_torque",
  }


def test_v3_world_frame_observation_is_explicit_and_observable(device: str) -> None:
  cfg = tony5_velocity_aero_omni_v3_env_cfg()
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    assert "linear_velocity_w" in terms
    assert "linear_velocity_b" not in terms
    assert "heading_sin_cos" in terms

  env = _make_v3_env(device, num_envs=2)
  try:
    observations, _ = env.reset()
    if not isinstance(observations, dict):
      raise AssertionError("Expected grouped actor/critic observations.")
    actor_observation = cast(torch.Tensor, observations["actor"])
    assert actor_observation.shape == (2, 120)
    command = world_velocity_command(env, "velocity")
    assert command.shape == (2, 4)
    velocity_term = cast(
      Tony5OmniVelocityCommand,
      env.command_manager.get_term("velocity"),
    )
    velocity_term.vel_command_b[:, 0] = 99.0
    velocity_term.vel_command_w[:, 0] = 3.0
    velocity_term.vel_command_w[:, 1] = -4.0
    velocity_term.vel_command_b[:, 2] = 0.25
    velocity_term.vel_command_w[:, 2] = 1.5
    world_command = world_velocity_command(env, "velocity")
    assert torch.allclose(
      world_command,
      torch.tensor([[3.0, -4.0, 0.25, 1.5]] * 2, device=device),
    )
    heading = heading_sin_cos(env)
    assert torch.allclose(
      torch.linalg.vector_norm(heading, dim=-1),
      torch.ones(2, device=device),
    )
    assert torch.isfinite(actor_observation).all()
  finally:
    env.close()


def test_v3_keyboard_override_stays_in_world_frame_after_yaw(device: str) -> None:
  env = _make_v3_env(device)
  try:
    env.reset()
    robot = env.scene["robot"]
    half_yaw = math.pi / 4.0
    pose = torch.tensor(
      [
        [
          0.0,
          0.0,
          1.0,
          math.cos(half_yaw),
          0.0,
          0.0,
          math.sin(half_yaw),
        ]
      ],
      device=device,
    )
    robot.write_root_link_pose_to_sim(pose)
    env.scene.write_data_to_sim()
    env.sim.forward()

    command = cast(
      Tony5OmniVelocityCommand,
      env.command_manager.get_term("velocity"),
    )
    command.enable_keyboard_control()
    speed = command.keyboard_max_speed
    command.set_keyboard_mode("forward")
    world_command = world_velocity_command(env, "velocity")
    assert torch.allclose(
      world_command,
      torch.tensor([[speed, 0.0, 0.0, 0.0]], device=device),
      atol=1.0e-6,
    )

    command.set_keyboard_mode("left")
    world_command = world_velocity_command(env, "velocity")
    assert torch.allclose(
      world_command,
      torch.tensor([[0.0, speed, 0.0, 0.0]], device=device),
      atol=1.0e-6,
    )
  finally:
    env.close()


def test_v3_motor_torque_reward_uses_native_actuator_force(device: str) -> None:
  env = _make_v3_env(device, num_envs=2)
  try:
    env.reset()
    torque_penalty_input = rotor_torque_l2(env)
    assert torque_penalty_input.shape == (2,)
    assert torch.isfinite(torque_penalty_input).all()
  finally:
    env.close()


def test_static_ct_cq_reproduce_existing_kt_kq_model() -> None:
  model = Tony5OmniV3PropModel(Tony5OmniV3PropCfg(), "cpu")
  omega = torch.tensor([[836.0, 1800.0, 2600.0, 0.0]])
  axial_velocity = torch.zeros_like(omega)
  _, ct, cq = model.compute_coefficients(omega, axial_velocity)
  n = omega.abs() / (2.0 * torch.pi)
  expected_thrust = model.rho * n.square() * model.diameter**4 * ct
  expected_torque = model.rho * n.square() * model.diameter**5 * cq
  assert torch.allclose(
    expected_thrust,
    1.00e-6 * omega.abs().square(),
    atol=1.0e-9,
    rtol=1.0e-6,
  )
  assert torch.allclose(
    expected_torque,
    1.25e-8 * omega.abs().square(),
    atol=1.0e-11,
    rtol=1.0e-6,
  )
  assert model.ct0 == pytest.approx(static_ct0())
  assert model.cq0 == pytest.approx(static_cq0())
  assert torch.isfinite(
    model.compute_coefficients(torch.zeros_like(omega), axial_velocity)[0]
  ).all()
  with pytest.raises(ValueError, match="explicit measured"):
    Tony5OmniV3PropModel(
      Tony5OmniV3PropCfg(enable_advanced_prop_model=True),
      "cpu",
    )


def test_static_curve_mode_does_not_double_count_rotor_loads(device: str) -> None:
  physics_cfg = Tony5OmniV3PhysicsCfg(
    prop=Tony5OmniV3PropCfg(
      enable_advanced_prop_model=True,
      ct_curve_j=(0.0,),
      ct_curve_values=(static_ct0(),),
      cq_curve_j=(0.0,),
      cq_curve_values=(static_cq0(),),
    )
  )
  no_wind = Tony5OmniV3WindCfg()
  fallback_env = _make_v3_env(
    device,
    wind_cfg=no_wind,
    physics_cfg=Tony5OmniV3PhysicsCfg(),
  )
  advanced_env = _make_v3_env(
    device,
    wind_cfg=Tony5OmniV3WindCfg(),
    physics_cfg=physics_cfg,
  )
  try:
    fallback_env.reset()
    advanced_env.reset()
    fallback_action = cast(
      Tony5OmniV3RotorSpeedAction,
      fallback_env.action_manager.get_term("rotor_speed"),
    )
    advanced_action = cast(
      Tony5OmniV3RotorSpeedAction,
      advanced_env.action_manager.get_term("rotor_speed"),
    )
    fallback_robot = fallback_env.scene["robot"]
    advanced_robot = advanced_env.scene["robot"]
    velocity = torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=device)
    omega = torch.full((1, 4), 1800.0, device=device)
    for robot, action, env in (
      (fallback_robot, fallback_action, fallback_env),
      (advanced_robot, advanced_action, advanced_env),
    ):
      robot.write_root_link_velocity_b_to_sim(velocity)
      robot.write_joint_velocity_to_sim(omega, joint_ids=action._joint_ids)
      env.scene.write_data_to_sim()
      env.sim.forward()
    fallback_wrench = fallback_action._aerodynamics.compute_wrenches()
    advanced_wrench = advanced_action._aerodynamics.compute_wrenches()
    assert torch.allclose(fallback_wrench[0], advanced_wrench[0], atol=1.0e-6)
    assert torch.allclose(fallback_wrench[1], advanced_wrench[1], atol=1.0e-8)
  finally:
    fallback_env.close()
    advanced_env.close()


def test_prop_curves_interpolate_vectorized_and_keep_signed_j() -> None:
  cfg = Tony5OmniV3PropCfg(
    enable_advanced_prop_model=True,
    ct_curve_j=(-1.0, 0.0, 1.0),
    ct_curve_values=(0.4, 0.3, 0.2),
    cq_curve_j=(-1.0, 0.0, 1.0),
    cq_curve_values=(0.08, 0.06, 0.04),
  )
  model = Tony5OmniV3PropModel(cfg, "cpu")
  query = torch.tensor([[-0.5, 0.0, 0.5]])
  assert torch.allclose(model.lookup_ct(query), torch.tensor([[0.35, 0.3, 0.25]]))
  assert torch.allclose(model.lookup_cq(query), torch.tensor([[0.07, 0.06, 0.05]]))
  omega = torch.full((1, 3), 1000.0)
  axial_velocity = torch.tensor([[-1.0, 0.0, 1.0]])
  advance_ratio, ct, cq = model.compute_coefficients(omega, axial_velocity)
  assert advance_ratio[0, 0] < 0.0 < advance_ratio[0, 2]
  assert torch.isfinite(torch.stack((advance_ratio, ct, cq))).all()


def test_blade_flapping_signs_and_zero_limits() -> None:
  axis = torch.tensor([[0.0, 0.0, 1.0]])
  omega = torch.tensor([1000.0])
  plus_x = compute_blade_flapping_moment(
    omega,
    torch.tensor([[1.0, 0.0, 0.0]]),
    axis,
    1.0e-7,
  )
  minus_x = compute_blade_flapping_moment(
    omega,
    torch.tensor([[-1.0, 0.0, 0.0]]),
    axis,
    1.0e-7,
  )
  plus_y = compute_blade_flapping_moment(
    omega,
    torch.tensor([[0.0, 1.0, 0.0]]),
    axis,
    1.0e-7,
  )
  assert plus_x[0, 1] > 0.0
  assert minus_x[0, 1] < 0.0
  assert plus_y[0, 0] < 0.0
  assert torch.equal(
    compute_blade_flapping_moment(
      torch.zeros_like(omega),
      torch.ones((1, 3)),
      axis,
      1.0e-7,
    ),
    torch.zeros((1, 3)),
  )
  assert torch.equal(
    compute_blade_flapping_moment(
      omega,
      torch.zeros((1, 3)),
      axis,
      1.0e-7,
    ),
    torch.zeros((1, 3)),
  )


def test_battery_sag_is_optional_and_monotonic() -> None:
  disabled = Tony5OmniV3Battery(Tony5OmniV3BatteryCfg(), 1, 4, "cpu")
  voltage = torch.full((1, 4), 22.2)
  omega = torch.zeros((1, 4))
  disabled.record_motor_current(voltage, omega)
  disabled.update_voltage_limit()
  assert disabled.loaded_voltage.item() == pytest.approx(22.2)

  enabled = Tony5OmniV3Battery(
    Tony5OmniV3BatteryCfg(enable_battery_sag=True, battery_resistance=0.02),
    1,
    4,
    "cpu",
  )
  enabled.record_motor_current(torch.full((1, 4), 10.0), omega)
  enabled.update_voltage_limit()
  low_load_voltage = enabled.loaded_voltage.item()
  enabled.record_motor_current(voltage, omega)
  enabled.update_voltage_limit()
  assert enabled.loaded_voltage.item() < low_load_voltage < 22.2
  assert torch.all(enabled.motor_current >= 0.0)


def test_vehicle_motion_and_wind_have_opposite_h_force_sign(device: str) -> None:
  motion_env = _make_v3_env(device, wind_cfg=Tony5OmniV3WindCfg())
  wind_env = _make_v3_env(
    device,
    Tony5OmniV3WindCfg(
      enable_wind=True,
      wind_x=10.0,
      reset_enable_probability=1.0,
    ),
  )
  try:
    motion_env.reset()
    wind_env.reset()
    motion_action = cast(
      Tony5OmniV3RotorSpeedAction,
      motion_env.action_manager.get_term("rotor_speed"),
    )
    wind_action = cast(
      Tony5OmniV3RotorSpeedAction,
      wind_env.action_manager.get_term("rotor_speed"),
    )
    motion_robot = motion_env.scene["robot"]
    wind_robot = wind_env.scene["robot"]
    omega = torch.full((1, 4), 1800.0, device=device)
    zero_velocity = torch.zeros((1, 6), device=device)
    forward_velocity = zero_velocity.clone()
    forward_velocity[:, 0] = 10.0
    motion_robot.write_root_link_velocity_b_to_sim(forward_velocity)
    motion_robot.write_joint_velocity_to_sim(omega, joint_ids=motion_action._joint_ids)
    wind_robot.write_root_link_velocity_b_to_sim(zero_velocity)
    wind_robot.write_joint_velocity_to_sim(omega, joint_ids=wind_action._joint_ids)
    motion_env.scene.write_data_to_sim()
    wind_env.scene.write_data_to_sim()
    motion_env.sim.forward()
    wind_env.sim.forward()
    motion_action._aerodynamics.compute_wrenches()
    wind_action._aerodynamics.compute_wrenches()
    motion_h = cast(Tony5OmniV3RotorAerodynamics, motion_action._aerodynamics).h_force_w
    wind_h = cast(Tony5OmniV3RotorAerodynamics, wind_action._aerodynamics).h_force_w
    assert torch.allclose(motion_h, -wind_h, atol=1.0e-5, rtol=1.0e-5)
  finally:
    motion_env.close()
    wind_env.close()


def test_disabled_v3_wind_does_not_consume_rng(device: str) -> None:
  env = _make_v3_env(device, wind_cfg=Tony5OmniV3WindCfg())
  try:
    env.reset()
    action = cast(
      Tony5OmniV3RotorSpeedAction, env.action_manager.get_term("rotor_speed")
    )
    torch.manual_seed(1234)
    state_before = torch.random.get_rng_state()
    action.wind.step()
    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)
    assert torch.equal(action.wind.wind_world, torch.zeros(3, device=device))
  finally:
    env.close()


def test_v3_wind_reset_probability_controls_the_shared_episode(device: str) -> None:
  disabled_env = _make_v3_env(
    device,
    wind_cfg=Tony5OmniV3WindCfg(
      enable_wind=True,
      wind_x=1.0,
      reset_enable_probability=0.0,
    ),
  )
  enabled_env = _make_v3_env(
    device,
    wind_cfg=Tony5OmniV3WindCfg(
      enable_wind=True,
      wind_x=1.0,
      reset_enable_probability=1.0,
    ),
  )
  try:
    disabled_action = cast(
      Tony5OmniV3RotorSpeedAction,
      disabled_env.action_manager.get_term("rotor_speed"),
    )
    enabled_action = cast(
      Tony5OmniV3RotorSpeedAction,
      enabled_env.action_manager.get_term("rotor_speed"),
    )
    disabled_env.reset()
    enabled_env.reset()
    assert not disabled_action.wind.episode_enabled
    assert enabled_action.wind.episode_enabled
    assert torch.equal(
      disabled_action.wind.wind_world,
      torch.zeros(3, device=device),
    )
    assert torch.equal(
      enabled_action.wind.wind_world,
      torch.tensor([1.0, 0.0, 0.0], device=device),
    )
  finally:
    disabled_env.close()
    enabled_env.close()


def test_randomized_background_wind_is_bounded_on_full_reset(device: str) -> None:
  env = _make_v3_env(
    device,
    Tony5OmniV3WindCfg(
      enable_wind=True,
      wind_x=2.0,
      randomize_background_wind=True,
      background_horizontal_speed_range=(0.0, 10.0),
      background_vertical_speed_range=(-1.0, 1.0),
      reset_enable_probability=1.0,
    ),
  )
  try:
    env.reset()
    action = cast(
      Tony5OmniV3RotorSpeedAction, env.action_manager.get_term("rotor_speed")
    )
    first_wind = action.wind.wind_world.clone()
    assert torch.linalg.vector_norm(first_wind[:2]) <= 10.0
    assert torch.abs(first_wind[2]) <= 1.0

    action.wind.reset(torch.arange(env.num_envs, device=device))
    second_wind = action.wind.wind_world
    assert torch.linalg.vector_norm(second_wind[:2]) <= 10.0
    assert torch.abs(second_wind[2]) <= 1.0
  finally:
    env.close()


def test_global_wind_reaches_native_and_custom_rotor_aero(device: str) -> None:
  env = _make_v3_env(
    device,
    Tony5OmniV3WindCfg(
      enable_wind=True,
      wind_x=1.0,
      reset_enable_probability=1.0,
    ),
    num_envs=2,
  )
  try:
    env.reset()
    action = cast(
      Tony5OmniV3RotorSpeedAction, env.action_manager.get_term("rotor_speed")
    )
    action.apply_actions()
    env.sim.forward()
    aero = cast(Tony5OmniV3RotorAerodynamics, action._aerodynamics)
    assert torch.allclose(
      action.wind.wind_world,
      torch.tensor([1.0, 0.0, 0.0], device=device),
    )
    assert torch.allclose(
      env.sim.model.opt.wind[0],
      action.wind.wind_world,
    )
    assert torch.allclose(
      aero.local_air_velocity_b[..., 0],
      torch.full((2, 4), -1.0, device=device),
      atol=1.0e-5,
    )
    assert torch.all(aero.h_force_w[..., 0] > 0.0)
    rotor_dofs = env.scene["robot"].indexing.joint_v_adr[action._joint_ids]
    assert torch.max(torch.abs(env.sim.data.qfrc_fluid[:, rotor_dofs])) < 1.0e-5
    root_dofs = env.scene["robot"].indexing.free_joint_v_adr
    assert torch.any(torch.abs(env.sim.data.qfrc_fluid[:, root_dofs[:3]]) > 0.0)

    class Visualizer:
      def __init__(self) -> None:
        self.arrows = []

      def get_env_indices(self, num_envs: int) -> list[int]:
        del num_envs
        return [0]

      def add_arrow(self, **kwargs: object) -> None:
        self.arrows.append(kwargs)

    visualizer = Visualizer()
    action.wind.debug_vis(cast(DebugVisualizer, visualizer))
    assert len(visualizer.arrows) == 1
  finally:
    env.close()


def test_gust_is_correlated_and_bounded(device: str) -> None:
  env = _make_v3_env(
    device,
    Tony5OmniV3WindCfg(
      enable_gusts=True,
      gust_sigma=0.5,
      gust_tau=1.0,
      max_gust_speed=0.25,
      reset_enable_probability=1.0,
    ),
  )
  try:
    env.reset()
    action = cast(
      Tony5OmniV3RotorSpeedAction, env.action_manager.get_term("rotor_speed")
    )
    previous = action.wind.gust_world.clone()
    changes = []
    for _ in range(32):
      action.wind.step()
      current = action.wind.gust_world.clone()
      changes.append(torch.linalg.vector_norm(current - previous))
      assert torch.linalg.vector_norm(current) <= 0.25 + 1.0e-6
      previous = current
    assert torch.any(torch.stack(changes) > 0.0)
  finally:
    env.close()
