"""Focused tests for the TONY5 V1 radial velocity curriculum."""

from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.manager_based.tony5.tony5_aero_env_cfg import (
  TONY5_AERO_REWARD_WEIGHTS,
  tony5_velocity_aero_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_ppo_cfg import (
  TONY5_AERO_ENTROPY_COEF,
)
from mjlab.tasks.manager_based.tony5.tony5_aero_rewards import (
  body_angular_acceleration_l2,
  high_speed_roll_uprightness_reward,
  roll_pitch_rate_weight,
  speed_scaled_uprightness_reward,
  track_angular_velocity_speed_scaled,
  upright_reward_scale,
)
from mjlab.tasks.manager_based.tony5.tony5_velocity_curriculum import (
  CURRICULUM_SETTLING_TIME_S,
  TONY5_VELOCITY_CURRICULUM_STEPS_PER_ITERATION,
  TONY5_VELOCITY_CURRICULUM_SWITCH_ITERATIONS,
  TONY5_VELOCITY_KEYBOARD_MAX_SPEED,
  TONY5_VELOCITY_STAGE_SPEEDS,
  Tony5RadialVelocityCommand,
  Tony5RadialVelocityCommandCfg,
  command_direction_limit,
  sample_horizontal_direction,
  track_linear_velocity_adaptive,
  velocity_reward_sigma,
  vertical_velocity_command_limit,
  yaw_rate_limit,
)
from mjlab.tasks.manager_based.tony5.tony5_velocity_env_cfg import (
  tony5_velocity_env_cfg,
)
from mjlab.tasks.registry import load_rl_cfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg


@pytest.fixture(scope="module")
def device() -> str:
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_env(device: str, num_envs: int = 1024) -> ManagerBasedRlEnv:
  cfg = tony5_velocity_aero_env_cfg()
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def _command(env: ManagerBasedRlEnv) -> Tony5RadialVelocityCommand:
  return cast(
    Tony5RadialVelocityCommand,
    env.command_manager.get_term("velocity"),
  )


def test_v1_radial_config_preserves_v0_and_uses_four_speed_stages() -> None:
  import mjlab.tasks  # noqa: F401

  v0_cfg = tony5_velocity_env_cfg()
  v1_cfg = tony5_velocity_aero_env_cfg()
  v0_command = v0_cfg.commands["velocity"]
  v1_command = v1_cfg.commands["velocity"]

  assert isinstance(v0_command, UniformVelocityCommandCfg)
  assert not isinstance(v0_command, Tony5RadialVelocityCommandCfg)
  assert isinstance(v1_command, Tony5RadialVelocityCommandCfg)
  assert "attitude" in v0_cfg.terminations
  assert "attitude" not in v1_cfg.terminations
  assert v1_command.stage_speeds == (2.0, 6.0, 10.0, 27.78)
  assert v1_command.stage_speeds == TONY5_VELOCITY_STAGE_SPEEDS
  assert v1_command.curriculum_switch_iterations == (
    TONY5_VELOCITY_CURRICULUM_SWITCH_ITERATIONS
  )
  assert (
    v1_command.curriculum_steps_per_iteration
    == TONY5_VELOCITY_CURRICULUM_STEPS_PER_ITERATION
  )
  assert v1_command.rel_standing_envs == 0.2
  assert v1_command.previous_speed_fraction == 0.50
  assert v1_command.current_speed_fraction == 0.30
  assert v1_command.high_speed_fraction == 0.20
  assert v1_command.sampling_high_speed_min_fraction == 0.80
  assert v1_command.high_speed_min_fraction == 0.70
  assert v1_command.ranges.lin_vel_z == (-3.0, 3.0)
  assert v1_command.ranges.ang_vel_z == (-1.5, 1.5)
  assert (
    v0_cfg.rewards["linear_velocity_tracking"].func
    is not track_linear_velocity_adaptive
  )
  assert (
    v1_cfg.rewards["linear_velocity_tracking"].func is track_linear_velocity_adaptive
  )
  assert v0_cfg.rewards["uprightness"].func is not speed_scaled_uprightness_reward
  assert v0_cfg.rewards["angular_velocity_tracking"].func is not (
    track_angular_velocity_speed_scaled
  )
  assert v1_cfg.rewards["uprightness"].func is high_speed_roll_uprightness_reward
  assert (
    v1_cfg.rewards["body_angular_acceleration"].func is body_angular_acceleration_l2
  )
  assert v1_cfg.rewards["body_angular_acceleration"].weight == pytest.approx(
    TONY5_AERO_REWARD_WEIGHTS["body_angular_acceleration"]
  )
  assert (
    v1_cfg.rewards["angular_velocity_tracking"].func
    is track_angular_velocity_speed_scaled
  )
  assert v1_cfg.rewards["linear_velocity_tracking"].weight == pytest.approx(
    TONY5_AERO_REWARD_WEIGHTS["linear_velocity_tracking"]
  )
  assert v1_cfg.rewards["angular_velocity_tracking"].weight == pytest.approx(
    TONY5_AERO_REWARD_WEIGHTS["angular_velocity_tracking"]
  )
  assert v1_cfg.rewards["action_rate"].weight == pytest.approx(
    TONY5_AERO_REWARD_WEIGHTS["action_rate"]
  )
  assert v1_cfg.rewards["crash"].weight == pytest.approx(
    TONY5_AERO_REWARD_WEIGHTS["crash"]
  )
  v0_rl_cfg = cast(
    RslRlOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tony5-Velocity-v0"),
  )
  v1_rl_cfg = cast(
    RslRlOnPolicyRunnerCfg,
    load_rl_cfg("Mjlab-Tony5-Velocity-Aero-v1"),
  )
  assert v0_rl_cfg.algorithm.entropy_coef == 0.01
  assert v1_rl_cfg.algorithm.entropy_coef == TONY5_AERO_ENTROPY_COEF
  assert v0_rl_cfg.actor.distribution_cfg is not None
  assert v1_rl_cfg.actor.distribution_cfg is not None
  assert v0_rl_cfg.actor.distribution_cfg["init_std"] == 0.6
  assert "learn_std" not in v0_rl_cfg.actor.distribution_cfg
  assert "std_range" not in v0_rl_cfg.actor.distribution_cfg
  assert v1_rl_cfg.actor.distribution_cfg["init_std"] == 0.6
  assert v1_rl_cfg.actor.distribution_cfg["learn_std"] is True
  assert "std_range" not in v1_rl_cfg.actor.distribution_cfg
  assert set(v1_cfg.metrics) == {
    "curriculum_stage",
    "current_vmax",
    "horizontal_velocity_rmse",
    "high_speed_steady_rmse",
    "mean_commanded_horizontal_speed",
    "mean_achieved_horizontal_speed",
    "numerical_safety_failure_rate",
    "max_root_speed",
    "max_body_angular_speed",
    "max_rotor_speed",
    "max_abs_qacc",
  }


def test_v1_radial_sampling_is_bounded_directional_and_has_hover(device: str) -> None:
  torch.manual_seed(1234)
  env = _make_env(device, num_envs=4096)
  try:
    env.reset()
    command = _command(env)
    env_ids = torch.arange(env.num_envs, device=device)
    command._resample_command(env_ids)
    sampled = command.command
    horizontal_speed = torch.linalg.vector_norm(sampled[:, :2], dim=-1)

    assert command.curriculum_stage == 0
    assert command.current_vmax == 2.0
    assert torch.all(horizontal_speed <= command.current_vmax + 1.0e-6)

    hover = torch.all(sampled == 0.0, dim=-1)
    hover_fraction = hover.float().mean().item()
    assert abs(hover_fraction - 0.2) < 0.05

    branches = command.sample_branch
    moving = ~hover
    moving_branch_fractions = [
      (branches[moving] == branch).float().mean().item() for branch in (1, 2, 3)
    ]
    for fraction, expected in zip(
      moving_branch_fractions, (0.50, 0.30, 0.20), strict=True
    ):
      assert abs(fraction - expected) < 0.05
    assert torch.all(
      horizontal_speed[branches == 3] >= 0.8 * command.current_vmax - 1.0e-6
    )

    angles = torch.atan2(sampled[moving, 1], sampled[moving, 0])
    assert angles.min().item() < -2.5
    assert angles.max().item() > 2.5
    assert torch.isfinite(sampled).all()
  finally:
    env.close()


def test_v1_previous_stage_sampling_stays_below_current_max(device: str) -> None:
  torch.manual_seed(5678)
  env = _make_env(device, num_envs=4096)
  try:
    env.reset()
    command = _command(env)
    command._stage = 3
    command._current_vmax = 27.78
    env_ids = torch.arange(env.num_envs, device=device)
    command._resample_command(env_ids)
    speed = torch.linalg.vector_norm(command.command[:, :2], dim=-1)

    previous = command.sample_branch == 1
    current = command.sample_branch == 2
    high = command.sample_branch == 3
    assert torch.all(speed[previous] <= 10.0 + 1.0e-6)
    assert torch.all(speed[current] <= 27.78 + 1.0e-6)
    assert torch.all(speed[high] >= 0.8 * 27.78 - 1.0e-6)
    assert torch.all(speed <= 27.78 + 1.0e-6)
  finally:
    env.close()


def test_v1_high_speed_play_sampling_uses_requested_band(device: str) -> None:
  torch.manual_seed(2468)
  cfg = tony5_velocity_aero_env_cfg(play=True)
  command_cfg = cast(Tony5RadialVelocityCommandCfg, cfg.commands["velocity"])
  command_cfg.play_high_speed_only = True
  command_cfg.play_high_speed_min = 15.0
  command_cfg.play_high_speed_max = 27.78
  cfg.scene.num_envs = 4096
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    command = _command(env)
    env_ids = torch.arange(env.num_envs, device=device)
    command._resample_command(env_ids)
    sampled = command.command
    horizontal_speed = torch.linalg.vector_norm(sampled[:, :2], dim=-1)

    assert command.curriculum_stage == 3
    assert command.current_vmax == 27.78
    assert torch.all(horizontal_speed >= 15.0 - 1.0e-6)
    assert torch.all(horizontal_speed <= 27.78 + 1.0e-6)
    assert torch.all(command.sample_branch == 3)
    assert not torch.all(command.is_standing_env)

    angles = torch.atan2(sampled[:, 1], sampled[:, 0])
    assert torch.all(angles.abs() <= torch.pi / 2.0 + 1.0e-6)
    assert torch.isfinite(sampled).all()
  finally:
    env.close()


def test_v1_speed_dependent_yaw_limit_and_reward_sigma() -> None:
  speeds = torch.tensor([0.0, 1.0, 10.0, 27.78])
  assert torch.allclose(
    yaw_rate_limit(speeds),
    torch.tensor([1.5, 1.5, 0.5, 0.10]),
  )
  assert torch.allclose(
    vertical_velocity_command_limit(torch.tensor([0.0, 15.0, 27.78])),
    torch.tensor([3.0, 3.0, 3.0]),
  )
  assert torch.allclose(
    velocity_reward_sigma(speeds),
    torch.full_like(speeds, 2.0),
  )


def test_v1_speed_dependent_direction_limits_preserve_magnitude() -> None:
  speeds = torch.tensor([10.0, 12.0, 17.0, 22.0, 27.0])
  expected_limits = torch.deg2rad(torch.tensor([180.0, 90.0, 45.0, 25.0, 10.0]))
  assert torch.allclose(command_direction_limit(speeds), expected_limits)

  uniform_samples = torch.linspace(0.0, 1.0, 4096)
  low_theta = sample_horizontal_direction(
    torch.full_like(uniform_samples, 10.0), uniform_samples
  )
  assert low_theta.min().item() == pytest.approx(-torch.pi)
  assert low_theta.max().item() == pytest.approx(torch.pi)

  for speed, limit in zip(speeds[1:], expected_limits[1:], strict=True):
    sampled_speed = torch.full((4096,), speed.item())
    theta = sample_horizontal_direction(sampled_speed, uniform_samples)
    assert torch.all(theta.abs() <= limit + 1.0e-6)
    vx = sampled_speed * torch.cos(theta)
    vy = sampled_speed * torch.sin(theta)
    assert torch.allclose(torch.sqrt(vx.square() + vy.square()), sampled_speed)

  for speed in speeds[2:]:
    sampled_speed = torch.full((4096,), speed.item())
    theta = sample_horizontal_direction(sampled_speed, uniform_samples)
    assert torch.all(sampled_speed * torch.cos(theta) > 0.0)


def test_v1_keyboard_velocity_override_and_speed_buttons(device: str) -> None:
  env = _make_env(device, num_envs=1)
  try:
    env.reset()
    command = _command(env)
    command.enable_keyboard_control()
    assert command.keyboard_max_speed == pytest.approx(2.0)
    assert torch.allclose(command.command, torch.zeros((1, 4), device=device))

    command.set_keyboard_mode("forward")
    assert torch.isclose(command.command[0, 0], torch.tensor(2.0, device=device))
    assert torch.allclose(command.command[0, 1:], torch.zeros(3, device=device))

    command.adjust_keyboard_max_speed(3.0)
    assert command.keyboard_max_speed == pytest.approx(5.0)
    assert torch.isclose(command.command[0, 0], torch.tensor(5.0, device=device))

    command.set_keyboard_mode("yaw_left")
    assert torch.isclose(command.command[0, 2], torch.tensor(1.0, device=device))
    assert torch.allclose(command.command[0, [0, 1, 3]], torch.zeros(3, device=device))

    command.adjust_keyboard_max_speed(100.0)
    assert command.keyboard_max_speed == pytest.approx(
      TONY5_VELOCITY_KEYBOARD_MAX_SPEED
    )
    command.set_keyboard_mode("hover")
    assert torch.allclose(command.command, torch.zeros((1, 4), device=device))
  finally:
    env.close()


def test_v1_gamepad_velocity_override_clamps_command(device: str) -> None:
  env = _make_env(device, num_envs=1)
  try:
    env.reset()
    command = _command(env)
    command.enable_gamepad_control()
    command.set_gamepad_command((2.0, 2.0, 2.0, 4.0))

    assert torch.linalg.vector_norm(command.command[0, :2]).item() == pytest.approx(2.0)
    assert command.command[0, 2].item() == pytest.approx(1.5)
    assert command.command[0, 3].item() == pytest.approx(3.0)
  finally:
    env.close()


def test_v1_speed_dependent_reward_scales() -> None:
  speeds = torch.tensor([0.0, 2.5, 5.0, 8.0, 10.0, 27.78])
  assert torch.allclose(
    upright_reward_scale(speeds),
    torch.tensor([1.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
  )
  assert torch.allclose(
    roll_pitch_rate_weight(speeds),
    torch.tensor([1.0, 0.75, 0.5, 0.2, 0.2, 0.2]),
  )


def test_v1_angular_reward_keeps_yaw_full_and_scales_roll_pitch(
  device: str,
) -> None:
  env = _make_env(device, num_envs=1)
  try:
    env.reset()
    command = _command(env)
    command.vel_command_b.zero_()
    command.vel_command_b[0, 0] = 8.0
    robot = env.scene["robot"]

    robot.write_root_link_velocity_b_to_sim(
      torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device)
    )
    env.sim.forward()
    yaw_reward = track_angular_velocity_speed_scaled(
      env,
      std=0.9,
      command_name="velocity",
    )
    expected_yaw = torch.exp(torch.tensor([-1.0 / 0.9**2], device=device))
    assert torch.allclose(yaw_reward, expected_yaw)

    robot.write_root_link_velocity_b_to_sim(
      torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], device=device)
    )
    env.sim.forward()
    roll_reward = track_angular_velocity_speed_scaled(
      env,
      std=0.9,
      command_name="velocity",
    )
    expected_roll = torch.exp(torch.tensor([-0.2 / 0.9**2], device=device))
    assert torch.allclose(roll_reward, expected_roll)
  finally:
    env.close()


def test_v1_body_angular_acceleration_penalty_uses_mujoco_sensor(
  device: str,
) -> None:
  env = _make_env(device, num_envs=4)
  try:
    env.reset()
    env.sim.forward()
    penalty = body_angular_acceleration_l2(env)
    sensor = envs_mdp.builtin_sensor(env, "robot/imu_angacc")
    assert torch.isfinite(penalty).all()
    assert torch.allclose(penalty, torch.sum(torch.square(sensor), dim=-1))
  finally:
    env.close()


def test_v1_curriculum_follows_four_fixed_speed_stages(device: str) -> None:
  cfg = tony5_velocity_aero_env_cfg()
  command_cfg = cast(Tony5RadialVelocityCommandCfg, cfg.commands["velocity"])
  cfg.scene.num_envs = 64
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    command = _command(env)
    assert command.curriculum_stage == 0
    assert command.current_vmax == 2.0

    first_switch_step = (
      command_cfg.curriculum_switch_iterations[0]
      * command_cfg.curriculum_steps_per_iteration
    )
    env.common_step_counter = first_switch_step - 1
    command.update_curriculum_metrics()
    assert command.curriculum_stage == 0
    assert command.current_vmax == 2.0

    env.common_step_counter = first_switch_step
    command.update_curriculum_metrics()
    assert command.curriculum_stage == 1
    assert command.current_vmax == 6.0

    second_switch_step = (
      command_cfg.curriculum_switch_iterations[1]
      * command_cfg.curriculum_steps_per_iteration
    )
    env.common_step_counter = second_switch_step
    command.update_curriculum_metrics()
    assert command.curriculum_stage == 2
    assert command.current_vmax == 10.0

    final_switch_step = (
      command_cfg.curriculum_switch_iterations[2]
      * command_cfg.curriculum_steps_per_iteration
    )
    env.common_step_counter = final_switch_step
    command.update_curriculum_metrics()
    assert command.curriculum_stage == 3
    assert command.current_vmax == 27.78

    env.common_step_counter += 1000
    command.update_curriculum_metrics()
    assert command.curriculum_stage == 3
    assert command.current_vmax == 27.78
  finally:
    env.close()


def test_v1_high_speed_rmse_uses_only_valid_high_speed_samples(device: str) -> None:
  cfg = tony5_velocity_aero_env_cfg()
  command_cfg = cast(Tony5RadialVelocityCommandCfg, cfg.commands["velocity"])
  command_cfg.curriculum_window_steps = 1
  cfg.scene.num_envs = 64
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    command = _command(env)
    env.scene["robot"].write_root_link_velocity_b_to_sim(
      torch.zeros((64, 6), device=device)
    )
    env.sim.forward()
    command.vel_command_b.zero_()
    command.vel_command_b[:32, 0] = command.current_vmax
    env.common_step_counter += 1
    command.update_curriculum_metrics()

    assert torch.isclose(
      command.high_speed_horizontal_rmse_value[0],
      torch.tensor(command.current_vmax, device=device),
    )
    assert torch.allclose(
      command.high_speed_sample_fraction_value,
      torch.full((64,), 0.5, device=device),
    )
    assert command.high_speed_steady_rmse_value[0] == 0.0
    assert command.settled_high_speed_sample_fraction_value[0] == 0.0

    command._time_since_command.fill_(CURRICULUM_SETTLING_TIME_S)
    env.common_step_counter += 1
    command.update_curriculum_metrics()
    assert torch.isclose(
      command.high_speed_steady_rmse_value[0],
      torch.tensor(command.current_vmax, device=device),
    )
    assert torch.allclose(
      command.settled_high_speed_sample_fraction_value,
      torch.full((64,), 0.5, device=device),
    )
    assert command.mean_time_since_command_value[0] >= CURRICULUM_SETTLING_TIME_S
    assert command.curriculum_stage == 0
  finally:
    env.close()


def test_v1_curriculum_does_not_advance_without_settled_high_speed_samples(
  device: str,
) -> None:
  cfg = tony5_velocity_aero_env_cfg()
  command_cfg = cast(Tony5RadialVelocityCommandCfg, cfg.commands["velocity"])
  command_cfg.curriculum_window_steps = 1
  cfg.scene.num_envs = 64
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    command = _command(env)
    assert torch.allclose(
      env.scene["robot"].data.root_link_lin_vel_b,
      torch.zeros((64, 3), device=device),
    )
    command.vel_command_b.zero_()
    command.vel_command_b[:, 0] = command.current_vmax
    command._time_since_command.zero_()
    env.common_step_counter += 1
    command.update_curriculum_metrics()
    assert command.high_speed_horizontal_rmse_value[0] == command.current_vmax
    assert command.high_speed_sample_fraction_value[0] == 1.0
    assert command.high_speed_steady_rmse_value[0] == 0.0
    assert command.settled_high_speed_sample_fraction_value[0] == 0.0
    assert command.curriculum_stage == 0

    command._time_since_command.fill_(CURRICULUM_SETTLING_TIME_S)
    env.common_step_counter += 1
    command.update_curriculum_metrics()
    assert command.curriculum_stage == 0
  finally:
    env.close()


def test_v1_radial_4096_env_rollout_is_finite(device: str) -> None:
  env = _make_env(device, num_envs=4096)
  try:
    obs, _ = env.reset()
    command = _command(env)
    assert command.current_vmax == 2.0
    for _ in range(8):
      action = torch.zeros((4096, 4), device=device)
      obs, reward, terminated, truncated, _ = env.step(action)
      actor_obs = obs["actor"]
      assert isinstance(actor_obs, torch.Tensor)
      assert torch.isfinite(actor_obs).all()
      assert torch.isfinite(reward).all()
      assert torch.isfinite(command.command).all()
      assert torch.isfinite(terminated.float()).all()
      assert torch.isfinite(truncated.float()).all()
  finally:
    env.close()
