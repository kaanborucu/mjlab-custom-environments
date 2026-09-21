"""Focused validation for the TONY5 V5 minimum-time target-pose task."""

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.manager_based.tony5 import (
  POSITION_MIN_TIME_TASK_ID,
  POSITION_MIN_TIME_TEACHER_TASK_ID,
  VELOCITY_AERO_OMNI_V3_TASK_ID,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_actions import (
  Tony5OmniV3RotorSpeedActionCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_env_cfg import (
  tony5_velocity_aero_omni_v3_env_cfg,
)
from mjlab.tasks.manager_based.tony5.tony5_position_min_time_command import (
  Tony5PositionMinTimeCommand,
)
from mjlab.tasks.manager_based.tony5.tony5_position_min_time_rewards import (
  TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS,
  crash_penalty,
  near_target_settling_cost,
  success_reward,
)
from mjlab.tasks.manager_based.tony5.tony5_position_min_time_terminations import (
  TONY5_POSITION_MIN_TIME_BODY_ANGULAR_SPEED_LIMIT,
  TONY5_POSITION_MIN_TIME_QACC_LIMIT,
  TONY5_POSITION_MIN_TIME_ROOT_SPEED_LIMIT,
  TONY5_SUCCESS_ANGULAR_SPEED_TOLERANCE,
  TONY5_SUCCESS_LINEAR_SPEED_TOLERANCE,
  TONY5_SUCCESS_POSITION_TOLERANCE,
  TONY5_SUCCESS_YAW_TOLERANCE,
  success_mask,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg


@pytest.fixture(scope="module")
def device() -> str:
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def _make_v5_env(device: str, num_envs: int = 1) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(POSITION_MIN_TIME_TASK_ID)
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def test_v5_is_registered_and_does_not_mutate_v3() -> None:
  import mjlab.tasks  # noqa: F401

  assert POSITION_MIN_TIME_TASK_ID in list_tasks()
  assert VELOCITY_AERO_OMNI_V3_TASK_ID in list_tasks()
  v3_cfg = tony5_velocity_aero_omni_v3_env_cfg()
  v5_cfg = load_env_cfg(POSITION_MIN_TIME_TASK_ID)
  assert set(v5_cfg.commands) == {"target_pose"}
  assert set(v3_cfg.commands) == {"velocity"}
  assert "linear_velocity_b" in v5_cfg.observations["actor"].terms
  assert "linear_velocity_w" not in v5_cfg.observations["actor"].terms
  assert "heading_sin_cos" in v5_cfg.observations["actor"].terms
  assert "linear_velocity_b" in v5_cfg.observations["critic"].terms
  assert "linear_velocity_w" not in v5_cfg.observations["critic"].terms
  assert "heading_sin_cos" in v5_cfg.observations["critic"].terms
  assert v5_cfg.scale_rewards_by_dt is False
  assert v5_cfg.episode_length_s == pytest.approx(10.0)
  assert v3_cfg.scale_rewards_by_dt is True
  assert cast(
    Tony5OmniV3RotorSpeedActionCfg, v3_cfg.actions["rotor_speed"]
  ).wind.enable_wind
  assert cast(
    Tony5OmniV3RotorSpeedActionCfg,
    v5_cfg.actions["rotor_speed"],
  ).wind.enable_wind
  assert cast(
    Tony5OmniV3RotorSpeedActionCfg,
    v5_cfg.actions["rotor_speed"],
  ).wind.randomize_background_wind
  assert cast(
    Tony5OmniV3RotorSpeedActionCfg,
    v5_cfg.actions["rotor_speed"],
  ).wind.enable_gusts
  assert load_rl_cfg(POSITION_MIN_TIME_TASK_ID).experiment_name == (
    "tony5_position_min_time_v5"
  )
  rl_cfg = cast(RslRlOnPolicyRunnerCfg, load_rl_cfg(POSITION_MIN_TIME_TASK_ID))
  assert not rl_cfg.resume
  assert rl_cfg.algorithm.gamma == pytest.approx(0.999)
  assert rl_cfg.algorithm.entropy_coef == pytest.approx(0.01)
  assert v5_cfg.terminations["time_out"].time_out
  assert not v5_cfg.terminations["success"].time_out
  safety_params = v5_cfg.terminations["numerical_safety_failure"].params
  assert safety_params["root_speed_limit"] == pytest.approx(100.0)
  assert safety_params["body_angular_speed_limit"] == pytest.approx(20.0)
  assert safety_params["qacc_limit"] == pytest.approx(300_000.0)
  assert TONY5_POSITION_MIN_TIME_ROOT_SPEED_LIMIT == pytest.approx(100.0)
  assert TONY5_POSITION_MIN_TIME_BODY_ANGULAR_SPEED_LIMIT == pytest.approx(20.0)
  assert TONY5_POSITION_MIN_TIME_QACC_LIMIT == pytest.approx(300_000.0)
  assert v5_cfg.rewards["success"].weight == pytest.approx(300.0)
  assert v5_cfg.rewards["crash"].weight == pytest.approx(-100.0)
  assert v5_cfg.scale_rewards_by_dt is False
  assert TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS == {
    "distance_progress": 10.0,
    "time_penalty": -0.11,
    "settling": -0.00001,
    "action_rate": -0.03,
    "motor_torque": -1.0e-8,
    "success": 300.0,
    "crash": -100.0,
  }
  assert set(v5_cfg.rewards) == set(TONY5_POSITION_MIN_TIME_REWARD_WEIGHTS)


def test_v5_teacher_is_privileged_single_frame_with_compact_network() -> None:
  import mjlab.tasks  # noqa: F401

  assert POSITION_MIN_TIME_TEACHER_TASK_ID in list_tasks()
  cfg = load_env_cfg(POSITION_MIN_TIME_TEACHER_TASK_ID)
  play_cfg = load_env_cfg(POSITION_MIN_TIME_TEACHER_TASK_ID, play=True)
  rl_cfg = cast(
    RslRlOnPolicyRunnerCfg,
    load_rl_cfg(POSITION_MIN_TIME_TEACHER_TASK_ID),
  )

  assert cfg.observations["actor"].history_length == 1
  assert cfg.observations["critic"].history_length == 1
  assert not cfg.observations["actor"].enable_corruption
  assert not cfg.observations["critic"].enable_corruption
  assert cfg.observations["actor"].flatten_history_dim
  assert cfg.observations["critic"].flatten_history_dim
  assert (
    cfg.observations["actor"].terms.keys() == cfg.observations["critic"].terms.keys()
  )
  assert {
    "target_pose",
    "root_position_w",
    "root_quaternion_w",
    "wind_world",
    "loaded_voltage",
  } <= set(cfg.observations["actor"].terms)
  assert play_cfg.scene.num_envs == 1
  assert rl_cfg.experiment_name == "tony5_position_min_time_v5_teacher"
  assert rl_cfg.actor.hidden_dims == (256, 128, 64)
  assert rl_cfg.critic.hidden_dims == (256, 128, 64)


def test_v5_samples_fixed_episode_targets_and_expected_observation_shape(
  device: str,
) -> None:
  env = _make_v5_env(device, num_envs=8)
  try:
    observations, _ = env.reset()
    command = cast(
      Tony5PositionMinTimeCommand,
      env.command_manager.get_term("target_pose"),
    )
    targets_before = command.command.clone()
    horizontal_distance = torch.linalg.vector_norm(targets_before[:, :2], dim=-1)
    target_world_z = targets_before[:, 2] + env.scene.env_origins[:, 2]
    assert torch.all((horizontal_distance >= 1.0) & (horizontal_distance <= 15.0))
    assert torch.all(target_world_z >= 0.5)
    assert isinstance(observations["actor"], torch.Tensor)
    assert observations["actor"].shape == (8, 125)
    assert isinstance(observations["critic"], torch.Tensor)
    assert observations["critic"].shape == (8, 125)

    settling_cost = near_target_settling_cost(env)
    assert torch.all((settling_cost >= 0.0) & (settling_cost <= 1.0))

    env.command_manager.compute(dt=env.step_dt)
    assert torch.equal(command.command, targets_before)
  finally:
    env.close()


def test_success_mask_requires_all_conditions() -> None:
  good = success_mask(
    torch.tensor([0.27]),
    torch.tensor([0.08]),
    torch.tensor([0.49]),
    torch.tensor([0.49]),
  )
  bad = success_mask(
    torch.tensor([0.29]),
    torch.tensor([0.08]),
    torch.tensor([0.51]),
    torch.tensor([0.49]),
  )
  assert bool(good.item())
  assert not bool(bad.item())
  assert TONY5_SUCCESS_POSITION_TOLERANCE == pytest.approx(0.28125)
  assert TONY5_SUCCESS_YAW_TOLERANCE == pytest.approx(5.0 * torch.pi / 180.0)
  assert TONY5_SUCCESS_LINEAR_SPEED_TOLERANCE == pytest.approx(0.5)
  assert TONY5_SUCCESS_ANGULAR_SPEED_TOLERANCE == pytest.approx(0.75)


def test_timeout_has_neither_success_nor_crash_reward() -> None:
  timeout_env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    termination_manager=SimpleNamespace(
      active_terms=["time_out"],
    ),
  )
  fake_env = cast(ManagerBasedRlEnv, timeout_env)
  assert torch.equal(success_reward(fake_env), torch.zeros(1))
  assert torch.equal(crash_penalty(fake_env), torch.zeros(1))
