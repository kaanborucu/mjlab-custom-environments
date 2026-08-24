"""Tests for the play script."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.play import (
  PlayConfig,
  RandomBaseTorqueDisturbance,
  Tony5KeyboardController,
  Tony5PlayResampler,
  _apply_gamepad_deadzone,
  _get_latest_local_checkpoint,
)
from mjlab.viewer.native.keys import (
  KEY_EQUAL,
  KEY_KP_ADD,
  KEY_KP_SUBTRACT,
  KEY_M,
  KEY_MINUS,
)


def test_play_defaults_enable_v3_wind_and_gusts() -> None:
  """V3 play keeps wind and gusts enabled unless explicitly disabled."""
  cfg = PlayConfig()
  assert cfg.wind
  assert cfg.gusts
  assert cfg.playback_speed == pytest.approx(1.2)


@pytest.mark.parametrize(
  ("value", "expected"),
  [(0.0, 0.0), (0.1, 0.0), (0.12, 0.0), (0.56, 0.5), (-1.0, -1.0)],
)
def test_gamepad_deadzone(value: float, expected: float) -> None:
  """Stick drift is removed and the remaining range is rescaled."""
  assert _apply_gamepad_deadzone(value) == pytest.approx(expected)


def test_keyboard_speed_keys_support_main_and_keypad_plus_minus() -> None:
  """Native TONY5 speed controls accept both keyboard layouts."""
  command = Mock()
  controller = object.__new__(Tony5KeyboardController)
  controller.command = command

  controller._handle_key(KEY_EQUAL)
  controller._handle_key(KEY_MINUS)
  controller._handle_key(KEY_KP_ADD)
  controller._handle_key(KEY_KP_SUBTRACT)

  step = 1.0
  assert command.adjust_keyboard_max_speed.call_args_list == [
    ((step,), {}),
    ((-step,), {}),
    ((step,), {}),
    ((-step,), {}),
  ]


def test_keyboard_m_key_requests_resampling() -> None:
  """Native M key invokes the play resampling callback."""
  callback = Mock()
  controller = object.__new__(Tony5KeyboardController)
  controller.command = Mock()
  controller._resample_callback = callback

  controller._handle_key(KEY_M)

  callback.assert_called_once_with()


def test_keyboard_lowercase_m_key_requests_resampling() -> None:
  """ASCII lowercase m is accepted by the native key path too."""
  callback = Mock()
  controller = object.__new__(Tony5KeyboardController)
  controller.command = Mock()
  controller._resample_callback = callback

  controller._handle_key(ord("m"))

  callback.assert_called_once_with()


def test_play_resampler_resamples_command_and_wind() -> None:
  """Play resampling updates both velocity and the optional wind process."""
  command = Mock()
  wind = Mock()
  resampler = Tony5PlayResampler(command, wind)

  resampler.resample()

  command.resample_now.assert_called_once_with()
  wind.reset.assert_called_once_with()


def test_get_latest_local_checkpoint_selects_latest_run_and_model(
  tmp_path: Path,
) -> None:
  """Automatic play mode selects the latest local model checkpoint."""
  experiment_dir = tmp_path / "bird_5dof_velocity"
  older_run = experiment_dir / "2026-08-10_10-00-00"
  latest_run = experiment_dir / "2026-08-10_11-00-00"
  older_run.mkdir(parents=True)
  latest_run.mkdir()
  (older_run / "model_1000.pt").touch()
  (latest_run / "model_100.pt").touch()
  (latest_run / "model_200.pt").touch()

  checkpoint = _get_latest_local_checkpoint(experiment_dir)

  assert checkpoint == latest_run / "model_200.pt"


def test_get_latest_local_checkpoint_reports_missing_checkpoints(
  tmp_path: Path,
) -> None:
  """Automatic play mode explains how to provide a checkpoint when missing."""
  with pytest.raises(ValueError, match="No local checkpoint found"):
    _get_latest_local_checkpoint(tmp_path / "missing_experiment")


def test_get_latest_local_checkpoint_can_skip_bootstrap_run(tmp_path: Path) -> None:
  """V3 play selection skips the incompatible V0 bootstrap checkpoint."""
  experiment_dir = tmp_path / "tony5_velocity_aero_omni_v3"
  bootstrap_run = experiment_dir / "v0_bootstrap"
  v3_run = experiment_dir / "2026-08-24_17-40-17"
  bootstrap_run.mkdir(parents=True)
  v3_run.mkdir()
  (bootstrap_run / "model_9999.pt").touch()
  (v3_run / "model_400.pt").touch()

  checkpoint = _get_latest_local_checkpoint(
    experiment_dir,
    excluded_run_names=frozenset({"v0_bootstrap"}),
  )

  assert checkpoint == v3_run / "model_400.pt"


def test_random_base_torque_disturbance_writes_only_bounded_torque() -> None:
  """Play disturbance writes zero force and bounded torque to the base body."""
  robot = Mock()
  robot.find_bodies.return_value = ([3], ["quad_base"])
  env = SimpleNamespace(
    scene={"robot": robot},
    num_envs=4,
    device="cpu",
    step_dt=0.01,
  )

  disturbance = RandomBaseTorqueDisturbance(
    cast(ManagerBasedRlEnv, env),
    peak_torque=0.01,
    resample_time_range=(0.03, 0.03),
    duration=0.02,
  )
  disturbance.apply()

  forces, torques = robot.write_external_wrench_to_sim.call_args.args[:2]
  assert torch.count_nonzero(forces) == 0
  assert forces.shape == (4, 1, 3)
  assert torques.shape == (4, 1, 3)
  assert torch.all(torques <= 0.01)
  assert torch.all(torques >= -0.01)
  assert robot.write_external_wrench_to_sim.call_args.kwargs["body_ids"] == [3]

  first_torques = torques.clone()
  disturbance.apply()
  _, active_torques = robot.write_external_wrench_to_sim.call_args.args[:2]
  assert torch.equal(active_torques, first_torques)
  disturbance.apply()
  _, cleared_torques = robot.write_external_wrench_to_sim.call_args.args[:2]
  assert torch.count_nonzero(cleared_torques) == 0
  disturbance.apply()
  _, resampled_torques = robot.write_external_wrench_to_sim.call_args.args[:2]
  assert torch.count_nonzero(resampled_torques) > 0
  assert not torch.equal(resampled_torques, first_torques)
