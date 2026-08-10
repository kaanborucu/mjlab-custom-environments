"""Tests for the play script."""

from pathlib import Path

import pytest

from mjlab.scripts.play import _get_latest_local_checkpoint


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
