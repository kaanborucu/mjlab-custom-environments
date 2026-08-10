"""Tests for the Quad Mini Tuned flat-terrain task."""

from collections.abc import Iterator

import mujoco
import pytest
import torch

from mjlab.asset_zoo.robots.quad_mini_tuned.quad_constants import (
  get_quad_mini_tuned_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.types import VecEnvObs
from mjlab.tasks.quad_mini_tuned import TASK_ID
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg


@pytest.fixture(scope="module")
def env() -> Iterator[ManagerBasedRlEnv]:
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 2
  instance = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  yield instance
  instance.close()


def test_quad_robot_asset_compiles() -> None:
  robot = Entity(get_quad_mini_tuned_robot_cfg())
  model: mujoco.MjModel = robot.compile()
  assert model.nq == 19
  assert model.nv == 18
  assert model.nu == 12
  assert robot.num_joints == 12
  assert robot.num_actuators == 12


def test_quad_task_is_registered() -> None:
  assert TASK_ID in list_tasks()


def test_quad_simulation_contact_limit_is_per_world() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert cfg.sim.nconmax == 128


def test_quad_policy_clips_actions() -> None:
  assert load_rl_cfg(TASK_ID).clip_actions == 1.0


def _tensor_observation(observations: VecEnvObs, name: str) -> torch.Tensor:
  value = observations[name]
  assert isinstance(value, torch.Tensor)
  return value


def test_quad_env_smoke(env: ManagerBasedRlEnv) -> None:
  observations, _ = env.reset()
  assert _tensor_observation(observations, "actor").shape == (2, 60)
  assert _tensor_observation(observations, "critic").shape == (2, 135)

  observations, reward, terminated, truncated, _ = env.step(
    torch.zeros((2, 12), device="cpu")
  )
  assert _tensor_observation(observations, "actor").shape == (2, 60)
  assert _tensor_observation(observations, "critic").shape == (2, 135)
  assert torch.isfinite(reward).all()
  assert not terminated.any()
  assert not truncated.any()
