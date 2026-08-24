"""Tests for the Quad Mini Tuned flat-terrain task."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import mujoco
import pytest
import torch

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.quad_mini_tuned.quad_constants import (
  DEFAULT_JOINT_POS,
  get_quad_mini_tuned_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.envs.types import VecEnvObs
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.quad_mini_tuned import (
  STUDENT_TASK_ID,
  TASK_ID,
  TEACHER_TASK_ID,
  mdp,
)
from mjlab.tasks.quad_mini_tuned.rl_cfg import (
  QuadMiniTunedDistillationRunnerCfg,
)
from mjlab.tasks.quad_mini_tuned.teacher_student_runner import (
  QuadMiniTunedTeacherRunner,
  _get_latest_teacher_checkpoint,
)
from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
)
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


@pytest.fixture(scope="module")
def env() -> Iterator[ManagerBasedRlEnv]:
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 2
  instance = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  yield instance
  instance.close()


@pytest.fixture(scope="module")
def teacher_student_envs() -> Iterator[dict[str, ManagerBasedRlEnv]]:
  instances: dict[str, ManagerBasedRlEnv] = {}
  for task_id in (TEACHER_TASK_ID, STUDENT_TASK_ID):
    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = 2
    instances[task_id] = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  yield instances
  for instance in instances.values():
    instance.close()


@pytest.fixture(scope="module")
def teacher_curriculum_env() -> Iterator[ManagerBasedRlEnv]:
  cfg = load_env_cfg(TEACHER_TASK_ID)
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
  assert all(
    "KNEE_collision" not in (model.geom(i).name or "") for i in range(model.ngeom)
  )
  for foot_geom_name in (
    "RF_FOOT_geom",
    "LF_FOOT_geom",
    "RH_FOOT_geom",
    "LH_FOOT_geom",
  ):
    geom_id = model.geom(foot_geom_name).id
    assert model.geom_contype[geom_id] == 1
    assert model.geom_rgba[geom_id, 3] == pytest.approx(0.0)


def test_quad_uses_ak70_10_dc_motor_limits() -> None:
  robot_cfg = get_quad_mini_tuned_robot_cfg()
  assert robot_cfg.articulation is not None
  assert len(robot_cfg.articulation.actuators) == 1
  actuator = robot_cfg.articulation.actuators[0]
  assert isinstance(actuator, DcMotorActuatorCfg)
  assert actuator.velocity_limit == pytest.approx(50.3)


def test_quad_task_is_registered() -> None:
  assert {TASK_ID, TEACHER_TASK_ID, STUDENT_TASK_ID} <= set(list_tasks())


def test_quad_simulation_contact_limit_is_per_world() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert cfg.sim.nconmax == 128


def test_quad_policy_clips_actions() -> None:
  assert load_rl_cfg(TASK_ID).clip_actions == 1.0


def test_quad_position_targets_clip_to_hard_limits(env: ManagerBasedRlEnv) -> None:
  action = cast(JointPositionAction, env.action_manager.get_term("joint_pos"))
  raw_actions = torch.full((env.num_envs, action.action_dim), 100.0, device=env.device)

  action.process_actions(raw_actions)
  action.apply_actions()

  limits = env.scene["robot"].data.default_joint_pos_limits[:, action.target_ids]
  targets = env.scene["robot"].data.joint_pos_target[:, action.target_ids]
  assert torch.all(targets >= limits[..., 0])
  assert torch.all(targets <= limits[..., 1])


def test_quad_qpos0_randomization_preserves_home_pose(
  env: ManagerBasedRlEnv,
) -> None:
  robot = env.scene["robot"]
  qpos0 = env.sim.model.qpos0[:, robot.indexing.joint_q_adr]
  expected_home = torch.tensor(
    [DEFAULT_JOINT_POS[name] for name in robot.joint_names],
    device=env.device,
  ).expand_as(robot.data.default_joint_pos)

  assert torch.all(torch.abs(qpos0) <= 0.03)
  torch.testing.assert_close(robot.data.default_joint_pos, expected_home)


def test_quad_reset_uses_exact_spawn_height(env: ManagerBasedRlEnv) -> None:
  env.reset()
  root_height = env.scene["robot"].data.root_link_pos_w[:, 2]

  torch.testing.assert_close(root_height, torch.full_like(root_height, 0.22))


def test_quad_feet_clearance_is_not_command_gated() -> None:
  site_pos_w = torch.tensor([[[0.0, 0.0, 0.10], [0.0, 0.0, 0.30]]], dtype=torch.float32)
  site_lin_vel_w = torch.tensor(
    [[[3.0, 4.0, 0.0], [0.0, 16.0, 0.0]]], dtype=torch.float32
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={
        "robot": SimpleNamespace(
          data=SimpleNamespace(
            site_pos_w=site_pos_w,
            site_lin_vel_w=site_lin_vel_w,
          )
        )
      }
    ),
  )
  asset_cfg = cast(SceneEntityCfg, SimpleNamespace(name="robot", site_ids=slice(None)))

  actual = mdp.quad_feet_clearance(env, target_height=0.25, asset_cfg=asset_cfg)
  expected = torch.tensor([0.15 * 5.0**0.5 + 0.05 * 4.0])

  torch.testing.assert_close(actual, expected)


def test_quad_feet_air_time_uses_completed_air_phase() -> None:
  last_air_time = torch.tensor([[0.3, 0.2, 0.1, 0.4]])
  first_contact = torch.tensor([[True, False, True, True]])
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      current_air_time=torch.zeros_like(last_air_time),
      last_air_time=last_air_time,
    ),
    compute_first_contact=lambda dt: first_contact,
  )
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={"feet": sensor},
      command_manager=SimpleNamespace(
        get_command=lambda name: torch.tensor([[1.0, 0.0, 0.0]])
      ),
      step_dt=0.02,
    ),
  )

  actual = mdp.quad_feet_air_time(env, sensor_name="feet", command_name="twist")

  torch.testing.assert_close(actual, torch.tensor([0.5]))


def test_quad_playground_soft_limit_penalty_scales_endpoints() -> None:
  hard_limits = torch.tensor(
    [
      [
        [-0.1745, 3.316],
        [-0.1745, 3.316],
        [-1.745529, 0.001],
        [-1.745529, 0.001],
      ]
    ],
    dtype=torch.float32,
  )
  joint_pos = torch.tensor([[-0.2, 3.2, -1.7, 0.1]], dtype=torch.float32)
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      scene={
        "robot": SimpleNamespace(
          data=SimpleNamespace(
            default_joint_pos_limits=hard_limits,
            joint_pos=joint_pos,
          )
        )
      }
    ),
  )

  actual = mdp.quad_joint_pos_limits(env, asset_cfg=SceneEntityCfg("robot"))
  lower = hard_limits[0, :, 0] * 0.95
  upper = hard_limits[0, :, 1] * 0.95
  expected = (
    (-(joint_pos[0] - lower).clip(max=0.0) + (joint_pos[0] - upper).clip(min=0.0))
    .sum()
    .reshape(1)
  )
  torch.testing.assert_close(actual, expected)


def test_quad_teacher_student_configs() -> None:
  base_cfg = load_env_cfg(TASK_ID)
  teacher_cfg = load_env_cfg(TEACHER_TASK_ID)
  student_cfg = load_env_cfg(STUDENT_TASK_ID)

  assert teacher_cfg.rewards == base_cfg.rewards
  assert student_cfg.rewards == base_cfg.rewards
  assert tuple(teacher_cfg.observations["actor"].terms) == ("privileged_state",)
  assert teacher_cfg.observations["actor"].history_length == 5
  assert teacher_cfg.observations["actor"].flatten_history_dim
  assert not teacher_cfg.observations["actor"].enable_corruption
  assert tuple(teacher_cfg.observations["critic"].terms) == ("privileged_state",)
  assert teacher_cfg.observations["critic"].history_length == 5
  assert teacher_cfg.observations["critic"].flatten_history_dim
  assert not teacher_cfg.observations["critic"].enable_corruption
  assert student_cfg.observations["actor"].history_length == 5
  assert student_cfg.observations["actor"].flatten_history_dim
  assert student_cfg.observations["actor"].enable_corruption
  assert tuple(student_cfg.observations["critic"].terms) == ("privileged_state",)
  assert student_cfg.observations["critic"].history_length == 5
  assert student_cfg.observations["critic"].flatten_history_dim
  assert not student_cfg.observations["critic"].enable_corruption


def test_quad_teacher_domain_randomization_curriculum_is_training_only() -> None:
  base_cfg = load_env_cfg(TASK_ID)
  teacher_cfg = load_env_cfg(TEACHER_TASK_ID)
  teacher_play_cfg = load_env_cfg(TEACHER_TASK_ID, play=True)
  student_cfg = load_env_cfg(STUDENT_TASK_ID)

  assert tuple(teacher_cfg.curriculum) == ("domain_randomization",)
  curriculum_params = teacher_cfg.curriculum["domain_randomization"].params
  assert curriculum_params["stages"] == [
    {"step": 0, "light": 0.0, "physical": 0.0},
    {"step": 12_000, "light": 0.25, "physical": 0.0},
    {"step": 36_000, "light": 0.5, "physical": 0.25},
    {"step": 72_000, "light": 0.75, "physical": 0.5},
    {"step": 120_000, "light": 1.0, "physical": 1.0},
  ]
  for group_targets in curriculum_params["event_targets"].values():
    for event_name, param_targets in group_targets.items():
      for param_name, (neutral, _target) in param_targets.items():
        assert teacher_cfg.events[event_name].params[param_name] == neutral
  assert not base_cfg.curriculum
  assert not student_cfg.curriculum
  assert not teacher_play_cfg.curriculum
  assert teacher_cfg.events["floor_friction"].params["ranges"] == (1.0, 1.0)
  assert teacher_cfg.events["pd_gains"].params["kp_range"] == (25.0, 25.0)
  assert teacher_cfg.events["torso_com"].params["ranges"] == {
    0: (0.0, 0.0),
    1: (0.0, 0.0),
    2: (0.0, 0.0),
  }
  assert teacher_cfg.events["reset_root_state"].params["velocity_range"]["z"] == (
    0.0,
    0.0,
  )
  assert teacher_cfg.events["reset_joint_state"].params["position_range"] == (
    0.0,
    0.0,
  )
  assert base_cfg.events["floor_friction"].params["ranges"] == (0.4, 1.5)
  assert student_cfg.events["floor_friction"].params["ranges"] == (0.4, 1.5)
  assert teacher_play_cfg.events["floor_friction"].params["ranges"] == (0.4, 1.5)


def test_quad_teacher_curriculum_progresses_to_full_ranges(
  teacher_curriculum_env: ManagerBasedRlEnv,
) -> None:
  env = teacher_curriculum_env
  env.common_step_counter = 0
  env.reset()
  robot = env.scene["robot"]

  torch.testing.assert_close(
    env.sim.model.qpos0[:, robot.indexing.joint_q_adr],
    torch.zeros_like(robot.data.default_joint_pos),
  )
  torch.testing.assert_close(
    robot.data.encoder_bias,
    torch.zeros_like(robot.data.encoder_bias),
  )
  torch.testing.assert_close(
    env.sim.model.body_mass[:],
    env.sim.get_default_field("body_mass").expand_as(env.sim.model.body_mass[:]),
  )
  torch.testing.assert_close(
    env.sim.model.body_inertia[:],
    env.sim.get_default_field("body_inertia").expand_as(env.sim.model.body_inertia[:]),
  )
  torch.testing.assert_close(
    robot.data.root_link_vel_w[:, 2],
    torch.zeros(env.num_envs, device=env.device),
  )

  env.common_step_counter = 12_000
  env.reset()
  floor_cfg = env.event_manager.get_term_cfg("floor_friction")
  torso_com_cfg = env.event_manager.get_term_cfg("torso_com")
  root_cfg = env.event_manager.get_term_cfg("reset_root_state")
  joint_cfg = env.event_manager.get_term_cfg("reset_joint_state")
  assert floor_cfg.params["ranges"] == pytest.approx((0.85, 1.125))
  assert torso_com_cfg.params["ranges"] == {
    0: (0.0, 0.0),
    1: (0.0, 0.0),
    2: (0.0, 0.0),
  }
  assert root_cfg.params["velocity_range"]["z"] == pytest.approx((-0.0125, 0.0125))
  assert joint_cfg.params["position_range"] == pytest.approx((-0.00125, 0.00125))
  assert env.extras["log"]["Curriculum/domain_randomization/stage"] == 1.0
  assert env.extras["log"]["Curriculum/domain_randomization/light"] == 0.25
  assert env.extras["log"]["Curriculum/domain_randomization/physical"] == 0.0

  env.common_step_counter = 120_000
  env.reset()
  assert floor_cfg.params["ranges"] == pytest.approx((0.4, 1.5))
  assert torso_com_cfg.params["ranges"] == {
    0: pytest.approx((-0.1, 0.1)),
    1: pytest.approx((-0.1, 0.1)),
    2: pytest.approx((-0.1, 0.1)),
  }
  assert root_cfg.params["velocity_range"]["z"] == pytest.approx((-0.05, 0.05))
  assert joint_cfg.params["position_range"] == pytest.approx((-0.005, 0.005))


def test_quad_mass_randomization_does_not_accumulate(
  teacher_curriculum_env: ManagerBasedRlEnv,
) -> None:
  env = teacher_curriculum_env
  all_bodies_cfg = SceneEntityCfg("robot")
  trunk_cfg = SceneEntityCfg("robot", body_names=("trunk",))
  all_bodies_cfg.resolve(env.scene)
  trunk_cfg.resolve(env.scene)
  robot = env.scene["robot"]
  all_body_ids = robot.indexing.body_ids[all_bodies_cfg.body_ids]
  trunk_id = robot.indexing.body_ids[trunk_cfg.body_ids]
  defaults = env.sim.get_default_field("body_mass")

  for _ in range(2):
    mdp.quad_body_mass(env, None, (2.0, 2.0), "scale", all_bodies_cfg)
    mdp.quad_body_mass(env, None, (1.0, 1.0), "add", trunk_cfg)

  expected = (
    defaults[all_body_ids].unsqueeze(0).expand(env.num_envs, -1) * 2.0
  ).clone()
  trunk_mask = all_body_ids == trunk_id.item()
  expected[:, trunk_mask] += 1.0
  torch.testing.assert_close(
    env.sim.model.body_mass[:, all_body_ids],
    expected,
  )


def test_quad_teacher_runner_reapplies_curriculum_after_resume() -> None:
  runner = QuadMiniTunedTeacherRunner.__new__(QuadMiniTunedTeacherRunner)
  wrapped_env = MagicMock()
  wrapped_env.unwrapped.curriculum_manager.active_terms = ["domain_randomization"]
  runner.env = wrapped_env

  with patch.object(
    VelocityOnPolicyRunner,
    "load",
    return_value={"env_state": {"common_step_counter": 120_000}},
  ):
    infos = runner.load("checkpoint.pt")

  assert infos == {"env_state": {"common_step_counter": 120_000}}
  wrapped_env.reset.assert_called_once_with()
  assert load_runner_cls(TEACHER_TASK_ID) is QuadMiniTunedTeacherRunner


def test_quad_student_runner_uses_clean_teacher_group() -> None:
  cfg = load_rl_cfg(STUDENT_TASK_ID)
  assert isinstance(cfg, QuadMiniTunedDistillationRunnerCfg)
  assert cfg.obs_groups == {"student": ("actor",), "teacher": ("critic",)}
  assert cfg.algorithm.class_name == "Distillation"


def test_quad_student_auto_selects_latest_teacher_checkpoint(tmp_path) -> None:
  log_root = tmp_path / "logs" / "rsl_rl"
  student_run = log_root / "quad_mini_tuned_student" / "student-run"
  teacher_old = log_root / "quad_mini_tuned_teacher" / "2026-01-01"
  teacher_latest = log_root / "quad_mini_tuned_teacher" / "2026-01-02"
  for checkpoint in (
    teacher_old / "model_100.pt",
    teacher_latest / "model_100.pt",
    teacher_latest / "model_200.pt",
  ):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()

  assert _get_latest_teacher_checkpoint(str(student_run)) == (
    teacher_latest / "model_200.pt"
  )


def test_quad_student_requires_a_teacher_checkpoint(tmp_path) -> None:
  student_run = tmp_path / "logs" / "rsl_rl" / "quad_mini_tuned_student" / "run"
  with pytest.raises(ValueError, match="Train the teacher first"):
    _get_latest_teacher_checkpoint(str(student_run))


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


def test_quad_teacher_student_observation_shapes(
  teacher_student_envs: dict[str, ManagerBasedRlEnv],
) -> None:
  teacher = teacher_student_envs[TEACHER_TASK_ID]
  student = teacher_student_envs[STUDENT_TASK_ID]

  teacher_obs, _ = teacher.reset()
  student_obs, _ = student.reset()
  assert _tensor_observation(teacher_obs, "actor").shape == (2, 675)
  assert _tensor_observation(teacher_obs, "critic").shape == (2, 675)
  assert _tensor_observation(student_obs, "actor").shape == (2, 300)
  assert _tensor_observation(student_obs, "critic").shape == (2, 675)

  student_obs, reward, terminated, truncated, _ = student.step(
    torch.zeros((2, 12), device="cpu")
  )
  assert _tensor_observation(student_obs, "actor").shape == (2, 300)
  assert _tensor_observation(student_obs, "critic").shape == (2, 675)
  assert torch.isfinite(reward).all()
  assert not terminated.any()
  assert not truncated.any()
