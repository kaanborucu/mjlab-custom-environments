"""Tests for the standalone crawler MJLab task."""

import math
from collections.abc import Iterator

import mujoco
import numpy as np
import pytest
import torch

import mjlab.tasks.crawler  # noqa: F401
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.crawler_3dof.robot_cfg import (
  ACTION_LIMIT,
  ACTION_SCALE,
  ACTUATOR_NAMES,
  CRAWLER_CFG,
  EXPECTED_ROBOT_MASS_KG,
  JOINT_LIMIT,
  JOINT_NAMES,
  PHYSICS_TIMESTEP,
  POLICY_DELAY_PHYSICS_STEPS,
  POLICY_DELAY_S,
  ROBOT_XML,
  SERVO_NO_LOAD_SPEED_RAD_S_7V4,
  SERVO_SPEED_SECONDS_PER_60_DEG_7V4,
  SERVO_STALL_TORQUE_NM_7V4,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.types import VecEnvObs
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg
from mjlab.tasks.crawler import (
  FLAT_STUDENT_TASK_ID,
  FLAT_TEACHER_TASK_ID,
  NO_IMU_ACTOR_TASK_ID,
  ROBUST_NO_IMU_ACTOR_TASK_ID,
  ROBUST_TASK_ID,
  STUDENT_TASK_ID,
  TASK_ID,
  TEACHER_TASK_ID,
)
from mjlab.tasks.crawler.actions import LowPassJointPositionActionCfg
from mjlab.tasks.crawler.env_cfg import (
  ACTION_TARGET_NEW_WEIGHT,
  ACTUATOR_RESPONSE_TIME_S,
  DOMAIN_RANDOMIZATION,
  FLAT_CURRICULUM_MAX_ITERATIONS,
  FLAT_CURRICULUM_STAGES,
  FOUNDATION_CURRICULUM_STAGES,
  IMU_BIAS_RANGES,
  IMU_DELAY_CONTROL_STEPS,
  IMU_NOISE_STD,
  OBSERVATION_NOISE,
  REWARD_WEIGHTS,
)
from mjlab.tasks.crawler.rl_cfg import (
  CRAWLER_FLAT_TEACHER_EXPERIMENT,
  CRAWLER_ROUGH_TEACHER_EXPERIMENT,
  CrawlerDistillationRunnerCfg,
)
from mjlab.tasks.crawler.teacher_student_runner import (
  CrawlerDistillationRunner,
  _get_latest_teacher_checkpoint,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import HfRandomUniformTerrainCfg
from mjlab.utils.noise import (
  GaussianNoiseCfg,
  NoiseModelWithAdditiveBiasCfg,
  UniformNoiseCfg,
)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  """Compile the robot-only model once."""
  return mujoco.MjSpec.from_file(str(ROBOT_XML)).compile()


@pytest.fixture(scope="module")
def env() -> Iterator[ManagerBasedRlEnv]:
  """Create a small CPU/GPU environment for integration tests."""
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = load_env_cfg(TASK_ID)
  cfg.scene.num_envs = 2
  cfg.sim.nan_guard.enabled = True
  instance = ManagerBasedRlEnv(cfg=cfg, device=device)
  yield instance
  instance.close()


def _tensor_observation(
  observations: VecEnvObs,
  name: str,
) -> torch.Tensor:
  value = observations[name]
  if not isinstance(value, torch.Tensor):
    raise TypeError(f"expected concatenated tensor for observation group {name!r}")
  return value


def _object_names(
  model: mujoco.MjModel,
  obj_type: mujoco.mjtObj,
  count: int,
) -> list[str]:
  return [mujoco.mj_id2name(model, obj_type, index) or "" for index in range(count)]


def test_robot_xml_compiles(model: mujoco.MjModel) -> None:
  assert model.nq == 10
  assert model.nv == 9
  assert model.nu == 3
  assert float(model.body_mass.sum()) == pytest.approx(EXPECTED_ROBOT_MASS_KG)
  geom_names = _object_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
  assert "ground" not in geom_names


def test_joint_names_and_axes(model: mujoco.MjModel) -> None:
  axes = {
    "joint_1": (0.0, 0.0, 1.0),
    "joint_2": (0.0, 1.0, 0.0),
    "joint_3": (0.0, 1.0, 0.0),
  }
  for name, expected_axis in axes.items():
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint_id >= 0
    np.testing.assert_allclose(model.jnt_axis[joint_id], expected_axis)


def test_joint_limits(model: mujoco.MjModel) -> None:
  for name in JOINT_NAMES:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    np.testing.assert_allclose(
      model.jnt_range[joint_id],
      (-JOINT_LIMIT, JOINT_LIMIT),
      atol=1.0e-9,
    )


def test_actuator_contract(model: mujoco.MjModel) -> None:
  names = _object_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
  assert names == list(ACTUATOR_NAMES)
  np.testing.assert_allclose(model.actuator_forcerange[:, 0], -3.491)
  np.testing.assert_allclose(model.actuator_forcerange[:, 1], 3.491)


def test_mjlab_actuator_uses_real_servo_no_load_speed() -> None:
  articulation = CRAWLER_CFG.articulation
  assert articulation is not None
  assert len(articulation.actuators) == 1
  actuator = articulation.actuators[0]
  assert isinstance(actuator, DcMotorActuatorCfg)
  assert actuator.target_names_expr == JOINT_NAMES
  assert actuator.velocity_limit == pytest.approx(math.radians(60.0) / 0.11)
  assert actuator.velocity_limit == pytest.approx(SERVO_NO_LOAD_SPEED_RAD_S_7V4)
  assert SERVO_SPEED_SECONDS_PER_60_DEG_7V4 == pytest.approx(0.11)
  assert actuator.saturation_effort == pytest.approx(SERVO_STALL_TORQUE_NM_7V4)
  assert POLICY_DELAY_S == pytest.approx(0.010)
  assert POLICY_DELAY_PHYSICS_STEPS == 5
  assert actuator.delay_min_lag == POLICY_DELAY_PHYSICS_STEPS
  assert actuator.delay_max_lag == POLICY_DELAY_PHYSICS_STEPS


def test_link_collision_and_visual_geometries_are_cylinders(
  model: mujoco.MjModel,
) -> None:
  for segment_index in range(4):
    for role in ("collision", "visual"):
      name = f"segment_{segment_index}_{role}"
      geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
      assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
      np.testing.assert_allclose(model.geom_size[geom_id, :2], (0.030, 0.070))


def test_task_is_registered() -> None:
  assert TASK_ID in list_tasks()
  assert NO_IMU_ACTOR_TASK_ID in list_tasks()
  assert ROBUST_TASK_ID in list_tasks()
  assert ROBUST_NO_IMU_ACTOR_TASK_ID in list_tasks()
  assert TEACHER_TASK_ID in list_tasks()
  assert STUDENT_TASK_ID in list_tasks()
  assert FLAT_TEACHER_TASK_ID in list_tasks()
  assert FLAT_STUDENT_TASK_ID in list_tasks()


def test_actor_uses_motor_feedback_controller_state_and_command() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert tuple(cfg.observations["actor"].terms) == (
    "joint_pos",
    "joint_vel",
    "imu_lin_acc",
    "imu_ang_vel",
    "gravity_vector",
    "last_action",
    "velocity_command",
  )
  assert cfg.observations["actor"].enable_corruption
  assert not cfg.observations["critic"].enable_corruption
  assert cfg.observations["actor"].terms["joint_pos"].params["biased"]
  assert not cfg.observations["critic"].terms["joint_pos"].params["biased"]


def test_no_imu_actor_variant_only_removes_imu_from_actor() -> None:
  cfg = load_env_cfg(ROBUST_NO_IMU_ACTOR_TASK_ID)
  assert cfg.observations["actor"].history_length == 5
  assert cfg.observations["actor"].flatten_history_dim
  assert cfg.observations["critic"].history_length is None
  assert tuple(cfg.observations["actor"].terms) == (
    "joint_pos",
    "joint_vel",
    "last_action",
    "velocity_command",
  )
  assert "imu_lin_acc" in cfg.observations["critic"].terms
  assert "imu_ang_vel" in cfg.observations["critic"].terms

  original_cfg = load_env_cfg(ROBUST_TASK_ID)
  assert cfg.actions == original_cfg.actions
  assert cfg.commands == original_cfg.commands
  assert cfg.rewards == original_cfg.rewards
  assert cfg.terminations == original_cfg.terminations
  assert cfg.curriculum == original_cfg.curriculum


def test_flat_no_imu_actor_variant_uses_continuous_flat_curriculum() -> None:
  cfg = load_env_cfg(NO_IMU_ACTOR_TASK_ID)
  foundation_cfg = load_env_cfg(TASK_ID)
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_type == "plane"
  assert cfg.scene.num_envs == foundation_cfg.scene.num_envs == 8192
  assert cfg.curriculum == foundation_cfg.curriculum
  assert cfg.observations["actor"].history_length == 5
  assert tuple(cfg.observations["actor"].terms) == (
    "joint_pos",
    "joint_vel",
    "last_action",
    "velocity_command",
  )


def test_flat_profile_uses_full_randomization_from_start() -> None:
  cfg = load_env_cfg(TASK_ID)
  curriculum_params = cfg.curriculum["training_stage"].params
  assert curriculum_params["stages"] == FLAT_CURRICULUM_STAGES
  assert "full_domain_randomization" not in cfg.events
  assert cfg.events["randomize_sliding_friction"].mode == "startup"
  assert (
    cfg.events["randomize_sliding_friction"].params["ranges"]
    == DOMAIN_RANDOMIZATION["sliding_friction"]
  )
  noise = cfg.observations["actor"].terms["joint_pos"].noise
  assert isinstance(noise, UniformNoiseCfg)
  assert (
    noise.n_min,
    noise.n_max,
  ) == OBSERVATION_NOISE["joint_pos"]
  assert "full_domain_randomization" not in load_env_cfg(ROBUST_TASK_ID).events


def test_rough_tasks_use_one_heightfield_without_an_extra_plane() -> None:
  foundation_cfg = load_env_cfg(TASK_ID)
  flat_no_imu_cfg = load_env_cfg(NO_IMU_ACTOR_TASK_ID)
  robust_cfg = load_env_cfg(ROBUST_TASK_ID)
  no_imu_cfg = load_env_cfg(ROBUST_NO_IMU_ACTOR_TASK_ID)
  assert foundation_cfg.scene.spec_fn is None
  assert flat_no_imu_cfg.scene.spec_fn is None
  assert robust_cfg.scene.spec_fn is None
  assert no_imu_cfg.scene.spec_fn is None
  assert robust_cfg.scene.terrain is not None
  assert robust_cfg.scene.terrain.terrain_generator is not None
  generator = robust_cfg.scene.terrain.terrain_generator
  assert generator.size == (50.0, 50.0)
  assert generator.num_rows == 1
  assert generator.num_cols == 1
  assert generator.border_width == 0.0
  rough = generator.sub_terrains["random_rough"]
  assert isinstance(rough, HfRandomUniformTerrainCfg)
  assert rough.horizontal_scale == 0.20
  assert rough.noise_range == (-0.020, 0.020)
  assert rough.downsampled_scale == 0.40


def test_teacher_student_tasks_are_isolated_and_share_robust_training() -> None:
  teacher_cfg = load_env_cfg(TEACHER_TASK_ID)
  student_cfg = load_env_cfg(STUDENT_TASK_ID)
  robust_cfg = load_env_cfg(ROBUST_TASK_ID)
  no_imu_cfg = load_env_cfg(ROBUST_NO_IMU_ACTOR_TASK_ID)

  assert teacher_cfg.rewards == robust_cfg.rewards
  assert teacher_cfg.curriculum == robust_cfg.curriculum
  assert student_cfg.rewards == no_imu_cfg.rewards
  assert student_cfg.curriculum == no_imu_cfg.curriculum
  assert teacher_cfg.scene.num_envs == student_cfg.scene.num_envs == 8192
  assert tuple(teacher_cfg.observations["actor"].terms) == ("privileged_state",)
  assert teacher_cfg.observations["actor"].history_length == 10
  assert teacher_cfg.observations["critic"].history_length == 10
  assert tuple(student_cfg.observations["critic"].terms) == ("privileged_state",)
  assert student_cfg.observations["critic"].history_length == 10
  assert student_cfg.observations["actor"].history_length == 5
  assert tuple(student_cfg.observations["actor"].terms) == (
    "joint_pos",
    "joint_vel",
    "imu_lin_acc",
    "imu_ang_vel",
    "last_action",
    "velocity_command",
  )
  assert "gravity_vector" not in student_cfg.observations["actor"].terms
  for name in ("imu_lin_acc", "imu_ang_vel"):
    term = student_cfg.observations["actor"].terms[name]
    assert isinstance(term.noise, NoiseModelWithAdditiveBiasCfg)
    assert term.delay_min_lag == IMU_DELAY_CONTROL_STEPS
    assert term.delay_max_lag == IMU_DELAY_CONTROL_STEPS
    assert term.noise.bias_noise_cfg is not None
    assert isinstance(term.noise.bias_noise_cfg, UniformNoiseCfg)
    assert term.noise.bias_noise_cfg.n_min == IMU_BIAS_RANGES[name][0]
    assert term.noise.bias_noise_cfg.n_max == IMU_BIAS_RANGES[name][1]
    assert isinstance(term.noise.noise_cfg, GaussianNoiseCfg)
    assert term.noise.noise_cfg.std == IMU_NOISE_STD[name]


def test_crawler_actuator_filter_has_ten_millisecond_time_constant() -> None:
  cfg = load_env_cfg(TASK_ID)
  action = cfg.actions["joint_pos"]
  assert isinstance(action, LowPassJointPositionActionCfg)
  assert action.new_target_weight == pytest.approx(ACTION_TARGET_NEW_WEIGHT)
  assert ACTION_TARGET_NEW_WEIGHT == pytest.approx(
    1.0 - math.exp(-(PHYSICS_TIMESTEP * 10) / ACTUATOR_RESPONSE_TIME_S)
  )


def test_flat_teacher_student_tasks_share_foundation_training() -> None:
  teacher_cfg = load_env_cfg(FLAT_TEACHER_TASK_ID)
  student_cfg = load_env_cfg(FLAT_STUDENT_TASK_ID)
  foundation_cfg = load_env_cfg(TASK_ID)
  no_imu_cfg = load_env_cfg(NO_IMU_ACTOR_TASK_ID)

  assert teacher_cfg.rewards == foundation_cfg.rewards
  assert teacher_cfg.curriculum == foundation_cfg.curriculum
  assert student_cfg.rewards == no_imu_cfg.rewards
  assert student_cfg.curriculum == no_imu_cfg.curriculum
  assert teacher_cfg.scene.terrain is not None
  assert teacher_cfg.scene.terrain.terrain_type == "plane"
  assert student_cfg.scene.terrain is not None
  assert student_cfg.scene.terrain.terrain_type == "plane"
  assert teacher_cfg.scene.num_envs == student_cfg.scene.num_envs == 8192
  assert tuple(teacher_cfg.observations["actor"].terms) == ("privileged_state",)
  assert teacher_cfg.observations["actor"].history_length == 10
  assert teacher_cfg.observations["critic"].history_length == 10
  assert tuple(student_cfg.observations["critic"].terms) == ("privileged_state",)
  assert student_cfg.observations["critic"].history_length == 10
  assert student_cfg.observations["actor"].history_length == 5
  assert tuple(student_cfg.observations["actor"].terms) == (
    "joint_pos",
    "joint_vel",
    "imu_lin_acc",
    "imu_ang_vel",
    "last_action",
    "velocity_command",
  )
  assert "gravity_vector" not in student_cfg.observations["actor"].terms


def test_teacher_student_runner_uses_separate_observation_sets() -> None:
  cfg = load_rl_cfg(STUDENT_TASK_ID)
  assert isinstance(cfg, CrawlerDistillationRunnerCfg)
  assert cfg.obs_groups == {"student": ("actor",), "teacher": ("critic",)}
  assert cfg.algorithm.class_name == "Distillation"
  assert cfg.algorithm.gradient_length == cfg.num_steps_per_env
  assert cfg.student.distribution_cfg == {
    "class_name": "GaussianDistribution",
    "init_std": 1.0e-6,
    "std_type": "scalar",
    "std_range": (1.0e-6, 1.0e-6),
    "learn_std": False,
  }
  assert cfg.student.hidden_dims == (256, 128, 64)
  assert cfg.teacher.hidden_dims == (256, 128, 64)
  assert cfg.experiment_name == "crawler_3dof_rough_student"
  assert cfg.teacher_experiment_name == CRAWLER_ROUGH_TEACHER_EXPERIMENT
  teacher_cfg = load_rl_cfg(TEACHER_TASK_ID)
  assert teacher_cfg.experiment_name == "crawler_3dof_rough_teacher"
  flat_cfg = load_rl_cfg(FLAT_STUDENT_TASK_ID)
  assert isinstance(flat_cfg, CrawlerDistillationRunnerCfg)
  assert flat_cfg.experiment_name == "crawler_3dof_flat_student"
  assert flat_cfg.teacher_experiment_name == CRAWLER_FLAT_TEACHER_EXPERIMENT
  flat_teacher_cfg = load_rl_cfg(FLAT_TEACHER_TASK_ID)
  assert flat_teacher_cfg.experiment_name == "crawler_3dof_flat_teacher"
  assert (
    flat_cfg.max_iterations
    == flat_teacher_cfg.max_iterations
    == FLAT_CURRICULUM_MAX_ITERATIONS
  )


def test_crawler_student_auto_selects_latest_teacher_checkpoint(tmp_path) -> None:
  log_root = tmp_path / "logs" / "rsl_rl"
  student_run = log_root / "crawler_3dof_flat_student" / "student-run"
  teacher_old = log_root / CRAWLER_FLAT_TEACHER_EXPERIMENT / "2026-01-01"
  teacher_latest = log_root / CRAWLER_FLAT_TEACHER_EXPERIMENT / "2026-01-02"
  for checkpoint in (
    teacher_old / "model_100.pt",
    teacher_latest / "model_100.pt",
    teacher_latest / "model_200.pt",
  ):
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()

  assert _get_latest_teacher_checkpoint(
    str(student_run), CRAWLER_FLAT_TEACHER_EXPERIMENT
  ) == (teacher_latest / "model_200.pt")


def test_crawler_student_requires_teacher_checkpoint(tmp_path) -> None:
  student_run = tmp_path / "logs" / "rsl_rl" / "student" / "run"
  with pytest.raises(ValueError, match="Train the matching teacher first"):
    _get_latest_teacher_checkpoint(str(student_run), CRAWLER_ROUGH_TEACHER_EXPERIMENT)


def test_crawler_student_resume_preserves_selected_teacher(monkeypatch) -> None:
  captured_load_cfg = None

  def fake_load(
    _runner,
    _path,
    load_cfg=None,
    strict=True,
    map_location=None,
  ):
    nonlocal captured_load_cfg
    captured_load_cfg = load_cfg
    return {}

  monkeypatch.setattr(MjlabOnPolicyRunner, "load", fake_load)
  runner = object.__new__(CrawlerDistillationRunner)
  runner.load("student_checkpoint.pt")

  assert captured_load_cfg == {
    "student": True,
    "optimizer": True,
    "iteration": True,
  }


def test_domain_randomization_is_training_only_and_centrally_tuned() -> None:
  cfg = load_env_cfg(TASK_ID)
  expected_events = {
    "randomize_sliding_friction",
    "randomize_torsional_friction",
    "randomize_rolling_friction",
    "randomize_inertial_properties",
    "randomize_joint_damping",
    "randomize_joint_armature",
    "randomize_motor_gains",
    "randomize_motor_effort",
    "randomize_imu_mount",
    "randomize_encoder_bias",
  }
  assert expected_events <= set(cfg.events)
  assert DOMAIN_RANDOMIZATION
  assert OBSERVATION_NOISE

  play_cfg = load_env_cfg(TASK_ID, play=True)
  assert expected_events.isdisjoint(play_cfg.events)
  assert not play_cfg.observations["actor"].enable_corruption
  assert not play_cfg.observations["actor"].terms["joint_pos"].params["biased"]


def test_reset_clearance_runs_after_root_and_joint_randomization() -> None:
  cfg = load_env_cfg(TASK_ID)
  event_names = tuple(cfg.events)
  clearance_index = event_names.index("lift_spawn_above_ground")
  assert event_names.index("reset_crawler_state") < clearance_index
  reset_cfg = cfg.events["reset_crawler_state"]
  assert reset_cfg.params["hard_joint_position_range"] == pytest.approx(
    (-ACTION_LIMIT, ACTION_LIMIT)
  )
  assert reset_cfg.func.__name__ == "reset_crawler_state_curriculum"


def test_training_configuration() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert cfg.scene.num_envs == 8192

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  assert action.scale == pytest.approx(ACTION_SCALE)
  assert action.clip is not None
  for name in JOINT_NAMES:
    assert action.clip[name] == pytest.approx((-ACTION_LIMIT, ACTION_LIMIT))

  command = cfg.commands["planar_velocity"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.linear_velocity_frame == "heading"
  assert command.linear_velocity_sampling == "ellipsoid"
  initial_speed = FOUNDATION_CURRICULUM_STAGES[0]["max_command_speed"]
  assert command.ranges.lin_vel_x == pytest.approx((-initial_speed, initial_speed))
  assert command.ranges.lin_vel_y == pytest.approx((-initial_speed, initial_speed))
  assert command.ranges.ang_vel_z == pytest.approx((0.0, 0.0))
  assert "training_stage" in cfg.curriculum


def test_robust_task_uses_rough_terrain_and_full_speed() -> None:
  foundation_cfg = load_env_cfg(TASK_ID)
  robust_cfg = load_env_cfg(ROBUST_TASK_ID)
  no_imu_cfg = load_env_cfg(ROBUST_NO_IMU_ACTOR_TASK_ID)
  assert foundation_cfg.scene.terrain is not None
  assert foundation_cfg.scene.terrain.terrain_type == "plane"
  assert foundation_cfg.scene.num_envs == 8192
  assert robust_cfg.scene.terrain is not None
  assert robust_cfg.scene.terrain.terrain_type == "generator"
  assert robust_cfg.scene.num_envs == 8192
  assert no_imu_cfg.scene.num_envs == 8192
  assert robust_cfg.scene.terrain.terrain_generator is not None
  assert tuple(robust_cfg.scene.terrain.terrain_generator.sub_terrains) == (
    "random_rough",
  )

  command = robust_cfg.commands["planar_velocity"]
  assert isinstance(command, UniformVelocityCommandCfg)
  assert command.ranges.lin_vel_x == pytest.approx((-1.5, 1.5))
  assert command.ranges.lin_vel_y == pytest.approx((-1.5, 1.5))
  assert robust_cfg.observations["actor"].terms["velocity_command"].clip == (
    -1.5,
    1.5,
  )


def test_network_architecture() -> None:
  cfg = load_rl_cfg(TASK_ID)
  assert isinstance(cfg, RslRlOnPolicyRunnerCfg)
  assert cfg.actor.hidden_dims == (256, 128, 64)
  assert cfg.critic.hidden_dims == (256, 128, 64)


def test_rewards_use_centralized_weights() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert set(cfg.rewards) == set(REWARD_WEIGHTS)
  for name, expected_weight in REWARD_WEIGHTS.items():
    if name == "perpendicular_drift":
      expected_weight = FOUNDATION_CURRICULUM_STAGES[0]["perpendicular_drift_weight"]
    assert cfg.rewards[name].weight == pytest.approx(expected_weight)
  no_motion_cfg = cfg.rewards["commanded_no_motion"]
  assert no_motion_cfg.params["command_threshold"] == pytest.approx(0.05)
  assert no_motion_cfg.weight < 0.0


def test_environment_reset(env: ManagerBasedRlEnv) -> None:
  observations, _ = env.reset(seed=123)
  assert torch.isfinite(_tensor_observation(observations, "actor")).all()
  assert torch.isfinite(_tensor_observation(observations, "critic")).all()


def test_action_dimension_is_three(env: ManagerBasedRlEnv) -> None:
  assert env.action_manager.total_action_dim == 3


def test_observation_shapes(env: ManagerBasedRlEnv) -> None:
  observations, _ = env.reset(seed=123)
  assert _tensor_observation(observations, "actor").shape == (2, 20)
  assert _tensor_observation(observations, "critic").shape == (2, 24)


def test_planar_commands_are_bounded_to_one_meter_per_second(
  env: ManagerBasedRlEnv,
) -> None:
  env.reset(seed=123)
  command = env.command_manager.get_command("planar_velocity")
  assert command is not None
  assert torch.linalg.vector_norm(command[:, :2], dim=1).max() <= 1.0 + 1.0e-6
  torch.testing.assert_close(command[:, 2], torch.zeros_like(command[:, 2]))


def test_short_rollout_has_no_nan(env: ManagerBasedRlEnv) -> None:
  observations, _ = env.reset(seed=123)
  assert torch.isfinite(_tensor_observation(observations, "actor")).all()
  for step in range(50):
    if step < 25:
      actions = torch.zeros((2, 3), device=env.device)
    else:
      actions = 0.25 * (2.0 * torch.rand((2, 3), device=env.device) - 1.0)
    observations, reward, terminated, truncated, _ = env.step(actions)
    assert torch.isfinite(_tensor_observation(observations, "actor")).all()
    assert torch.isfinite(_tensor_observation(observations, "critic")).all()
    assert torch.isfinite(reward).all()
    assert not terminated.any()
    assert not truncated.any()


def test_control_rates() -> None:
  cfg = load_env_cfg(TASK_ID)
  assert math.isclose(cfg.sim.mujoco.timestep, 0.002)
  assert cfg.decimation == 10
  assert math.isclose(1.0 / cfg.sim.mujoco.timestep, 500.0)
  assert math.isclose(
    1.0 / (cfg.sim.mujoco.timestep * cfg.decimation),
    50.0,
  )
