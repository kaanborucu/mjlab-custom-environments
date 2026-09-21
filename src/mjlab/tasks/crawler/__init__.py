"""MJLab task registration for the three-DoF crawler."""

from mjlab.tasks.crawler.env_cfg import FLAT_CURRICULUM_MAX_ITERATIONS, crawler_env_cfg
from mjlab.tasks.crawler.rl_cfg import (
  CRAWLER_FLAT_TEACHER_EXPERIMENT,
  CRAWLER_ROUGH_TEACHER_EXPERIMENT,
  crawler_distillation_runner_cfg,
  crawler_ppo_runner_cfg,
)
from mjlab.tasks.crawler.teacher_student_env_cfg import (
  crawler_student_env_cfg,
  crawler_teacher_env_cfg,
)
from mjlab.tasks.crawler.teacher_student_runner import CrawlerDistillationRunner
from mjlab.tasks.registry import register_mjlab_task

TASK_ID = "Mjlab-Crawl-Flat-ThreeDofCrawler"
NO_IMU_ACTOR_TASK_ID = "Mjlab-Crawl-Flat-ThreeDofCrawler-NoImuActor"
ROBUST_TASK_ID = "Mjlab-Crawl-Flat-ThreeDofCrawler-Robust"
ROBUST_NO_IMU_ACTOR_TASK_ID = "Mjlab-Crawl-Flat-ThreeDofCrawler-Robust-NoImuActor"
TEACHER_TASK_ID = "Mjlab-Crawl-Rough-ThreeDofCrawler-Teacher"
STUDENT_TASK_ID = "Mjlab-Crawl-Rough-ThreeDofCrawler-Student"
FLAT_TEACHER_TASK_ID = "Mjlab-Crawl-Flat-ThreeDofCrawler-Teacher"
FLAT_STUDENT_TASK_ID = "Mjlab-Crawl-Flat-ThreeDofCrawler-Student"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=crawler_env_cfg(profile="flat"),
  play_env_cfg=crawler_env_cfg(play=True, profile="flat"),
  rl_cfg=crawler_ppo_runner_cfg(max_iterations=FLAT_CURRICULUM_MAX_ITERATIONS),
  runner_cls=None,
)

register_mjlab_task(
  task_id=NO_IMU_ACTOR_TASK_ID,
  env_cfg=crawler_env_cfg(
    profile="flat",
    include_imu_in_actor=False,
    actor_history_length=5,
  ),
  play_env_cfg=crawler_env_cfg(
    play=True,
    profile="flat",
    include_imu_in_actor=False,
    actor_history_length=5,
  ),
  rl_cfg=crawler_ppo_runner_cfg(max_iterations=FLAT_CURRICULUM_MAX_ITERATIONS),
  runner_cls=None,
)

register_mjlab_task(
  task_id=ROBUST_TASK_ID,
  env_cfg=crawler_env_cfg(profile="robust"),
  play_env_cfg=crawler_env_cfg(play=True, profile="robust"),
  rl_cfg=crawler_ppo_runner_cfg(max_iterations=400),
  runner_cls=None,
)

register_mjlab_task(
  task_id=ROBUST_NO_IMU_ACTOR_TASK_ID,
  env_cfg=crawler_env_cfg(
    profile="robust",
    include_imu_in_actor=False,
    actor_history_length=5,
  ),
  play_env_cfg=crawler_env_cfg(
    play=True,
    profile="robust",
    include_imu_in_actor=False,
    actor_history_length=5,
  ),
  rl_cfg=crawler_ppo_runner_cfg(max_iterations=400),
  runner_cls=None,
)

register_mjlab_task(
  task_id=TEACHER_TASK_ID,
  env_cfg=crawler_teacher_env_cfg(),
  play_env_cfg=crawler_teacher_env_cfg(play=True),
  rl_cfg=crawler_ppo_runner_cfg(
    max_iterations=400,
    experiment_name=CRAWLER_ROUGH_TEACHER_EXPERIMENT,
  ),
  runner_cls=None,
)

register_mjlab_task(
  task_id=STUDENT_TASK_ID,
  env_cfg=crawler_student_env_cfg(),
  play_env_cfg=crawler_student_env_cfg(play=True),
  rl_cfg=crawler_distillation_runner_cfg(max_iterations=400),
  runner_cls=CrawlerDistillationRunner,
)

register_mjlab_task(
  task_id=FLAT_TEACHER_TASK_ID,
  env_cfg=crawler_teacher_env_cfg(profile="flat"),
  play_env_cfg=crawler_teacher_env_cfg(play=True, profile="flat"),
  rl_cfg=crawler_ppo_runner_cfg(
    max_iterations=FLAT_CURRICULUM_MAX_ITERATIONS,
    experiment_name=CRAWLER_FLAT_TEACHER_EXPERIMENT,
  ),
  runner_cls=None,
)

register_mjlab_task(
  task_id=FLAT_STUDENT_TASK_ID,
  env_cfg=crawler_student_env_cfg(profile="flat"),
  play_env_cfg=crawler_student_env_cfg(play=True, profile="flat"),
  rl_cfg=crawler_distillation_runner_cfg(
    max_iterations=FLAT_CURRICULUM_MAX_ITERATIONS,
    experiment_name="crawler_3dof_flat_student",
    teacher_experiment_name=CRAWLER_FLAT_TEACHER_EXPERIMENT,
  ),
  runner_cls=CrawlerDistillationRunner,
)

__all__ = [
  "NO_IMU_ACTOR_TASK_ID",
  "FLAT_STUDENT_TASK_ID",
  "FLAT_TEACHER_TASK_ID",
  "ROBUST_NO_IMU_ACTOR_TASK_ID",
  "ROBUST_TASK_ID",
  "STUDENT_TASK_ID",
  "TEACHER_TASK_ID",
  "TASK_ID",
]
