from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import bird_5dof_forward_3d_env_cfg
from .rl_cfg import bird_5dof_forward_3d_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Bird-Forward-3D",
  env_cfg=bird_5dof_forward_3d_env_cfg(),
  play_env_cfg=bird_5dof_forward_3d_env_cfg(play=True),
  rl_cfg=bird_5dof_forward_3d_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
