![Project banner](https://raw.githubusercontent.com/mujocolab/mjlab/main/docs/source/_static/mjlab-banner.jpg)

# mjlab

[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/mujocolab/mjlab/ci.yml?branch=main)](https://github.com/mujocolab/mjlab/actions/workflows/ci.yml?query=branch%3Amain)
[![Documentation](https://github.com/mujocolab/mjlab/actions/workflows/docs.yml/badge.svg)](https://mujocolab.github.io/mjlab/)
[![License](https://img.shields.io/github/license/mujocolab/mjlab)](https://github.com/mujocolab/mjlab/blob/main/LICENSE)
[![MuJoCo Warp](https://img.shields.io/badge/MuJoCo_Warp-3.11.0-blue)](https://github.com/google-deepmind/mujoco_warp/releases/tag/v3.11.0)
[![Nightly Benchmarks](https://img.shields.io/badge/Nightly-Benchmarks-blue)](https://mujocolab.github.io/mjlab/nightly/)
[![PyPI](https://img.shields.io/pypi/v/mjlab)](https://pypi.org/project/mjlab/)
[![PyPI downloads](https://img.shields.io/pypi/dm/mjlab?color=blue)](https://pypistats.org/packages/mjlab)

mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s manager-based API with [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), a GPU-accelerated version of [MuJoCo](https://github.com/google-deepmind/mujoco).
The framework provides composable building blocks for environment design,
with minimal dependencies and direct access to native MuJoCo data structures.

## Getting Started

mjlab requires an NVIDIA GPU for training. macOS is supported for evaluation only.

**Try it now:**

Run the demo (no installation needed):

```bash
uvx --from mjlab --refresh demo
```

Or try in [Google Colab](https://colab.research.google.com/github/mujocolab/mjlab/blob/main/notebooks/demo.ipynb) (no local setup required).

**Install from source:**

```bash
git clone https://github.com/mujocolab/mjlab.git && cd mjlab
uv run demo
```

For alternative installation methods (PyPI, Docker), see the [Installation Guide](https://mujocolab.github.io/mjlab/main/source/installation.html).

## Project environments

This workspace contains three first-class robot environments. They share the
same mjlab framework and are kept in the main source tree so each one is easy
to find, test, and modify:

| Environment | Purpose | Status |
| --- | --- | --- |
| Bird | Flapping-flight and velocity-control experiments | Migrated |
| Crawler | Crawling locomotion on flat and rough terrain | Migrated |
| Quad Mini | Joystick velocity control on flat terrain | Migrated |

The environment-specific organization is documented in the
[project structure guide](https://mujocolab.github.io/mjlab/main/source/project_structure.html).

### Task IDs

The main task for each environment is:

| Environment | Task ID |
| --- | --- |
| Bird | `Mjlab-Velocity-Bird-5DoF` |
| Crawler | `Mjlab-Crawl-Flat-ThreeDofCrawler` |
| Quad Mini | `Mjlab-QuadMiniTuned-Joystick-FlatTerrain` |
| Quad Mini teacher | `Mjlab-QuadMiniTuned-Joystick-FlatTerrain-Teacher` |
| Quad Mini student | `Mjlab-QuadMiniTuned-Joystick-FlatTerrain-Student` |

Crawler also includes robust, rough-terrain, teacher, and student variants;
see `src/mjlab/tasks/crawler/__init__.py` for their task IDs.

### Check that the environments work

Run the focused tests for all three custom environments:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_bird_velocity_task.py \
  tests/test_crawler_environment.py \
  tests/test_quad_mini_tuned.py
```

Run the full formatting, lint, and type checks:

```bash
make check
```

To open each environment in the viewer without a trained checkpoint, use the
zero-action agent. Stop the viewer with `Ctrl+C` when you are finished:

```bash
uv run play Mjlab-Velocity-Bird-5DoF --agent zero
uv run play Mjlab-Crawl-Flat-ThreeDofCrawler --agent zero
uv run play Mjlab-QuadMiniTuned-Joystick-FlatTerrain --agent zero
uv run play Mjlab-Tony5-Velocity-v0 --agent zero
uv run play Mjlab-Tony5-Velocity-Aero-v1 --agent zero
```

For TONY5 velocity play, omit the agent flag to use one viewer environment and
automatically load the newest local ``tony5_velocity_v0_history5`` checkpoint.
The policy observes five flattened frames, so retrain after this observation
change before loading a new trained checkpoint:

```bash
uv run play Mjlab-Tony5-Velocity-v0
```

The aerodynamic V1 task has the same policy interface and uses its own latest
checkpoint namespace:

```bash
uv run play Mjlab-Tony5-Velocity-Aero-v1
```

To inspect V1 using only high-speed moving commands, sample a radial speed from
15 to 27.78 m/s in every direction:

```bash
uv run play Mjlab-Tony5-Velocity-Aero-v1 --high-speed-only True
```

The band can be overridden with ``--high-speed-min`` and ``--high-speed-max``.
This play-only option does not change training command sampling.

For manual V1 velocity control, add ``--keyboard True``. In the native viewer,
use ``W/S`` for forward/backward, ``A/D`` for lateral motion, ``Q/E`` for yaw,
``R/F`` for vertical motion, and ``X`` for hover. The ``+`` and ``-`` keys
increase or decrease the horizontal command-speed ceiling by 1 m/s, clamped to
27.78 m/s. The Viser viewer provides equivalent on-screen buttons.

```bash
uv run play Mjlab-Tony5-Velocity-Aero-v1 --keyboard True
```

For analog gamepad control, use ``--gamepad True`` with the native viewer. The
left stick controls forward/lateral velocity, the right stick horizontal axis
controls yaw, and the left/right triggers control down/up velocity. The left
and right bumpers lower/raise the horizontal speed ceiling by 1 m/s.

```bash
uv run play Mjlab-Tony5-Velocity-Aero-Omni-v0 --gamepad True
```

The V1 training curriculum uses 2 m/s for the first 50 PPO iterations, 6 m/s
for the next 50, 10 m/s for the next 100, and 27.78 m/s thereafter.
During fast commands, yaw-rate and vertical command limits taper down to
approximately 0.10 rad/s and 0.10 m/s at 27.78 m/s.
V1 command directions remain omnidirectional through 10 m/s and progressively
limit to ±90°, ±45°, ±25°, and ±10° as speed increases toward 27.78 m/s.

### Train the environments

Training requires an NVIDIA GPU. Start a run for each main environment with:

```bash
uv run train Mjlab-Velocity-Bird-5DoF --env.scene.num-envs 4096
uv run train Mjlab-Crawl-Flat-ThreeDofCrawler --env.scene.num-envs 4096
uv run train Mjlab-QuadMiniTuned-Joystick-FlatTerrain --env.scene.num-envs 4096
uv run train Mjlab-Tony5-Velocity-v0 --env.scene.num-envs 4096
uv run train Mjlab-Tony5-Velocity-Aero-v1 --env.scene.num-envs 4096
```

The TONY5 velocity policy uses five flattened observation frames, giving the
actor and critic 110 input values each.

The Quad Mini teacher receives clean privileged observations with five frames
of history. The student uses the deployable observation with five frames of
history and is trained by distillation from a teacher checkpoint:

```bash
# 1. Train the clean privileged teacher.
uv run train Mjlab-QuadMiniTuned-Joystick-FlatTerrain-Teacher \
  --env.scene.num-envs 4096

# 2. Train the five-frame student. The newest teacher run/checkpoint is loaded
#    automatically from logs/rsl_rl/quad_mini_tuned_teacher/.
uv run train Mjlab-QuadMiniTuned-Joystick-FlatTerrain-Student \
  --env.scene.num-envs 4096
```

To override automatic selection, pass an explicit checkpoint with
`--agent.teacher-checkpoint /path/to/model.pt`.

The teacher policy uses `5 x 135 = 675` observation values. The student policy
uses `5 x 60 = 300` values, while its teacher input uses the same five-frame,
675-value privileged observation. Both tasks share the same robot, terrain,
commands, rewards, and domain randomizers.

For Crawler teacher/student training, use the rough-terrain or flat-terrain
task pair:

```bash
# Rough-terrain teacher and student.
uv run train Mjlab-Crawl-Rough-ThreeDofCrawler-Teacher --env.scene.num-envs 4096
uv run train Mjlab-Crawl-Rough-ThreeDofCrawler-Student --env.scene.num-envs 4096

# Flat-terrain teacher and student.
uv run train Mjlab-Crawl-Flat-ThreeDofCrawler-Teacher --env.scene.num-envs 4096
uv run train Mjlab-Crawl-Flat-ThreeDofCrawler-Student --env.scene.num-envs 4096
```

Play the newest local teacher or student checkpoint automatically with:

```bash
uv run play Mjlab-Crawl-Rough-ThreeDofCrawler-Teacher
uv run play Mjlab-Crawl-Rough-ThreeDofCrawler-Student
uv run play Mjlab-Crawl-Flat-ThreeDofCrawler-Teacher
uv run play Mjlab-Crawl-Flat-ThreeDofCrawler-Student
uv run play Mjlab-QuadMiniTuned-Joystick-FlatTerrain-Teacher
uv run play Mjlab-QuadMiniTuned-Joystick-FlatTerrain-Student
```

To train a variant, replace the task ID with one from the task registration
files. After training, `play` automatically loads the newest local checkpoint
for that task. You can also select a checkpoint explicitly:

```bash
uv run play Mjlab-QuadMiniTuned-Joystick-FlatTerrain \
  --checkpoint-file path/to/model_1000.pt
```

### Push changes to GitHub

This workspace is connected to
`git@github.com:kaanborucu/mjlab-custom-environments.git`. After making and
checking changes, push them with:

```bash
git status
git add .
git commit -m "Describe your change"
git push origin main
```

## Training Examples

### 1. Velocity Tracking

Train a Unitree G1 humanoid to follow velocity commands on flat terrain:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
```

**Multi-GPU Training:** Scale to multiple GPUs using `--gpu-ids`:

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 \
  --gpu-ids "[0, 1]" \
  --env.scene.num-envs 4096
```

See the [Distributed Training guide](https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html) for details.

Evaluate a policy while training (fetches latest checkpoint from Weights & Biases):

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 2. Motion Imitation

Train a humanoid to mimic reference motions. See the [motion imitation guide](https://mujocolab.github.io/mjlab/main/source/training/motion_imitation.html) for preprocessing setup.

```bash
uv run train Mjlab-Tracking-Flat-Unitree-G1 --registry-name your-org/motions/motion-name --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-Unitree-G1 --wandb-run-path your-org/mjlab/run-id
```

### 3. Sanity-check with Dummy Agents

Use built-in agents to sanity check your MDP before training:

```bash
uv run play Mjlab-Your-Task-Id --agent zero  # Sends zero actions
uv run play Mjlab-Your-Task-Id --agent random  # Sends uniform random actions
```

When running motion-tracking tasks, add `--registry-name your-org/motions/motion-name` to the command.


## Documentation

Full documentation is available at **[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/)**.

## Development

```bash
make test          # Run all tests
make test-fast     # Skip slow tests
make format        # Format and lint
make docs          # Build docs locally
```

For development setup: `uvx pre-commit install`

## Citation

mjlab is used in published research and open-source robotics projects. See the [Research](https://mujocolab.github.io/mjlab/main/source/research.html) page for publications and projects, or share your own in [Show and Tell](https://github.com/mujocolab/mjlab/discussions/categories/show-and-tell).

If you use mjlab in your research, please consider citing:

```bibtex
@misc{zakka2026mjlablightweightframeworkgpuaccelerated,
  title={mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning},
  author={Kevin Zakka and Qiayuan Liao and Brent Yi and Louis Le Lay and Koushil Sreenath and Pieter Abbeel},
  year={2026},
  eprint={2601.22074},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2601.22074},
}
```

## License

mjlab is licensed under the [Apache License, Version 2.0](LICENSE).

### Third-Party Code

Some portions of mjlab are forked from external projects:

- **`src/mjlab/utils/lab_api/`** — Utilities forked from [NVIDIA Isaac
  Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause license, see file
  headers)

Forked components retain their original licenses. See file headers for details.

## Acknowledgments

mjlab wouldn't exist without the excellent work of the Isaac Lab team, whose API
design and abstractions mjlab builds upon.

Thanks to the MuJoCo Warp team — especially Erik Frey and Taylor Howell — for
answering our questions, giving helpful feedback, and implementing features
based on our requests countless times.
