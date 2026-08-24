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

This workspace contains four first-class robot environments. They share the
same mjlab framework and are kept in the main source tree so each one is easy
to find, test, and modify:

| Environment | Purpose | Status |
| --- | --- | --- |
| Bird | Flapping-flight and velocity-control experiments | Migrated |
| Crawler | Crawling locomotion on flat and rough terrain | Migrated |
| Quad Mini | Joystick velocity control on flat terrain | Migrated |
| TONY5 | 5-inch quadrotor and aerodynamic-flight experiments | Active |

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
| TONY5 position/yaw | `Mjlab-Tony5-PositionYaw-v0` |
| TONY5 velocity | `Mjlab-Tony5-Velocity-v0` |
| TONY5 aerodynamic velocity | `Mjlab-Tony5-Velocity-Aero-v1` |
| TONY5 aerodynamic Omni V0 | `Mjlab-Tony5-Velocity-Aero-Omni-v0` |
| TONY5 aerodynamic Omni V3 | `Mjlab-Tony5-Velocity-Aero-Omni-v3` |

Crawler also includes robust, rough-terrain, teacher, and student variants;
see `src/mjlab/tasks/crawler/__init__.py` for their task IDs.

### TONY5 environment guide

TONY5 is a 5-inch quadrotor model with four individually controlled rotors.
The physics runs at 1 kHz (`dt = 0.001 s`) and the policy runs at 100 Hz
(`dt = 0.01 s`, ten physics steps per policy action). The default reset is a
stationary vehicle at approximately 1.5 m with rotor speed initialized to
836 rad/s, the nominal hover speed.

| Task | Command frame | Command range | Observation |
| --- | --- | --- | --- |
| `Velocity-v0` | Body `vx`, body `vy`, body yaw rate, world `vz` | Horizontal box ±2 m/s; `vz` −0.75..0.75 m/s | 5-frame history, 110 values |
| `Velocity-Aero-v1` | Body `vx`, body `vy`, body yaw rate, world `vz` | Radial horizontal curriculum to 27.78 m/s; `vz` −3..3 m/s | 5-frame history, 110 values |
| `Aero-Omni-v0` | Body `vx`, body `vy`, body yaw rate, world `vz` | Radial horizontal speed ≤10 m/s; `vz` −3..3 m/s | 5-frame history, 110 values |
| `Aero-Omni-v3` | World `vx`, world `vy`, body yaw rate, world `vz` | Radial horizontal speed ≤10 m/s; `vz` −3..3 m/s | 5-frame history, 120 values |

The V0 ground plane remains visible but is non-colliding for the velocity
tasks. V1 and both aerodynamic Omni tasks use the same four normalized rotor
actions. Each action is mapped to a desired rotor speed from 0 to 2600 rad/s:
`a = 0` corresponds to 836 rad/s, the negative half maps from 0 to hover, and
the positive half maps from hover to the 2600 rad/s ceiling.

V1 uses the native MuJoCo DC motor model and a 1 kHz filtered PID speed loop.
Its ESC voltage is one-directional, `0..22.2 V`, with asymmetric anti-windup.
The original V0 motor action retains its original bidirectional voltage path.
The estimated rotor model uses `kT = 1.0e-6` and `kQ = 1.25e-8`. V1 adds a
MuJoCo ellipsoid body-fluid load with density `1.225 kg/m³`, size
`(0.065, 0.050, 0.022) m`, position `(0, 0, 0.015) m`, and fluid coefficients
`(0.5, 0.25, 1.0, 0, 0)`. Rotor H-force uses
`ROTOR_DRAG_KH = 1.0e-5`; this is estimated and still marked TODO CALIBRATE.

V1 command curriculum and direction limits are:

| PPO iterations | Maximum speed |
| --- | ---: |
| 0–49 | 2 m/s |
| 50–99 | 6 m/s |
| 100–199 | 10 m/s |
| 200 onward | 27.78 m/s |

Moving commands use 50% previous-stage speed, 30% current-stage speed, and
20% high-speed sampling; 20% of all commands are exact hover commands. The
high-speed branch samples from 80% to 100% of the current maximum. Direction is
fully omnidirectional through 10 m/s, then limited to ±90°, ±45°, ±25°, and
±10° in the 10–15, 15–20, 20–25, and 25–27.78 m/s bands. V1 uses a 512-control-
step rolling curriculum window and ignores the first 0.75 seconds of a new
command when computing its settled high-speed RMSE.

Omni V0 has no curriculum and smoothly transitions commands over approximately
0.333 seconds. Omni V3 uses world-frame linear velocity commands, no command
smoothing, and resamples commands every 1–10 seconds. Its actor and critic
observe world linear velocity, explicit world-frame commands, and sine/cosine
heading, which increases the flattened history from 110 to 120 values.

Omni V3 enables the optional physics extensions currently available in this
workspace: static CT/CQ fallback derived from `kT`/`kQ`, deterministic blade
flapping with `K_FLAP = 1.0e-7`, algebraic battery sag with `V_oc = 22.2 V` and
`R_battery = 0.02 ohm`, randomized shared background wind, and a shared
Ornstein–Uhlenbeck gust process. Background wind samples 0–10 m/s horizontally
in any direction and −1..1 m/s vertically; gusts use `sigma = 1`, `tau = 1 s`,
and a 5 m/s speed cap. During training, wind/gust activation is selected on
50% of full resets. In play, `--wind` or `--gusts` explicitly enables 100%
activation for the selected effects.

The complete editable reward tables are kept separately:

| Task | Reward configuration |
| --- | --- |
| V0 velocity | `src/mjlab/tasks/manager_based/tony5/tony5_velocity_env_cfg.py` |
| Aero V1 | `src/mjlab/tasks/manager_based/tony5/tony5_aero_env_cfg.py` |
| Aero Omni V0 | `src/mjlab/tasks/manager_based/tony5/tony5_omni_rewards.py` |
| Aero Omni V3 | `src/mjlab/tasks/manager_based/tony5/tony5_omni_v3_rewards.py` |

The current principal reward weights are linear velocity tracking `5.0`,
angular velocity tracking `3.0` for Omni tasks, action-rate penalty `-0.04`,
crash penalty `-100`, and V3 motor-torque penalty `-1e-6`. Aero V1 uses
linear tracking `5.0`, angular tracking `2.0`, action-rate `-0.01`, crash `-10`,
and measured downward-velocity penalty `-0.1`. Zero-weight terms remain in the
editable tables so they can be enabled without changing another task.

### Check that the environments work

Run the focused tests for the custom environments:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q \
  tests/test_bird_velocity_task.py \
  tests/test_crawler_environment.py \
  tests/test_quad_mini_tuned.py \
  tests/test_tony5.py \
  tests/test_tony5_aero.py \
  tests/test_tony5_omni_v3.py \
  tests/test_tony5_velocity_curriculum.py \
  tests/test_play.py
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
uv run play Mjlab-Tony5-Velocity-Aero-Omni-v0 --agent zero
uv run play Mjlab-Tony5-Velocity-Aero-Omni-v3 --agent zero
```

For trained TONY5 play, omit the agent flag to use the one-environment play
configuration and automatically load the newest local checkpoint for that
task's experiment directory. V3 deliberately skips the incompatible
`v0_bootstrap` run when selecting a checkpoint:

```bash
uv run play Mjlab-Tony5-Velocity-v0
uv run play Mjlab-Tony5-Velocity-Aero-v1
uv run play Mjlab-Tony5-Velocity-Aero-Omni-v0
uv run play Mjlab-Tony5-Velocity-Aero-Omni-v3
```

To inspect V1 using only high-speed moving commands, sample a radial speed from
15 to 27.78 m/s:

```bash
uv run play Mjlab-Tony5-Velocity-Aero-v1 --high-speed-only True
```

The band can be overridden with `--high-speed-min` and `--high-speed-max`.
This play-only option does not change training command sampling.

For manual TONY5 velocity control, add `--keyboard True`. In the native viewer,
use `W/S` for forward/backward, `A/D` for lateral motion, `Q/E` for yaw,
`R/F` for vertical motion, and `X` for hover. With keyboard control enabled,
main or keypad `+/-` changes the manual horizontal command-speed ceiling by
1 m/s, up to 100 m/s. In V3, the manual horizontal axes are world X/Y; in V0
and V1 they are body-frame X/Y. The Viser viewer provides equivalent buttons.

`M` resamples the velocity command immediately. For Omni V3 it also resets the
shared background-wind and gust process. The Viser viewer has the same
`M: Resample velocity + wind` button. The terminal prints a confirmation when
the resample action is processed.

```bash
uv run play Mjlab-Tony5-Velocity-Aero-v1 --keyboard True
uv run play Mjlab-Tony5-Velocity-Aero-Omni-v3 --viewer native --keyboard True
```

For analog gamepad control, use `--gamepad True` with the native viewer. The
left stick controls horizontal velocity, the right-stick horizontal axis
controls yaw, and the left/right triggers control down/up velocity. The left
and right bumpers lower/raise the horizontal speed ceiling by 1 m/s.

```bash
uv run play Mjlab-Tony5-Velocity-Aero-Omni-v0 --gamepad True
```

To enable V3 wind and gusts explicitly, use the following. They are already
enabled by default in V3 play:

```bash
uv run play Mjlab-Tony5-Velocity-Aero-Omni-v3 \
  --viewer native \
  --wind True \
  --gusts True
```

Disable either effect independently:

```bash
uv run play Mjlab-Tony5-Velocity-Aero-Omni-v3 \
  --wind False \
  --gusts False
```

Add a play-only random base-torque push with a visible magenta arrow. The
default push is ±0.05 N·m, lasts 0.5 seconds, and is resampled every 3–6
seconds. The total wind is shown as a green arrow.

```bash
uv run play Mjlab-Tony5-Velocity-Aero-Omni-v3 \
  --disturbance True
```

Use `--no-terminations True` only for diagnostics or dummy-agent viewing.
Normal V1/V3 episodes include rapid-descent termination based on measured
downward velocity below −5 m/s and a numerical-safety termination for a
non-finite state, root speed above 150 m/s, body angular speed above 200 rad/s,
or absolute generalized acceleration above `4e5`.

### Train the environments

Training requires an NVIDIA GPU. Start a run for each main environment with:

```bash
uv run train Mjlab-Velocity-Bird-5DoF --env.scene.num-envs 4096
uv run train Mjlab-Crawl-Flat-ThreeDofCrawler --env.scene.num-envs 4096
uv run train Mjlab-QuadMiniTuned-Joystick-FlatTerrain --env.scene.num-envs 4096
uv run train Mjlab-Tony5-Velocity-v0 --env.scene.num-envs 4096
uv run train Mjlab-Tony5-Velocity-Aero-v1 --env.scene.num-envs 4096
uv run train Mjlab-Tony5-Velocity-Aero-Omni-v0 --env.scene.num-envs 4096
uv run train Mjlab-Tony5-Velocity-Aero-Omni-v3 --env.scene.num-envs 4096
```

The V0, V1, and Omni V0 policies use five flattened observation frames with
110 actor and critic inputs. Omni V3 uses 120 inputs. Do not resume a V3 run
from a 110-input V0 bootstrap checkpoint; use a timestamped V3 run instead.

Start a fresh V3 run with a custom iteration count:

```bash
uv run train Mjlab-Tony5-Velocity-Aero-Omni-v3 \
  --env.scene.num-envs 8192 \
  --agent.max-iterations 100000
```

Resume the latest compatible local V3 run:

```bash
uv run train Mjlab-Tony5-Velocity-Aero-Omni-v3 \
  --agent.resume True \
  --env.scene.num-envs 8192
```

The V3 PPO configuration uses the same main PPO settings as the TONY5 Aero
family, with `init_std = 0.6`, learnable standard deviation, no explicit
standard-deviation range, and `entropy_coef = 0.003`. V0 remains on its own
PPO configuration.

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
