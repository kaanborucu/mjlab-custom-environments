"""Script to play RL agent with RSL-RL."""

import os
import re
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import glfw
import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.manager_based.tony5.tony5_omni_v3_wind import (
  Tony5OmniV3GlobalWind,
  Tony5OmniV3WindCfg,
)
from mjlab.tasks.manager_based.tony5.tony5_velocity_curriculum import (
  TONY5_VELOCITY_KEYBOARD_SPEED_STEP,
  TONY5_VELOCITY_MAX_YAW_RATE,
  TONY5_VELOCITY_VERTICAL_COMMAND_MAX,
  Tony5RadialVelocityCommand,
  Tony5RadialVelocityCommandCfg,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


def _get_latest_local_checkpoint(
  log_root_path: Path,
  excluded_run_names: frozenset[str] = frozenset(),
) -> Path:
  """Find the newest local checkpoint, optionally excluding bootstrap runs."""
  run_dir = r".*"
  if excluded_run_names:
    excluded_pattern = "|".join(re.escape(name) for name in excluded_run_names)
    run_dir = rf"^(?!({excluded_pattern})$).*"
  try:
    return get_checkpoint_path(
      log_root_path,
      run_dir=run_dir,
      checkpoint=r"^model_\d+\.pt$",
    )
  except ValueError as exc:
    raise ValueError(
      f"No local checkpoint found under {log_root_path}. Train this task first, "
      "or provide `--checkpoint-file /path/to/model.pt` or "
      "`--wandb-run-path entity/project/run_id`."
    ) from exc


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  playback_speed: float = 1.2
  """Initial viewer playback speed; ``1.2`` means 20% faster than real time."""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  high_speed_only: bool = False
  """For TONY5 Aero V1, sample play commands only from the high-speed band."""
  high_speed_min: float = 15.0
  """Lower horizontal-speed bound for ``--high-speed-only`` play mode."""
  high_speed_max: float = 27.78
  """Upper horizontal-speed bound for ``--high-speed-only`` play mode."""
  keyboard: bool = False
  """Enable TONY5 V1 keyboard velocity control in play mode."""
  gamepad: bool = False
  """Enable native-viewer analog gamepad velocity control in play mode."""
  disturbance: bool = False
  """Apply a random base torque disturbance during play."""
  disturbance_torque: float = 0.05
  """Peak random torque in N*m per base axis when disturbance is enabled."""
  disturbance_interval_min: float = 3.0
  """Minimum seconds between disturbance resamples."""
  disturbance_interval_max: float = 6.0
  """Maximum seconds between disturbance resamples."""
  disturbance_duration: float = 0.5
  """Seconds that each sampled disturbance torque remains active."""
  wind: bool = True
  """Enable V3 background wind; use ``--wind False`` to disable it."""
  gusts: bool = True
  """Enable V3 stochastic gusts; use ``--gusts False`` to disable them."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


class Tony5KeyboardController:
  """Queue-driven keyboard and button control for TONY5 V1 velocity commands."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    resample_callback: Callable[[], None] | None = None,
  ):
    command = env.command_manager.get_term("velocity")
    if not isinstance(command, Tony5RadialVelocityCommand):
      raise ValueError(
        "Keyboard control requires the TONY5 Aero V1 radial velocity command."
      )
    self.command = command
    self.command.enable_keyboard_control()
    self._resample_callback = resample_callback
    self._status_html: Any | None = None

  def _refresh_status(self) -> None:
    if self._status_html is not None:
      self._status_html.content = (
        "<strong>Keyboard velocity control</strong><br/>"
        f"Mode: {self.command.keyboard_mode}<br/>"
        f"Horizontal frame: {self.command.cfg.linear_velocity_frame}<br/>"
        "Vertical frame: world<br/>"
        f"Yaw frame: {self.command.cfg.yaw_velocity_frame}<br/>"
        f"Max horizontal speed: {self.command.keyboard_max_speed:.2f} m/s"
      )

  def handle_action(self, action: ViewerAction, payload: object | None) -> bool:
    """Apply a queued keyboard or GUI action on the simulation thread."""
    if action != ViewerAction.CUSTOM or not isinstance(payload, dict):
      return False
    payload_dict = cast(dict[str, object], payload)
    if payload_dict.get("type") == "keyboard_mode":
      self.command.set_keyboard_mode(str(payload_dict["mode"]))
    elif payload_dict.get("type") == "keyboard_speed":
      self.command.adjust_keyboard_max_speed(float(cast(float, payload_dict["delta"])))
    elif payload_dict.get("type") == "keyboard_key":
      self._handle_key(int(cast(int, payload_dict["key"])))
    else:
      return False
    self._refresh_status()
    return True

  def _handle_key(self, key: int) -> None:
    """Translate native GLFW key codes into velocity command modes."""
    from mjlab.viewer.native.keys import (
      KEY_A,
      KEY_D,
      KEY_E,
      KEY_EQUAL,
      KEY_F,
      KEY_KP_ADD,
      KEY_KP_SUBTRACT,
      KEY_M,
      KEY_MINUS,
      KEY_Q,
      KEY_R,
      KEY_S,
      KEY_W,
      KEY_X,
    )

    modes = {
      KEY_W: "forward",
      KEY_S: "backward",
      KEY_A: "left",
      KEY_D: "right",
      KEY_Q: "yaw_left",
      KEY_E: "yaw_right",
      KEY_R: "up",
      KEY_F: "down",
      KEY_X: "hover",
    }
    # When keyboard control is enabled, the native viewer bypasses its own
    # +/- playback shortcuts. KEY_EQUAL is the GLFW code for both '=' and '+'.
    if key in (KEY_EQUAL, KEY_KP_ADD):
      self.command.adjust_keyboard_max_speed(TONY5_VELOCITY_KEYBOARD_SPEED_STEP)
    elif key in (KEY_MINUS, KEY_KP_SUBTRACT):
      self.command.adjust_keyboard_max_speed(-TONY5_VELOCITY_KEYBOARD_SPEED_STEP)
    elif key in (KEY_M, ord("m"), ord("M")) and self._resample_callback is not None:
      self._resample_callback()
    elif key in modes:
      self.command.set_keyboard_mode(modes[key])

  def create_viser_gui(
    self,
    server: Any,
    request_action: Callable[[str, Any], None],
  ) -> None:
    """Create button controls for keyboard-equivalent Viser input."""
    with server.gui.add_folder("Keyboard Velocity"):
      self._status_html = server.gui.add_html("")
      speed_buttons = server.gui.add_button_group("Max speed (m/s)", options=["-", "+"])

      @speed_buttons.on_click
      def _(event) -> None:
        delta = (
          TONY5_VELOCITY_KEYBOARD_SPEED_STEP
          if event.target.value == "+"
          else -TONY5_VELOCITY_KEYBOARD_SPEED_STEP
        )
        request_action("CUSTOM", {"type": "keyboard_speed", "delta": delta})

      direction_buttons = server.gui.add_button_group(
        "Direction", options=["Forward", "Backward", "Left", "Right", "Hover"]
      )

      @direction_buttons.on_click
      def _(event) -> None:
        mode = {
          "Forward": "forward",
          "Backward": "backward",
          "Left": "left",
          "Right": "right",
          "Hover": "hover",
        }[event.target.value]
        request_action("CUSTOM", {"type": "keyboard_mode", "mode": mode})

      vertical_buttons = server.gui.add_button_group("Vertical", options=["Up", "Down"])

      @vertical_buttons.on_click
      def _(event) -> None:
        mode = "up" if event.target.value == "Up" else "down"
        request_action("CUSTOM", {"type": "keyboard_mode", "mode": mode})

      yaw_buttons = server.gui.add_button_group(
        "Yaw", options=["Yaw left", "Yaw right"]
      )

      @yaw_buttons.on_click
      def _(event) -> None:
        mode = "yaw_left" if event.target.value == "Yaw left" else "yaw_right"
        request_action("CUSTOM", {"type": "keyboard_mode", "mode": mode})

    self._refresh_status()


class Tony5PlayResampler:
  """Resample play commands and optional V3 wind on demand."""

  def __init__(
    self,
    command: Tony5RadialVelocityCommand,
    wind: Tony5OmniV3GlobalWind | None = None,
  ) -> None:
    self.command = command
    self.wind = wind

  def resample(self) -> None:
    """Sample a new velocity command and reset the shared V3 wind process."""
    self.command.resample_now()
    if self.wind is not None:
      self.wind.reset()
    print("[INFO] Play resampled velocity command and wind/gust state")

  def handle_action(self, action: ViewerAction, payload: object | None) -> bool:
    """Handle the native-key and Viser-button resample request."""
    if action != ViewerAction.CUSTOM or not isinstance(payload, dict):
      return False
    payload_dict = cast(dict[str, object], payload)
    if payload_dict.get("type") == "resample_velocity_wind":
      self.resample()
      return True
    return False

  def create_viser_gui(
    self,
    server: Any,
    request_action: Callable[[str, Any], None],
  ) -> None:
    """Create the Viser button for immediate resampling."""
    with server.gui.add_folder("Play Controls"):
      resample_button = server.gui.add_button("M: Resample velocity + wind")

      @resample_button.on_click
      def _(_) -> None:
        request_action("CUSTOM", {"type": "resample_velocity_wind"})


def _apply_gamepad_deadzone(value: float, deadzone: float = 0.12) -> float:
  """Remove stick drift while preserving the full useful input range."""
  magnitude = abs(value)
  if magnitude <= deadzone:
    return 0.0
  scaled = (magnitude - deadzone) / (1.0 - deadzone)
  return scaled if value >= 0.0 else -scaled


class Tony5GamepadController:
  """Poll a GLFW gamepad and translate it into TONY5 velocity commands."""

  def __init__(self, env: ManagerBasedRlEnv):
    command = env.command_manager.get_term("velocity")
    if not isinstance(command, Tony5RadialVelocityCommand):
      raise ValueError("Gamepad control requires the TONY5 Aero velocity command.")
    self.command = command
    self.command.enable_gamepad_control()
    self._joystick_id: int | None = None
    self._last_buttons: tuple[int, ...] = ()
    self._reported_missing = False
    self._reported_name: str | None = None

  def _find_joystick(self) -> tuple[int, bool] | None:
    """Find a mapped gamepad, or a raw joystick with enough axes."""
    raw_candidate: int | None = None
    for joystick_id in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
      if not glfw.joystick_present(joystick_id):
        continue
      if glfw.joystick_is_gamepad(joystick_id):  # ty: ignore[possibly-missing-attribute]
        return joystick_id, True
      name = glfw.get_joystick_name(joystick_id).decode(errors="replace").lower()
      if "mouse" not in name and "keyboard" not in name:
        axes, axis_count = glfw.get_joystick_axes(joystick_id)
        if axis_count >= 4 and raw_candidate is None:
          del axes
          raw_candidate = joystick_id
    return None if raw_candidate is None else (raw_candidate, False)

  @staticmethod
  def _trigger_value(value: float) -> float:
    """Convert GLFW's [-1, 1] trigger range to a released-to-pressed value."""
    return max(0.0, min(1.0, (value + 1.0) * 0.5))

  def poll(self) -> None:
    """Read the current gamepad state and update the manual command."""
    device = self._find_joystick()
    if device is None:
      if not self._reported_missing:
        print("[WARN] No gamepad detected; connect one and keep the viewer open.")
        self._reported_missing = True
      self._joystick_id = None
      return

    joystick_id, mapped = device
    self._reported_missing = False
    self._joystick_id = joystick_id
    name = glfw.get_joystick_name(joystick_id).decode(errors="replace")
    if name != self._reported_name:
      print(f"[INFO] Gamepad velocity control enabled: {name}")
      self._reported_name = name

    if mapped:
      state = glfw.get_gamepad_state(joystick_id)  # ty: ignore[possibly-missing-attribute]
      if state is None:
        return
      axes = state.axes
      buttons = state.buttons
      left_x = float(axes[glfw.GAMEPAD_AXIS_LEFT_X])
      left_y = float(axes[glfw.GAMEPAD_AXIS_LEFT_Y])
      right_x = float(axes[glfw.GAMEPAD_AXIS_RIGHT_X])
      left_trigger = self._trigger_value(float(axes[glfw.GAMEPAD_AXIS_LEFT_TRIGGER]))
      right_trigger = self._trigger_value(float(axes[glfw.GAMEPAD_AXIS_RIGHT_TRIGGER]))
      button_values = tuple(int(button) for button in buttons)
      decrease_button = glfw.GAMEPAD_BUTTON_LEFT_BUMPER
      increase_button = glfw.GAMEPAD_BUTTON_RIGHT_BUMPER
    else:
      axes_pointer, axis_count = glfw.get_joystick_axes(joystick_id)
      if axis_count < 4:
        return
      axes = [float(axes_pointer[index]) for index in range(axis_count)]
      buttons_pointer, button_count = glfw.get_joystick_buttons(joystick_id)
      button_values = tuple(
        int(buttons_pointer[index]) for index in range(button_count)
      )
      left_x, left_y = axes[:2]
      left_trigger = self._trigger_value(axes[2]) if axis_count > 2 else 0.0
      right_x = axes[3] if axis_count > 3 else 0.0
      right_trigger = self._trigger_value(axes[5]) if axis_count > 5 else 0.0
      decrease_button = 4
      increase_button = 5

    if (
      decrease_button < len(button_values)
      and button_values[decrease_button]
      and not self._button_was_pressed(button_values, decrease_button)
    ):
      self.command.adjust_keyboard_max_speed(-TONY5_VELOCITY_KEYBOARD_SPEED_STEP)
    if (
      increase_button < len(button_values)
      and button_values[increase_button]
      and not self._button_was_pressed(button_values, increase_button)
    ):
      self.command.adjust_keyboard_max_speed(TONY5_VELOCITY_KEYBOARD_SPEED_STEP)
    self._last_buttons = button_values

    left_x = _apply_gamepad_deadzone(left_x)
    left_y = _apply_gamepad_deadzone(left_y)
    right_x = _apply_gamepad_deadzone(right_x)
    speed = self.command.keyboard_max_speed
    self.command.set_gamepad_command(
      (
        -left_y * speed,
        left_x * speed,
        right_x * TONY5_VELOCITY_MAX_YAW_RATE,
        (right_trigger - left_trigger) * TONY5_VELOCITY_VERTICAL_COMMAND_MAX,
      )
    )

  def _button_was_pressed(self, buttons: tuple[int, ...], index: int) -> bool:
    return index < len(self._last_buttons) and bool(self._last_buttons[index])


class RandomBaseTorqueDisturbance:
  """Apply one random world-frame base-torque impulse during play."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    peak_torque: float,
    resample_time_range: tuple[float, float] = (3.0, 6.0),
    duration: float = 0.5,
  ) -> None:
    if peak_torque <= 0.0:
      raise ValueError("peak_torque must be positive.")
    interval_min, interval_max = resample_time_range
    if interval_min <= 0.0 or interval_max < interval_min:
      raise ValueError("Disturbance resampling interval must be positive and ordered.")
    if duration <= 0.0:
      raise ValueError("Disturbance duration must be positive.")

    robot = env.scene["robot"]
    body_ids, body_names = robot.find_bodies(("quad_base",), preserve_order=True)
    if tuple(body_names) != ("quad_base",):
      raise ValueError(f"Unexpected TONY5 base body order: {body_names}")

    self._env = env
    self._robot = robot
    self._body_ids = body_ids
    self._peak_torque = peak_torque
    self._interval_min = interval_min
    self._interval_max = interval_max
    self._duration = duration
    self._forces = torch.zeros(
      (env.num_envs, len(body_ids), 3),
      device=env.device,
    )
    self._sampled_torques = torch.zeros_like(self._forces)
    self._torques = torch.zeros_like(self._forces)
    self._time_until_resample = torch.zeros(env.num_envs, device=env.device)
    self._active_time_remaining = torch.zeros(env.num_envs, device=env.device)

  @property
  def torques(self) -> torch.Tensor:
    """Currently applied world-frame torque for each environment."""
    return self._torques

  def apply(self) -> None:
    """Apply each sampled torque for 0.5 seconds every 3–6 seconds."""
    self._time_until_resample.sub_(self._env.step_dt)
    self._active_time_remaining.sub_(self._env.step_dt)
    resample = self._time_until_resample <= 0.0
    if torch.any(resample):
      sampled_torques = torch.empty_like(self._torques).uniform_(
        -self._peak_torque,
        self._peak_torque,
      )
      sampled_intervals = torch.empty(
        self._env.num_envs,
        device=self._env.device,
      ).uniform_(self._interval_min, self._interval_max)
      resample_expanded = resample.view(-1, 1, 1)
      self._sampled_torques[:] = torch.where(
        resample_expanded,
        sampled_torques,
        self._sampled_torques,
      )
      self._time_until_resample[:] = torch.where(
        resample,
        sampled_intervals,
        self._time_until_resample,
      )
      self._active_time_remaining[:] = torch.where(
        resample,
        torch.full_like(self._active_time_remaining, self._duration),
        self._active_time_remaining,
      )

    active = self._active_time_remaining > 0.0
    self._torques[:] = torch.where(
      active.view(-1, 1, 1),
      self._sampled_torques,
      torch.zeros_like(self._torques),
    )

    self._robot.write_external_wrench_to_sim(
      self._forces,
      self._torques,
      body_ids=self._body_ids,
    )

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    """Draw the active one-step torque impulse as a magenta arrow."""
    env_indices = visualizer.get_env_indices(self._env.num_envs)
    if not env_indices:
      return

    positions = self._robot.data.root_link_pos_w.detach().cpu().numpy()
    torques = self._torques[:, 0].detach().cpu().numpy()
    for env_idx in env_indices:
      torque = torques[env_idx]
      magnitude = float(np.linalg.norm(torque))
      if magnitude < 1.0e-9:
        visualizer.add_sphere(
          center=positions[env_idx],
          radius=0.025,
          color=(0.8, 0.1, 0.8, 0.9),
          label=f"tony5_disturbance_{env_idx}",
        )
        continue
      direction = torque / magnitude
      arrow_length = 0.30 * min(magnitude / self._peak_torque, 1.0)
      visualizer.add_arrow(
        start=positions[env_idx],
        end=positions[env_idx] + arrow_length * direction,
        color=(1.0, 0.1, 0.8, 1.0),
        width=0.025,
        label=f"tony5_disturbance_{env_idx}",
      )


class DisturbedPolicy:
  """Apply a play-only disturbance before forwarding observations to a policy."""

  def __init__(
    self,
    policy: Callable[[Any], torch.Tensor],
    disturbance: RandomBaseTorqueDisturbance,
  ) -> None:
    self._policy = policy
    self._disturbance = disturbance

  def __call__(self, obs: Any) -> torch.Tensor:
    self._disturbance.apply()
    return self._policy(obs)

  def reset(self) -> None:
    """Forward policy reset hooks used by recurrent policies, if present."""
    reset_fn = getattr(self._policy, "reset", None)
    if reset_fn is not None:
      reset_fn()


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  keyboard_task_ids = {
    "Mjlab-Tony5-Velocity-Aero-v1",
    "Mjlab-Tony5-Velocity-Aero-Omni-v0",
    "Mjlab-Tony5-Velocity-Aero-Omni-v3",
  }
  disturbance_task_ids = keyboard_task_ids | {
    "Mjlab-Tony5-Velocity-Aero-Omni-v3",
  }
  if (cfg.keyboard or cfg.gamepad) and task_id not in keyboard_task_ids:
    raise ValueError(
      "--keyboard and --gamepad are supported only for the TONY5 Aero velocity tasks."
    )
  if cfg.disturbance and task_id not in disturbance_task_ids:
    raise ValueError(
      "--disturbance is supported only for the TONY5 Aero velocity tasks."
    )
  if cfg.disturbance and cfg.disturbance_torque <= 0.0:
    raise ValueError(
      "--disturbance-torque must be positive when disturbance is enabled."
    )
  if cfg.disturbance_interval_min <= 0.0:
    raise ValueError("--disturbance-interval-min must be positive.")
  if cfg.disturbance_interval_max < cfg.disturbance_interval_min:
    raise ValueError(
      "--disturbance-interval-max must be at least the minimum interval."
    )
  if cfg.disturbance_duration <= 0.0:
    raise ValueError("--disturbance-duration must be positive.")
  if cfg.playback_speed <= 0.0:
    raise ValueError("--playback-speed must be positive.")
  if cfg.keyboard and cfg.gamepad:
    raise ValueError("Choose either --keyboard or --gamepad, not both.")

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  if task_id == "Mjlab-Tony5-Velocity-Aero-Omni-v3":
    action_cfg = env_cfg.actions.get("rotor_speed")
    wind_cfg = getattr(action_cfg, "wind", None)
    if isinstance(wind_cfg, Tony5OmniV3WindCfg):
      wind_cfg.enable_wind = cfg.wind
      wind_cfg.enable_gusts = cfg.gusts
      wind_cfg.reset_enable_probability = 1.0 if (cfg.wind or cfg.gusts) else 0.0
      print(
        "[INFO]: V3 play wind: "
        f"background={'on' if cfg.wind else 'off'}, "
        f"gusts={'on' if cfg.gusts else 'off'}, "
        f"episode_activation={'100%' if (cfg.wind or cfg.gusts) else '0%'}"
      )

  if cfg.high_speed_only:
    velocity_cfg = env_cfg.commands.get("velocity")
    if not isinstance(velocity_cfg, Tony5RadialVelocityCommandCfg):
      raise ValueError(
        "--high-speed-only is supported only for Mjlab-Tony5-Velocity-Aero-v1."
      )
    if cfg.high_speed_min < 0.0:
      raise ValueError("--high-speed-min must be non-negative.")
    if cfg.high_speed_max <= cfg.high_speed_min:
      raise ValueError("--high-speed-max must exceed --high-speed-min.")
    if cfg.high_speed_max > velocity_cfg.stage_speeds[-1]:
      raise ValueError(
        "--high-speed-max cannot exceed the V1 final speed of "
        f"{velocity_cfg.stage_speeds[-1]:g} m/s."
      )
    velocity_cfg.play_high_speed_only = True
    velocity_cfg.play_high_speed_min = cfg.high_speed_min
    velocity_cfg.play_high_speed_max = cfg.high_speed_max
    print(
      "[INFO]: V1 high-speed play sampling enabled: "
      f"{cfg.high_speed_min:g}-{cfg.high_speed_max:g} m/s"
    )

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        excluded_runs = (
          frozenset({"v0_bootstrap"})
          if task_id == "Mjlab-Tony5-Velocity-Aero-Omni-v3"
          else frozenset()
        )
        resume_path = _get_latest_local_checkpoint(
          log_root_path,
          excluded_run_names=excluded_runs,
        )
        print(
          f"[INFO]: Loading latest local checkpoint: {resume_path.name} "
          f"(run: {resume_path.parent.name})"
        )
      else:
        resume_path, was_cached = get_wandb_checkpoint_path(
          log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
        )
        # Extract run_id and checkpoint name from path for display.
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
        )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  velocity_command = env.command_manager.get_term("velocity")
  resampler: Tony5PlayResampler | None = None
  wind: Tony5OmniV3GlobalWind | None = None
  if task_id == "Mjlab-Tony5-Velocity-Aero-Omni-v3":
    action_term = env.action_manager.get_term("rotor_speed")
    candidate_wind = getattr(action_term, "wind", None)
    if isinstance(candidate_wind, Tony5OmniV3GlobalWind):
      wind = candidate_wind
      env.manager_visualizers["play_wind"] = wind
  if isinstance(velocity_command, Tony5RadialVelocityCommand):
    resampler = Tony5PlayResampler(velocity_command, wind)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  keyboard_controller: Tony5KeyboardController | None = None
  gamepad_controller: Tony5GamepadController | None = None
  if cfg.keyboard:
    keyboard_controller = Tony5KeyboardController(
      env.unwrapped,
      resample_callback=resampler.resample if resampler is not None else None,
    )
    print(
      "[INFO] Keyboard velocity control enabled: "
      "W/S forward, A/D lateral, Q/E yaw, R/F vertical, X hover, "
      "+/- (or keypad +/-) decrease/increase max speed (up to 100 m/s), M "
      "resamples velocity and wind"
    )
  if cfg.gamepad:
    gamepad_controller = Tony5GamepadController(env.unwrapped)

  disturbance: RandomBaseTorqueDisturbance | None = None
  if cfg.disturbance:
    disturbance = RandomBaseTorqueDisturbance(
      env.unwrapped,
      peak_torque=cfg.disturbance_torque,
      resample_time_range=(
        cfg.disturbance_interval_min,
        cfg.disturbance_interval_max,
      ),
      duration=cfg.disturbance_duration,
    )
    env.unwrapped.manager_visualizers["play_disturbance"] = disturbance
    policy = DisturbedPolicy(policy, disturbance)
    print(
      "[INFO] Random base torque disturbance enabled: "
      f"+/-{cfg.disturbance_torque:g} N*m for "
      f"{cfg.disturbance_duration:g} s every "
      f"{cfg.disturbance_interval_min:g}-{cfg.disturbance_interval_max:g} s"
    )

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      loaded_policy = _ckpt_runner.get_inference_policy(device=device)
      if disturbance is not None:
        return DisturbedPolicy(loaded_policy, disturbance)
      return loaded_policy

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if gamepad_controller is not None and resolved_viewer != "native":
    raise ValueError("--gamepad requires the native viewer (use --viewer native).")

  def handle_custom_action(action: ViewerAction, payload: object | None) -> bool:
    if keyboard_controller is not None and keyboard_controller.handle_action(
      action, payload
    ):
      return True
    return resampler is not None and resampler.handle_action(action, payload)

  has_custom_controls = keyboard_controller is not None or resampler is not None

  def create_viser_controls(
    server: Any,
    request_action: Callable[[str, Any], None],
  ) -> None:
    if keyboard_controller is not None:
      keyboard_controller.create_viser_gui(server, request_action)
    if resampler is not None:
      resampler.create_viser_gui(server, request_action)

  if resolved_viewer == "native":
    native_viewer: NativeMujocoViewer | None = None

    def queue_keyboard_key(key: int) -> None:
      if native_viewer is not None:
        from mjlab.viewer.native.keys import KEY_M

        if key in (KEY_M, ord("m"), ord("M")):
          native_viewer.request_action(
            "CUSTOM",
            {"type": "resample_velocity_wind"},
          )
          return
        native_viewer.request_action(
          "CUSTOM",
          {"type": "keyboard_key", "key": key},
        )

    native_viewer = NativeMujocoViewer(
      env,
      policy,
      key_callback=queue_keyboard_key if has_custom_controls else None,
      keyboard_control=keyboard_controller is not None,
      input_poll_callback=(
        gamepad_controller.poll if gamepad_controller is not None else None
      ),
      custom_action_handler=handle_custom_action if has_custom_controls else None,
      initial_speed_multiplier=cfg.playback_speed,
    )
    native_viewer.run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(
      env,
      policy,
      checkpoint_manager=ckpt_manager,
      custom_action_handler=handle_custom_action if has_custom_controls else None,
      custom_gui_setup=create_viser_controls if has_custom_controls else None,
      initial_speed_multiplier=cfg.playback_speed,
    ).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  maybe_print_top_level_help("play")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
