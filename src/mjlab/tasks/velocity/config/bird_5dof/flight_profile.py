"""Shared tuning for velocity-vector-aligned bird flight."""

DIRECTIONAL_FLIGHT_REWARD_WEIGHTS = {
  "velocity_direction_alignment": 0.5,
  "low_speed_world_up": 1.0,
  "level_roll": 1.0,
  "roll_pitch_angular_velocity": -0.001,
}

DIRECTION_REWARD_STD = 0.5
DIRECTION_MIN_SPEED = 0.25
LOW_SPEED_WORLD_UP_MAX_SPEED = DIRECTION_MIN_SPEED
ROLL_REWARD_STD = 0.5
ROLL_MIN_HORIZONTAL = 0.1
ROLL_FULL_SPEED = 0.25
ROLL_ZERO_SPEED = 1.5
TILT_LIMIT_DEG = 120.0
