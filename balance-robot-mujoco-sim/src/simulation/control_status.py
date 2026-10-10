from dataclasses import dataclass


@dataclass(frozen=True)
class ControlStatus:
    """One completed control update, shared with the GUI by reference.

    Controllers publish a new immutable snapshot; the GUI never samples sensors
    or calls the controller. None means no control sample is available yet.
    """

    controller: str
    pitch: float | None = None  # radians, with the controller's sign convention
    pitch_rate: float | None = None  # radians/s; filtered LQR/NN, raw local X for RL
    left_wheel_velocity: float | None = None  # joint radians/s
    right_wheel_velocity: float | None = None
    average_wheel_velocity: float | None = None  # forward-positive radians/s
    target_linear_velocity: float | None = None  # m/s
    velocity_error: float | None = None  # m/s; filtered wheel LQR/NN, raw body speed RL
    control_output: float | None = None  # radians/s, before motor limiting/mixing
    left_motor_command: float | None = None
    right_motor_command: float | None = None
    yaw_command: float | None = None
    simulation_time: float | None = None  # seconds at the control sample
