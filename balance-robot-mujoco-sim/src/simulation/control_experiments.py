"""Deterministic headless MuJoCo rollouts at the controller's 200 Hz rate."""
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from robot_lqr import MAX_MOTOR_VEL, RobotLqr
from control_timing import CONTROL_DT, control_substeps

SCENE = Path(__file__).with_name('scene.xml')
FALL_PITCH = np.deg2rad(45)


@dataclass(frozen=True)
class Scenario:
    name: str
    pitch: float
    heading: float = 0.0
    pitch_rate: float = 0.0
    wheel_velocity: float = 0.0
    targets: tuple = (0.0,)
    yaw: float = 0.0
    duration: float = 5.0
    ramp: bool = False
    roll: float = 0.0

    def target_at(self, t):
        if self.ramp:
            return float(np.interp(t, np.linspace(0, self.duration, len(self.targets)), self.targets))
        return self.targets[min(int(t / self.duration * len(self.targets)), len(self.targets) - 1)]


def prepare(model, data, robot, scenario):
    mujoco.mj_resetData(model, data)
    # Constructors already zero controller filters; avoid the GUI's random reset.
    quat = Rotation.from_euler('xyz', [-scenario.pitch, scenario.roll, scenario.heading]).as_quat()
    data.qpos[3:7] = quat[[3, 0, 1, 2]]  # MuJoCo uses w, x, y, z.
    data.qvel[3] = scenario.pitch_rate
    data.joint('torso_l_wheel').qvel[0] = -scenario.wheel_velocity
    data.joint('torso_r_wheel').qvel[0] = scenario.wheel_velocity
    mujoco.mj_forward(model, data)
    return control_substeps(model)


def control_input(robot):
    status = robot.control_status
    return (status.pitch, status.pitch_rate, robot.velocity_angular_filtered,
            status.target_linear_velocity)


def applied_velocity(robot):
    # After limiting, before yaw mixing. Read the actual commands, not a label
    # recomputed by a second controller call (which would advance the filters).
    s = robot.control_status
    return (s.right_motor_command - s.left_motor_command) / 2


def lqr_reference(inputs):
    from robot_lqr import LQR_K, WHEEL_RADIUS
    pitch, rate, wheel, target = np.asarray(inputs).T
    error = target - wheel * WHEEL_RADIUS
    raw = -(LQR_K[0] * -pitch + LQR_K[1] * rate + LQR_K[2] * 0 + LQR_K[3] * error) / WHEEL_RADIUS
    return np.clip(raw, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

