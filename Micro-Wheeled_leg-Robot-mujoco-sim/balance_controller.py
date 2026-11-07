"""Minimal four-state balance feedback adapted from wl_pro_robot.ino.

Source: external/Micro-Wheeled_leg-Robot/3.Software/wl_pro_robot/wl_pro_robot.ino
The upstream lqr_balance_loop sums angle, gyro, wheel angle and wheel speed
feedback. Here the gains are tuned for this MJCF in SI units, directly producing
torque; they are not the result of solving an LQR optimization. ESP32 I/O,
voltage/FOC control, integral compensation, adaptive trim, yaw and jump control
are not ported. This is not an electrical motor model or firmware emulation.

Copyright (c) 2024 Mu Shibo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BalanceGains:
    # Per-wheel output in N m; all four input states use radians and seconds.
    angle: float = 0.15
    gyro: float = 0.01
    distance: float = 0.0005
    speed: float = 0.002
    # Empirical equilibrium of scene.xml, including loaded hip-servo deflection.
    pitch_reference_deg: float = 11.55
    # Negative hip rotation extends the four-bar legs from the CAD posture.
    hip_target_rad: float = -0.1
    fall_angle_deg: float = 45.0


def pitch_angle(data):
    """Chassis pitch in radians, using the simulated orientation sensor."""
    w, x, y, z = data.sensor('orientation').data
    return float(np.arcsin(np.clip(2 * (w*y - z*x), -1, 1)))


class BalanceController:
    def __init__(self, model, data, gains=None):
        self.gains = gains or BalanceGains()
        self.wheel_qpos = [int(model.joint(s+'_wheel_joint').qposadr[0])
                           for s in ('left', 'right')]
        self.wheel_dof = [int(model.joint(s+'_wheel_joint').dofadr[0])
                          for s in ('left', 'right')]
        self.wheel_actuators = [model.actuator(s+'_wheel_motor').id
                               for s in ('left', 'right')]
        self.hip_actuators = [model.actuator(s+'_hip_servo').id
                             for s in ('left', 'right')]
        self.torque_ranges = model.actuator_ctrlrange[self.wheel_actuators].copy()
        self.reset(data)

    def reset(self, data):
        self.wheel_reference = float(np.mean(data.qpos[self.wheel_qpos]))
        self.fallen = False
        self.max_abs_torque = 0.0
        data.ctrl[self.hip_actuators] = self.gains.hip_target_rad
        data.ctrl[self.wheel_actuators] = 0

    def update(self, data):
        gains = self.gains
        w, x, y, z = data.sensor('orientation').data
        tilt = np.arccos(np.clip(1 - 2*(x*x + y*y), -1, 1))
        # Latch off after a fall; R/reset is required to resume.
        self.fallen |= bool(tilt > np.deg2rad(gains.fall_angle_deg))
        data.ctrl[self.hip_actuators] = gains.hip_target_rad
        if self.fallen:
            data.ctrl[self.wheel_actuators] = 0
            return

        pitch_error = pitch_angle(data) - np.deg2rad(gains.pitch_reference_deg)
        gyro = float(data.sensor('gyro').data[1])
        distance = float(np.mean(data.qpos[self.wheel_qpos])) - self.wheel_reference
        speed = float(np.mean(data.qvel[self.wheel_dof]))
        # Both MJCF wheel axes are +Y, so positive torque rolls forward (+X).
        # Upstream's wiring sign and degree-based gains must not be copied here.
        torque = (gains.angle*pitch_error + gains.gyro*gyro
                  + gains.distance*distance + gains.speed*speed)
        output = np.clip(torque, self.torque_ranges[:, 0], self.torque_ranges[:, 1])
        data.ctrl[self.wheel_actuators] = output
        self.max_abs_torque = max(self.max_abs_torque, float(np.max(np.abs(output))))
