import math
from dataclasses import replace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from control_status import ControlStatus

# obtained from running `calculate_lqr_gains.py`
LQR_K = [-2.1402165848237837, -0.03501370844016172, 5.9748026764525894e-18, 2.236067977499789]
WHEEL_RADIUS = 0.034
MAX_MOTOR_VEL = 500.0 # rad/s


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


"""
Basic LQR controller implementation
"""
class RobotLqr:

    controller_name = "LQR"

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

        self.velocity_angular = 0.0
        self.velocity_linear_set_point = 0.0
        self.yaw = 0

        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0
        self.control_status = ControlStatus(controller=self.controller_name)

    def set_velocity_linear_set_point(self, vel: float) -> None:
        """
        Sets the target velocity of the robot
        """
        self.velocity_linear_set_point = vel

    def set_yaw(self, yaw: float) -> None:
        """
        The yaw value is added to one wheel, and subtracted from the other
        to produce a yaw motion (turn)
        """
        self.yaw = yaw

    def get_pitch(self) -> float:
        quat = self.data.body("robot_body").xquat
        if quat[0] == 0:
            return 0

        rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])  # Quaternion order is [x, y, z, w]
        angles = rotation.as_euler('xyz', degrees=False)
        # print(angles)
        return angles[0]

    def get_pitch_dot(self) -> float:
        angular = self.data.joint('robot_body_joint').qvel[-3:]
        # print(angular)
        return angular[0]

    def get_wheel_velocity(self) -> float:
        vel_m_0 = self.data.joint('torso_l_wheel').qvel[0]
        vel_m_1 = self.data.joint('torso_r_wheel').qvel[0]

        # both wheels spin "forward", but one is spinning in a negative
        # direction as it's rotated 180deg from the other
        average = (vel_m_0 * -1 + vel_m_1) / 2.0
        self._wheel_sample = (float(vel_m_0), float(vel_m_1), float(average))
        return average

    def sample_control_input(self):
        """Sample once per control tick; both controllers use identical filters."""
        pitch = -self.get_pitch()
        pitch_dot = self.get_pitch_dot()

        # apply a filter to pitch dot, and velocity
        # without these filters the controller seems to lack necessary dampening
        # would like to know why!
        self.pitch_dot_filtered = (self.pitch_dot_filtered * .975) + (pitch_dot * .025)
        self.velocity_angular_filtered = (self.velocity_angular_filtered * .975) + (self.get_wheel_velocity() * .025)

        return (pitch, self.pitch_dot_filtered, self.velocity_angular_filtered,
                self.velocity_linear_set_point)

    def calculate_lqr_velocity(self) -> float:
        pitch, _, _, _ = self.sample_control_input()

        velocity_linear_error = self.velocity_linear_set_point - self.velocity_angular_filtered * WHEEL_RADIUS

        lqr_v = LQR_K[0] * (0 - pitch) + LQR_K[1] * self.pitch_dot_filtered + LQR_K[2] * 0 + LQR_K[3] * velocity_linear_error 
        output = -lqr_v / WHEEL_RADIUS
        self.record_control_output(pitch, output)
        return output

    def record_control_output(self, pitch, output):
        # Keep the exact inputs and output of this calculation. Publish only
        # after update_motor_speed has applied the corresponding motor commands.
        left, right, average = self._wheel_sample
        self._pending_status = ControlStatus(
            controller=self.controller_name,
            pitch=float(pitch),
            pitch_rate=float(self.pitch_dot_filtered),
            left_wheel_velocity=left,
            right_wheel_velocity=right,
            average_wheel_velocity=average,
            target_linear_velocity=float(self.velocity_linear_set_point),
            velocity_error=float(self.velocity_linear_set_point - self.velocity_angular_filtered * WHEEL_RADIUS),
            control_output=float(output),
            yaw_command=float(self.yaw),
            simulation_time=float(self.data.time),
        )

    def calculate_motor_velocity(self):
        return self.calculate_lqr_velocity()
    
    def update_motor_speed(self, ) -> None:
        vel = self.calculate_motor_velocity()
        vel = clamp(vel, -MAX_MOTOR_VEL, MAX_MOTOR_VEL)

        self.data.actuator('motor_l_wheel').ctrl = [-vel + self.yaw]
        self.data.actuator('motor_r_wheel').ctrl = [vel + self.yaw]
        self.control_status = replace(
            self._pending_status,
            left_motor_command=float(self.data.actuator('motor_l_wheel').ctrl[0]),
            right_motor_command=float(self.data.actuator('motor_r_wheel').ctrl[0]),
        )

    def reset(self):
        self.velocity_angular = 0.0
        self.velocity_linear_set_point = 0.0
        self.yaw = 0
        self.pitch_dot_filtered = 0.0
        self.velocity_angular_filtered = 0.0

        # face a random direction
        x_rot = (np.random.random() - 0.5) * 2 * math.pi
        # rotate and pitch slightly
        y_rot = (np.random.random() - 0.5) * 0.4
        z_rot = (np.random.random() - 0.5) * 0.4
        euler_angles = [x_rot, y_rot, z_rot]
        # Convert to quaternion
        rotation = Rotation.from_euler('xyz', euler_angles)
        self.data.qpos[3:7] = rotation.as_quat()

        self.data.actuator('motor_l_wheel').ctrl = [0]
        self.data.actuator('motor_r_wheel').ctrl = [0]
        self.control_status = ControlStatus(
            controller=self.controller_name,
            target_linear_velocity=0.0,
            left_motor_command=0.0,
            right_motor_command=0.0,
            yaw_command=0.0,
            simulation_time=float(self.data.time),
        )
