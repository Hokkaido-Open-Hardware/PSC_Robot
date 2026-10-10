"""Robot process launched only by cocotb; never computes a local control output."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from control_timing import control_substeps
from psc_protocol import RobotLink
from robot_lqr import RobotLqr


class PSCSensors(RobotLqr):
    controller_name = 'PSC-ONE / PSC-OS'

    def reset(self):
        super().reset()
        q = Rotation.from_euler('x', .02).as_quat()
        self.data.qpos[3:7] = [q[3], q[0], q[1], q[2]]


class RobotPlant:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.sensors = PSCSensors(model, data)
        mujoco.mj_resetData(model, data)
        self.sensors.reset()
        mujoco.mj_forward(model, data)
        self.substeps = control_substeps(model)

    def sample(self, speed=0, yaw=0):
        self.sensors.set_velocity_linear_set_point(speed)
        self.sensors.set_yaw(yaw)
        self.state = self.sensors.sample_control_input()
        return self.state

    def advance(self, output):
        self.sensors.record_control_output(self.state[0], output)
        left, right = -output + self.sensors.yaw, output + self.sensors.yaw
        self.data.actuator('motor_l_wheel').ctrl = left
        self.data.actuator('motor_r_wheel').ctrl = right
        self.sensors.control_status = replace(
            self.sensors._pending_status, left_motor_command=left, right_motor_command=right)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise ValueError('Non-finite robot physics')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--socket', required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()
    link = RobotLink(args.socket)
    try:
        if args.gui:
            from psc_panel import run_robot_gui
            return run_robot_gui(link, args.steps)
        model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name('scene.xml')))
        data = mujoco.MjData(model)
        plant = RobotPlant(model, data)
        for _ in range(args.steps):
            plant.advance(link.exchange(plant.sample()))
        print(json.dumps(dict(robot_steps=args.steps, simulation_time=data.time,
                              final_output=plant.sensors.control_status.control_output)), flush=True)
        return 0
    finally:
        link.close()


if __name__ == '__main__':
    sys.exit(main())
