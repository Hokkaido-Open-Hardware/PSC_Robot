"""GUI AUTO uses simulation time and keeps inference separate from training."""
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simulation'))
from auto_commands import DEFAULT_DURATION, SCENARIOS, SPEED_SLEW, YAW_SLEW
from control_timing import CONTROL_DT
from nn_model import DEFAULT_MODEL, NNPredictor
from rl_control import RLConfig, SCENE, body_tilt
from simulate_robot import UpdateSimThread


class AutoModeTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.worker = UpdateSimThread(self.model, self.data)
        self.addCleanup(self.worker.stop)

    def step(self, count=1):
        with self.worker.data_lock:
            for _ in range(count):
                self.worker._step_locked()

    def finish_episode(self):
        # Exercise the exact timeout boundary without simulating eight seconds.
        self.worker.physics_steps = round(DEFAULT_DURATION / self.model.opt.timestep)
        self.step()

    def test_all_patterns_repeat_and_commands_only_change_at_control_ticks(self):
        self.worker.set_auto(True)
        seen = []
        for episode in range(1, len(SCENARIOS) + 2):
            status = self.worker.auto_status
            seen.append(status.scenario)
            self.assertEqual(status.episode, episode)
            self.assertEqual(status.falls, 0)
            self.finish_episode()
        self.assertEqual(seen, list(SCENARIOS) + [SCENARIOS[0]])
        # We are now in Gentle Forward, immediately after its first control tick.
        self.assertEqual(self.worker.auto_status.scenario, 'Gentle Forward')
        first = self.worker.speed
        self.assertAlmostEqual(first, SPEED_SLEW * CONTROL_DT)
        self.step(self.worker.control_substeps - 1)
        self.assertEqual(self.worker.speed, first)
        self.step()
        self.assertAlmostEqual(self.worker.speed, first * 2)
        self.assertEqual(self.worker.yaw, 0)
        while self.worker.auto_status.scenario != 'Left Turn':
            self.finish_episode()
        self.assertAlmostEqual(self.worker.yaw, YAW_SLEW * CONTROL_DT)

    def test_fall_in_any_direction_and_invalid_state_reset(self):
        self.worker.set_auto(True)
        quat = Rotation.from_euler('y', 1.).as_quat()
        self.data.qpos[3:7] = quat[[3, 0, 1, 2]]
        mujoco.mj_forward(self.model, self.data)
        self.step()
        self.assertEqual(self.worker.auto_status.episode, 2)
        self.assertEqual(self.worker.auto_status.falls, 1)
        self.assertLess(body_tilt(self.data), .1)
        self.assertAlmostEqual(self.data.time, self.model.opt.timestep)
        self.worker.physics_steps = self.worker.control_substeps
        self.data.qvel[0] = np.nan
        self.step()
        self.assertEqual(self.worker.auto_status.episode, 3)
        self.assertEqual(self.worker.auto_status.falls, 2)
        self.assertTrue(np.isfinite(self.data.qvel).all())

    def test_manual_input_is_ignored_until_manual_mode_restores_zero_commands(self):
        self.worker.set_auto(True)
        self.finish_episode()
        speed = self.worker.speed
        self.worker.set_speed(4)
        self.worker.set_yaw(10)
        self.assertEqual(self.worker.speed, speed)
        self.assertEqual(self.worker.yaw, 0)
        before = self.data.qpos.copy()
        self.worker.set_auto(False)
        np.testing.assert_array_equal(before, self.data.qpos)
        self.assertEqual((self.worker.speed, self.worker.yaw), (0, 0))
        self.worker.set_speed(.2)
        self.worker.set_yaw(.3)
        self.assertEqual((self.worker.speed, self.worker.yaw), (.2, .3))

    def test_reset_restarts_seeded_sequence_and_preserves_each_controller(self):
        predictors = [('LQR', None), ('Neural Network', NNPredictor(DEFAULT_MODEL)),
                      ('Reinforcement Learning', Mock(config=RLConfig(), predict=Mock(return_value=[0.])))]
        for name, predictor in predictors:
            with self.subTest(controller=name):
                self.worker.set_auto(False)
                self.worker.set_controller(name, predictor)
                self.worker.set_auto(True)
                initial = self.data.qpos.copy()
                self.finish_episode()
                self.worker.reset()
                self.assertTrue(self.worker.auto_status.enabled)
                self.assertEqual(self.worker.auto_status.episode, 1)
                self.assertEqual(self.worker.auto_status.falls, 0)
                np.testing.assert_array_equal(self.data.qpos, initial)
                self.step()
                self.assertEqual(self.worker.control_status.controller, name)
                self.assertEqual(self.worker.control_status.target_linear_velocity, 0)

    def test_controller_switch_preserves_auto_schedule_and_physics(self):
        self.worker.set_auto(True)
        self.finish_episode()
        before = self.data.qpos.copy()
        status = self.worker.auto_status
        self.worker.set_controller('Neural Network', NNPredictor(DEFAULT_MODEL))
        np.testing.assert_array_equal(before, self.data.qpos)
        self.assertEqual(status, self.worker.auto_status)


if __name__ == '__main__':
    unittest.main()
