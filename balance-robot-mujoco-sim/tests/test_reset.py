"""Run with: python -m unittest discover -s tests -v (no display needed)."""
from collections import deque
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'simulation'))
from simulate_robot import UpdateSimThread, Viewport

SCENE = Path(__file__).resolve().parents[1] / 'src' / 'simulation' / 'scene.xml'


class ResetTests(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.worker = UpdateSimThread(self.model, self.data)
        self.addCleanup(self.worker.stop)

    def test_reset_restores_state_and_refreshes_pose(self):
        self.data.time = 12
        self.data.qvel[:] = 4
        self.data.ctrl[:] = 5
        self.worker.set_speed(2)
        self.worker.set_yaw(3)
        self.worker.robot.pitch_dot_filtered = 7
        self.worker.robot.velocity_angular_filtered = 8
        self.worker.reset()
        self.assertEqual(self.data.time, 0)
        np.testing.assert_array_equal(self.data.qvel, 0)
        np.testing.assert_array_equal(self.data.ctrl, 0)
        np.testing.assert_allclose(self.data.body('robot_body').xquat, self.data.qpos[3:7])
        self.assertEqual(self.worker.speed, 0)
        self.assertEqual(self.worker.yaw, 0)
        self.assertEqual(self.worker.robot.pitch_dot_filtered, 0)
        self.assertEqual(self.worker.robot.velocity_angular_filtered, 0)
        self.assertEqual(self.worker.control_status.simulation_time, 0)
        self.assertIsNone(self.worker.control_status.pitch)

    def test_reset_waits_for_inflight_step(self):
        step_entered = threading.Event()
        release_step = threading.Event()
        reset_started = threading.Event()
        reset_done = threading.Event()
        real_step = mujoco.mj_step

        def blocked_step(model, data):
            step_entered.set()
            release_step.wait(5)
            real_step(model, data)

        def reset():
            reset_started.set()
            self.worker.reset()
            reset_done.set()

        with patch('simulate_robot.mujoco.mj_step', side_effect=blocked_step):
            self.worker.start()
            reset_thread = threading.Thread(target=reset)
            try:
                self.assertTrue(step_entered.wait(2))
                reset_thread.start()
                self.assertTrue(reset_started.wait(2))
                self.assertFalse(reset_done.wait(0.05), 'reset overlapped mj_step')
            finally:
                release_step.set()
                if reset_thread.ident is not None:
                    reset_thread.join(3)
                self.worker.stop()
            self.assertTrue(reset_done.is_set())

    def test_scene_preparation_uses_same_lock(self):
        # Exercise paintGL without a display or an OpenGL context.
        lock = self.worker.data_lock
        viewport = SimpleNamespace(
            data_lock=lock, model=self.model, data=self.data,
            opt=mujoco.MjvOption(), cam=mujoco.MjvCamera(),
            scn=mujoco.MjvScene(self.model, maxgeom=1000),
            width=100, height=100, scale=1, con=None,
            runtime=deque(maxlen=10), updateRuntime=Mock(),
        )
        real_update = mujoco.mjv_updateScene

        def update(*args):
            self.assertTrue(lock.locked())
            real_update(*args)

        def render(*args):
            self.assertFalse(lock.locked())

        with patch('simulate_robot.mujoco.mjv_updateScene', side_effect=update), \
                patch('simulate_robot.mujoco.mjr_render', side_effect=render):
            Viewport.paintGL(viewport)

    def test_repeated_reset_while_running(self):
        self.worker.start()
        for _ in range(500):
            time.sleep(0.0003)
            self.worker.set_speed(1)
            self.worker.set_yaw(2)
            self.worker.reset()
        with self.worker.data_lock:
            self.assertTrue(np.isfinite(self.data.qpos).all())
            self.assertTrue(np.isfinite(self.data.qvel).all())
            self.assertFalse(self.data.warning.number.any())
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            with self.worker.data_lock:
                if self.data.time > 0.01:
                    break
            time.sleep(0.001)
        else:
            self.fail('simulation failed to advance after reset')
        self.assertTrue(self.worker.isRunning())


if __name__ == '__main__':
    unittest.main()
