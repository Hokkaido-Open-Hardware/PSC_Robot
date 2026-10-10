"""Remote GUI commands, AUTO progression and reset during an IPC wait."""
import os
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simulation'))
from PySide6.QtWidgets import QApplication
from auto_commands import DEFAULT_DURATION
from psc_panel import PSCThread
from psc_protocol import quantize
from robot_lqr import RobotLqr
from simulate_robot import Window


class Link:
    def __init__(self, hook=None):
        self.frames = []
        self.hook = hook
        self.closed = False

    def exchange(self, values):
        self.frames.append(values)
        if self.hook:
            self.hook()
        return 2.5

    def close(self):
        self.closed = True


class PSCCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = Window(psc_child=True)
        self.addCleanup(self.window.close)
        self.link = Link()
        self.worker = PSCThread(self.window, self.link, 1)

    def test_manual_widgets_feed_remote_target_and_motor_yaw(self):
        w = self.window
        self.assertTrue(w.command_mode.isEnabled())
        self.assertTrue(w.speed_slider.isEnabled())
        self.assertTrue(w.yaw_slider.isEnabled())
        self.assertTrue(w.reset_button.isEnabled())
        self.assertFalse(w.controller_selector.isEnabled())
        w.speed_slider.setValue(200)
        w.yaw_slider.setValue(300)
        with patch.object(RobotLqr, 'calculate_lqr_velocity', side_effect=AssertionError('local control')):
            self.worker.run()
        self.assertFalse(self.worker.failed)
        self.assertEqual(quantize(self.link.frames[0])[3], 50)
        status = w.th.control_status
        self.assertEqual(status.controller, 'PSC-ONE / PSC-OS')
        self.assertAlmostEqual(status.left_motor_command, -2.2)
        self.assertAlmostEqual(status.right_motor_command, 2.8)
        self.assertAlmostEqual(w.data.time, .005)
        self.assertEqual(w.speed_slider.maximum(), 508)
        self.assertEqual(w.speed_slider.minimum(), -512)

    def test_auto_advances_in_plant_time_and_manual_restores_controls(self):
        w = self.window
        w.command_mode.setCurrentText('AUTO')
        self.assertTrue(w.th.auto_status.enabled)
        self.assertFalse(w.speed_slider.isEnabled())
        w.th.physics_steps = round(DEFAULT_DURATION / w.model.opt.timestep)
        self.worker.run()
        self.assertFalse(self.worker.failed)
        self.assertEqual(w.th.auto_status.episode, 2)
        self.assertEqual(w.th.auto_status.scenario, 'Gentle Forward')
        self.assertGreater(self.link.frames[0][3], 0)
        self.assertAlmostEqual(w.th.auto_status.elapsed, .005)
        w.command_mode.setCurrentText('MANUAL')
        self.assertTrue(w.speed_slider.isEnabled())
        self.assertTrue(w.yaw_slider.isEnabled())
        self.assertEqual((w.th.speed, w.th.yaw), (0, 0))

    def test_reset_or_mode_change_discards_pending_response(self):
        w = self.window
        for action in (w.reset_simulation, lambda: w.command_mode.setCurrentText('AUTO'),
                       lambda: w.command_mode.setCurrentText('MANUAL')):
            with self.subTest(action=action):
                epoch = w.th.command_epoch
                self.link.hook = action
                self.worker.run()
                self.assertFalse(self.worker.failed)
                self.assertGreater(w.th.command_epoch, epoch)
                self.assertEqual(w.data.time, 0)
                self.assertEqual(w.th.physics_steps, 0)
                self.assertEqual(w.th.control_status.control_output, None)

    def test_gui_commands_do_not_wait_for_ipc_lock(self):
        entered, release = threading.Event(), threading.Event()
        def wait():
            entered.set()
            if not release.wait(3):
                raise TimeoutError('GUI blocked on IPC')
        self.link.hook = wait
        self.worker.start()
        try:
            self.assertTrue(entered.wait(2))
            self.window.speed_slider.setValue(-100)
            self.window.yaw_slider.setValue(500)
            self.window.command_mode.setCurrentText('AUTO')
            self.assertTrue(self.window.th.auto_enabled)
        finally:
            release.set()
            self.assertTrue(self.worker.wait(5000))
        self.assertFalse(self.worker.failed)
        self.assertEqual(self.window.data.time, 0)


if __name__ == '__main__':
    unittest.main()
