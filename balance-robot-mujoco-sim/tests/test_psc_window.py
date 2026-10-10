"""Attach the existing window after boot; stop, restart and cancel safely."""
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simulation'))
from PySide6.QtWidgets import QApplication
from psc_protocol import receive_message, send_message, validate_frame
from simulate_robot import Window


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = Window()
        self.addCleanup(self.window.close)
        self.panel = self.window.psc_panel

    def test_boot_attaches_same_window_and_stop_restores_controller(self):
        w, panel = self.window, self.panel
        original = w.th.robot
        windows = set(self.app.topLevelWidgets())
        with tempfile.TemporaryDirectory() as tmp, socket.socket(socket.AF_UNIX) as server:
            endpoint = str(Path(tmp) / 'robot.sock')
            server.bind(endpoint)
            server.listen(1)
            server.settimeout(3)
            for _ in range(2):
                with patch.object(panel.process, 'start') as start:
                    panel.start()
                args = start.call_args.args[1]
                self.assertIn('--external-robot', args)
                self.assertNotIn('--gui', args)
                panel.consume_output('PSC-ONE robot sock')
                self.assertIsNone(w.psc_worker)
                panel.consume_output('et: ' + endpoint + '\n')
                peer, _ = server.accept()
                with peer:
                    peer.settimeout(3)
                    frame = receive_message(peer)
                    validate_frame(frame, 0)
                    self.assertFalse(w.th.isRunning())
                    self.assertIs(w.psc_worker.plant.data, w.data)
                    self.assertEqual(set(self.app.topLevelWidgets()), windows)
                    send_message(peer, dict(version=1, sequence=0, output=4))
                    validate_frame(receive_message(peer), 1)
                    self.assertAlmostEqual(w.data.time, .005)
                    self.assertEqual(w.th.control_status.controller, 'PSC-ONE / PSC-OS')
                    # Duplicate readiness must not replace the worker.
                    worker = w.psc_worker
                    panel.consume_output('PSC-ONE robot socket: ' + endpoint + '\n')
                    self.assertIs(w.psc_worker, worker)
                    panel.stop()  # interrupts the pending second response
                    self.assertFalse(worker.isRunning())
                    panel.finished(130, None)
                self.assertIsNone(w.psc_worker)
                self.assertIs(w.th.robot, original)
                self.assertTrue(w.th.isRunning())
                self.assertTrue(w.controller_selector.isEnabled())
                self.assertTrue(w.training_tabs.isEnabled())
                self.assertGreater(w.speed_slider.maximum(), 508)

    def test_cancel_ignores_late_boot_notification(self):
        self.panel.stop()
        with patch('psc_protocol.RobotLink') as connect:
            self.panel.consume_output('PSC-ONE robot socket: /unused.sock\n')
        connect.assert_not_called()
        self.assertIsNone(self.window.psc_worker)
        self.assertTrue(self.window.th.isRunning())

    def test_shutdown_does_not_restart_local_thread(self):
        w, panel = self.window, self.panel
        with tempfile.TemporaryDirectory() as tmp, socket.socket(socket.AF_UNIX) as server:
            endpoint = str(Path(tmp) / 'robot.sock')
            server.bind(endpoint)
            server.listen(1)
            panel.attach_robot(endpoint)
            peer, _ = server.accept()
            with peer:
                peer.settimeout(3)
                receive_message(peer)
                panel.shutdown()
        self.assertIsNone(w.psc_worker)
        self.assertFalse(w.th.isRunning())


if __name__ == '__main__':
    unittest.main()
