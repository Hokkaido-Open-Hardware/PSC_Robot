"""OS state transitions and streaming build/UART output without a desktop."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'balance-robot-mujoco-sim/src/simulation'))
sys.path.insert(0, str(ROOT / 'integration/psc_one'))
from psc_panel import PSCPanel
from os_state import memset_range
from uart_output import UARTLines


class StatusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_fragmented_state_messages_and_stop(self):
        panel = PSCPanel()
        try:
            panel.consume_output('compiler output\n10200ns INFO PSC-OS sta')
            self.assertIn('not running', panel.os_state.text())
            panel.consume_output('te: memset (PC=0x00206810)\n')
            self.assertIn('memset 実行中', panel.os_state.text())
            panel.consume_output('[PSC-OS UART] hello\n')
            self.assertIn('memset 実行中', panel.os_state.text())
            panel.consume_output('INFO PSC-OS state: running (outside memset)\n')
            self.assertNotIn('実行中', panel.os_state.text())
            panel.consume_output('INFO PSC-OS state: memset (PC=0x00206810)\n')
            panel.finished(0, None)
            self.assertEqual(panel.os_state.text(), 'PSC-OS state: stopped')
        finally:
            panel.close()

    def test_map_wrapped_section_and_missing_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'kernel.map'
            for space in (' ', '\n                '):
                path.write_text('.text.memset' + space + '0x0020680c 0x20 object.o\n')
                self.assertEqual(memset_range(path), (0x20680c, 0x20682c))
            path.write_text('.text.memset_other 0x100 0x20 object.o\n')
            with self.assertRaises(ValueError):
                memset_range(path)

    def test_live_log_append_and_partial_lines(self):
        spec = importlib.util.spec_from_file_location(
            'psc_run', ROOT / 'integration/psc_one/run.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.log'
            path.write_text('compile\n')
            with path.open() as reader, contextlib.redirect_stdout(output):
                self.assertTrue(runner.forward_log(reader))
                self.assertFalse(runner.forward_log(reader))
                with path.open('a') as writer:
                    writer.write('[PSC-OS UART] PSC_OS> ')
                self.assertTrue(runner.forward_log(reader))
                self.assertFalse(runner.forward_log(reader))
        self.assertEqual(output.getvalue(), 'compile\n[PSC-OS UART] PSC_OS> ')

    def test_uart_crlf_utf8_prompt_and_bounded_lines(self):
        output = []
        uart = UARTLines(output.append)
        for value in '起動\r\nPSC_OS> '.encode():
            uart.feed(value)
        self.assertEqual(output, ['起動'])
        uart.flush()
        self.assertEqual(output, ['起動', 'PSC_OS> '])
        uart.flush()
        self.assertEqual(len(output), 2)
        for _ in range(513):
            uart.feed(ord('a'))
        self.assertEqual(output[-1], 'a' * 512)
        self.assertEqual(uart.pending, 'a')


if __name__ == '__main__':
    unittest.main()
