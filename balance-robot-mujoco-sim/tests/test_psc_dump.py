"""Live log lifecycle and failed-transfer visibility, without a desktop."""
import asyncio
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
from psc_dump import I2CDumpPanel
from i2c_trace import I2CTraceMixin


class DumpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_live_append_partial_line_clear_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'i2c.log'
            panel = I2CDumpPanel(path=path)
            panel.timer.stop()
            try:
                path.write_bytes(b'TX:84[ACK]\nRX:F6')
                panel.read_trace()
                self.assertEqual(panel.text.toPlainText(), 'TX:84[ACK]')
                with path.open('ab') as log:
                    log.write(b'[NACK]\n')
                panel.read_trace()
                self.assertIn('RX:F6[NACK]', panel.text.toPlainText())
                panel.clear_button.click()
                panel.read_trace()
                self.assertEqual(panel.text.toPlainText(), '')
                with path.open('ab') as log:
                    log.write(b'new transfer\n')
                panel.read_trace()
                self.assertEqual(panel.text.toPlainText(), 'new transfer')
                # A replacement can be larger than the old file: size alone
                # does not detect a new run.
                replacement = path.with_suffix('.new')
                replacement.write_text('new run\n' * 20)
                replacement.replace(path)
                panel.read_trace()
                self.assertEqual(panel.text.toPlainText(), ('new run\n' * 20).rstrip())
                path.write_text('truncated\n')
                panel.read_trace()
                self.assertEqual(panel.text.toPlainText(), 'truncated')
            finally:
                panel.close()

    def test_history_is_bounded_and_can_stop_following(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'i2c.log'
            path.write_text(''.join(f'transfer {i}\n' for i in range(2500)))
            panel = I2CDumpPanel(path=path)
            panel.timer.stop()
            try:
                self.assertEqual(panel.text.document().blockCount(), 2000)
                self.assertTrue(panel.text.toPlainText().startswith('transfer 500\n'))
                panel.follow.setChecked(False)
                panel.text.verticalScrollBar().setValue(10)
                position = panel.text.verticalScrollBar().value()
                with path.open('a') as log:
                    log.write('next\n')
                panel.read_trace()
                self.assertEqual(panel.text.verticalScrollBar().value(), position)
            finally:
                panel.close()


class TraceTests(unittest.TestCase):
    def test_real_master_transactions_and_nack_are_preserved(self):
        # Reuse the actual master register/byte routines with an in-memory
        # bit transport. This checks logging without simulating RTL here.
        import ast
        source = ROOT / 'external/PSC-ONE/hardware/sim/cocotb_tb/mmio/psc_rv32_i2c.py'
        tree = ast.parse(source.read_text())
        master_class = next(n for n in tree.body
                            if isinstance(n, ast.ClassDef) and n.name == 'I2CMaster')
        namespace = {'I2C_ADDR': 0x42}
        exec(compile(ast.Module(body=[master_class], type_ignores=[]),
                     str(source), 'exec'), namespace)

        class BitMaster(namespace['I2CMaster']):
            async def start(self):
                pass

            async def stop(self):
                pass

            async def write_bit(self, bit):
                pass

            async def read_bit(self):
                return next(self.bits)

            def sda(self, low):
                pass

        class Master(I2CTraceMixin, BitMaster):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace'
            master = Master(None)
            master.configure_trace(path, lambda: 123000)
            master.bits = iter([0] * 6)
            asyncio.run(master.write_registers(0, [10, -1, -128, 127]))
            master.bits = iter([0] * 3 + [int(b) for b in '1111011000000010'])
            self.assertEqual(asyncio.run(master.read_registers(8, 2)), [246, 2])
            master.bits = iter([1])
            with self.assertRaisesRegex(AssertionError, r'SLA\+W NACK'):
                asyncio.run(master.write_registers(0, [0]))
            log = path.read_text()
            self.assertIn('WRITE addr=0x42 reg=0x00', log)
            self.assertIn('TX:0A[ACK] TX:FF[ACK] TX:80[ACK] TX:7F[ACK] STOP', log)
            self.assertIn('START TX:85[ACK] RX:F6[ACK] RX:02[NACK] STOP', log)
            self.assertIn('TX:84[NACK] ERROR: SLA+W NACK', log)


if __name__ == '__main__':
    unittest.main()
