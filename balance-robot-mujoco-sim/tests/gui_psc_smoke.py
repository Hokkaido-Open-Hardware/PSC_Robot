"""Real desktop smoke: start, wait for an RTL-controlled robot tick, stop, restart.
PSC_COCOTB_PYTHON must name the Python environment containing cocotb.
"""
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'balance-robot-mujoco-sim/src/simulation'))
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication
from simulate_robot import Window

if len(sys.argv) > 1:
    os.environ['PSC_COCOTB_PYTHON'] = sys.argv[1]
app = QApplication([])
window = Window()
window.show()
panel = window.psc_panel


def wait_for(predicate, seconds):
    deadline = time.monotonic() + seconds
    while not predicate():
        app.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError('GUI timeout: ' + panel.status.text())
        time.sleep(.02)
    app.processEvents()


try:
    log = ROOT / 'build/psc-one/run.log'
    old_mtime = log.stat().st_mtime_ns if log.exists() else 0
    panel.start_button.click()
    wait_for(lambda: panel.process.state() == QProcess.Running, 10)
    wait_for(lambda: log.exists() and log.stat().st_mtime_ns != old_mtime
             and 'robot tick=0' in log.read_text(errors='replace'), 900)
    assert not panel.start_button.isEnabled()
    assert panel.stop_button.isEnabled()
    wait_for(lambda: 'RX:' in window.i2c_dump.text.toPlainText(), 5)
    dump = window.i2c_dump.text.toPlainText()
    assert 'TX:' in dump and 'reg=0x08' in dump, dump
    assert not window.th.isRunning(), 'Local controller must stop during PSC-OS control'
    assert window.psc_worker is not None and window.psc_worker.isRunning()
    assert window.th.control_status.controller == 'PSC-ONE / PSC-OS'
    assert [w for w in app.topLevelWidgets() if isinstance(w, Window)] == [window]
    window.grab().save(str(ROOT / 'build/psc-gui.png'))
    panel.stop_button.click()
    wait_for(lambda: panel.process.state() == QProcess.NotRunning, 15)
    (ROOT / 'build/psc-gui-connected.log').write_text(log.read_text(errors='replace'))
    assert panel.start_button.isEnabled()
    assert window.psc_worker is None
    assert window.th.isRunning(), 'Local controller must resume after stopping PSC-OS'
    # Immediate cancellation while building/booting must also collect children.
    panel.start_button.click()
    wait_for(lambda: panel.process.state() == QProcess.Running, 10)
    time.sleep(.3)
    panel.stop_button.click()
    wait_for(lambda: panel.process.state() == QProcess.NotRunning, 15)
    print('PASS: GUI start / RTL robot tick / I2C dump / stop / restart / early stop')
finally:
    window.close()
    app.processEvents()
