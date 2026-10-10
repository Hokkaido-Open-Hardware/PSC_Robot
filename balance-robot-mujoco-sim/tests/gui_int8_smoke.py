"""Real-display INT8 control, controller switching, AUTO, reset and rendering."""
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simulation'))
from simulate_robot import Window
from rl_model import RLPredictor, DEFAULT_RL_MODEL

app = QApplication([])
window = Window()
window.show()
errors = []
completed = []
modes = ['NN INT8', 'Neural Network', 'LQR', 'Reinforcement Learning', 'NN INT8']


def fail(kind, error, traceback):
    errors.append(str(error))
    sys.__excepthook__(kind, error, traceback)
    window.close()


sys.excepthook = fail


def switch():
    assert window.int8_predictor is not None, window.int8_status.text()
    if window.rl_predictor is None:
        window.install_rl_model(RLPredictor(DEFAULT_RL_MODEL))
    if completed:
        assert window.th.control_status.controller == completed[-1]
        assert window.th.control_status.pitch is not None
        assert window.th.isRunning()
    if not modes:
        frame = window.viewport.grabFramebuffer()
        assert not frame.isNull()
        assert np.frombuffer(frame.constBits(), dtype=np.uint8).std() > 10
        frame.save('/tmp/psc-int8-viewport.png')
        window.grab().save('/tmp/psc-int8-gui.png')
        print('PASS: real TFLite INT8, FP32, LQR, RL, AUTO, reset and OpenGL rendering', flush=True)
        window.close()
        return
    name = modes.pop(0)
    window.controller_selector.setCurrentText(name)
    window.command_mode.setCurrentText('AUTO')
    window.reset_simulation()
    completed.append(name)
    QTimer.singleShot(1200, switch)


QTimer.singleShot(500, switch)
QTimer.singleShot(30000, window.close)
app.exec()
assert not errors, errors
assert len(completed) == 5 and not modes
assert not window.th.isRunning()
