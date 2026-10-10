"""Real-display AUTO smoke, including concurrent NN and RL training."""
from pathlib import Path
import sys
import tempfile
import time

import mujoco
import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simulation'))
from simulate_robot import Window

app = QApplication([])
work = tempfile.TemporaryDirectory(prefix='psc-auto-gui-')
window = Window()
nn, rl = window.training_panel, window.rl_panel
nn.model_path = Path(work.name) / 'nn.npz'
nn.episodes.setValue(18)
nn.epochs.setValue(8)
rl.model_path = Path(work.name) / 'rl.rl.zip'
rl.steps.setValue(1024)
window.show()
started = time.monotonic()
phase = 'start'
errors = []
finished = False


def fail(kind, error, traceback):
    errors.append(str(error))
    sys.__excepthook__(kind, error, traceback)
    poll.stop()
    window.close()


sys.excepthook = fail


def tick():
    global phase, finished
    assert time.monotonic() - started < 90, f'Timed out: {phase}'
    assert window.th.isRunning()
    assert nn.status.get('state') != 'Failed', nn.message.text()
    assert rl.status.get('state') != 'Failed', rl.message.text()
    status = window.th.auto_status
    if phase == 'start':
        assert window.command_mode.currentText() == 'MANUAL'
        window.command_mode.setCurrentText('AUTO')
        assert not window.speed_slider.isEnabled()
        assert not window.yaw_slider.isEnabled()
        assert window.th.auto_status.enabled
        nn.train_button.click()
        rl.start_button.click()
        phase = 'forward'
    elif phase == 'forward' and status.episode >= 2 and status.speed > .01:
        assert status.scenario == 'Gentle Forward'
        window.update_control_status()
        assert window.speed_slider.value() > 0
        assert 'AUTO' in window.auto_status_label.text()
        window.controller_selector.setCurrentIndex(1)
        with window.data_lock:
            # A sideways fall must advance the AUTO episode in the real worker.
            window.data.qpos[3:7] = [np.cos(.6), 0, np.sin(.6), 0]
            mujoco.mj_forward(window.model, window.data)
        phase = 'fall'
    elif phase == 'fall' and status.falls > 0:
        assert status.episode >= 3
        assert window.th.robot.controller_name == 'Neural Network'
        window.reset_simulation()
        assert window.th.auto_status.episode == 1
        assert window.th.auto_status.falls == 0
        assert window.th.auto_status.enabled
        phase = 'training'
    elif phase == 'training' and nn.process is None and rl.process is None and rl.loader is None:
        assert nn.status['state'] == 'Completed'
        assert rl.predictor is not None
        window.controller_selector.setCurrentIndex(2)
        window.reset_simulation()
        phase = 'rl'
    elif phase == 'rl' and window.th.control_status.pitch is not None:
        assert window.th.control_status.controller == 'Reinforcement Learning'
        assert window.th.auto_status.enabled
        window.command_mode.setCurrentText('MANUAL')
        assert window.speed_slider.isEnabled() and window.yaw_slider.isEnabled()
        assert window.speed_slider.value() == window.yaw_slider.value() == 0
        assert window.th.speed == window.th.yaw == 0
        window.speed_slider.setValue(200)
        window.yaw_slider.setValue(300)
        assert window.th.speed == .2 and window.th.yaw == .3
        window.command_mode.setCurrentText('AUTO')
        window.update_control_status()
        frame = window.viewport.grabFramebuffer()
        assert not frame.isNull()
        assert np.frombuffer(frame.constBits(), dtype=np.uint8).std() > 10
        assert len(window.viewport.runtime) > 10
        window.grab().save('/tmp/psc-auto-gui-smoke.png')
        finished = True
        poll.stop()
        window.close()


poll = QTimer()
poll.setInterval(100)
poll.timeout.connect(tick)
poll.start()
app.exec()
assert not errors, errors
assert finished
assert not window.th.isRunning()
print(f'PASS: AUTO commands/fall/reset/manual/LQR/NN/RL, concurrent training, '
      f'{len(window.viewport.runtime)} rendered frames')
work.cleanup()
