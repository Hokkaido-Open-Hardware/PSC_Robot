"""Real-display RL smoke: render, train, continue, stop/save, load/switch, close."""
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simulation'))
from simulate_robot import Window

app = QApplication([])
work = tempfile.TemporaryDirectory(prefix='psc-rl-gui-')
window = Window()
window.training_tabs.setCurrentIndex(1)
panel = window.rl_panel
panel.model_path = Path(work.name) / 'training.rl.zip'
panel.steps.setValue(2048)
window.show()
started = time.monotonic()
phase = 'start'
errors = []
resets = 0
first_predictor = None
phase_started = started
finished = False


def fail(kind, error, traceback):
    errors.append(str(error))
    sys.__excepthook__(kind, error, traceback)
    window.close()


sys.excepthook = fail


def tick():
    global phase, resets, first_predictor, phase_started, finished
    assert time.monotonic() - started < 90, f'Timed out in phase {phase}: {panel.message.text()}'
    assert window.th.isRunning()
    assert panel.status.get('state') != 'Failed', panel.message.text()
    if phase == 'start':
        assert window.th.robot.controller_name == 'LQR'
        assert not window.controller_selector.model().item(2).isEnabled()
        panel.start_button.click()
        phase = 'training'
    elif phase == 'training':
        if panel.process is not None:
            assert not panel.start_button.isEnabled()
            window.controller_selector.setCurrentIndex(1 if window.nn_predictor and resets % 2 else 0)
            window.speed_slider.setValue(200)
            window.yaw_slider.setValue(500)
            assert window.th.speed == .2 and window.th.yaw == .5
            window.reset_simulation()
            resets += 1
        if panel.predictor is not None and panel.loader is None:
            assert panel.status['steps'] == 2048
            assert panel.status['episodes'] > 0
            assert panel.status['mean_duration'] > 0
            assert panel.status['fps'] > 0
            assert resets > 2
            assert window.controller_selector.model().item(2).isEnabled()
            window.controller_selector.setCurrentIndex(2)
            assert window.speed_slider.minimum() == -400
            assert window.speed_slider.maximum() == 400
            window.reset_simulation()
            first_predictor = panel.predictor
            first_predictor.save(Path(work.name) / 'snapshot.rl.zip')
            phase = 'inference'
            phase_started = time.monotonic()
    elif phase == 'inference' and time.monotonic() - phase_started > .3:
        assert window.th.control_status.controller == 'Reinforcement Learning'
        assert window.th.control_status.pitch is not None
        window.speed_slider.setValue(100)
        window.yaw_slider.setValue(300)
        panel.steps.setValue(100000)
        panel.stage.setCurrentText('velocity')
        panel.resume_button.click()
        phase = 'continuing'
    elif phase == 'continuing':
        assert window.rl_predictor is first_predictor
        if panel.status.get('steps', 0) > 2200:
            panel.stop_button.click()
            phase = 'stopping'
    elif phase == 'stopping' and panel.process is None and panel.loader is None:
        assert panel.status['state'] == 'Stopped'
        assert panel.predictor is not first_predictor
        assert panel.predictor.metadata['training']['stage'] == 'velocity'
        assert panel.status['steps'] > 2048
        panel.load_model(Path(work.name) / 'snapshot.rl.zip')
        phase = 'loading'
    elif phase == 'loading' and panel.loader is None:
        assert panel.predictor.metadata['training']['steps'] == 2048
        assert window.rl_predictor is panel.predictor
        first_predictor = panel.predictor
        bad = Path(work.name) / 'bad.rl.zip'
        bad.write_bytes(b'not a checkpoint')
        panel.load_model(bad)
        phase = 'bad-load'
    elif phase == 'bad-load' and panel.loader is None:
        assert panel.predictor is first_predictor
        assert 'Load failed' in panel.message.text()
        frame = window.viewport.grabFramebuffer()
        assert not frame.isNull()
        assert np.frombuffer(frame.constBits(), dtype=np.uint8).std() > 10
        frame.save('/tmp/psc-rl-viewport-smoke.png')
        window.grab().save('/tmp/psc-rl-gui-smoke.png')
        assert len(window.viewport.runtime) > 10
        panel.start_button.click()
        phase = 'closing'
    elif phase == 'closing' and panel.status.get('state') == 'Training':
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
assert panel.process is None or panel.process.state() == panel.process.NotRunning
assert panel.loader is None or not panel.loader.isRunning()
print(f'PASS: RL GUI training/continue/stop/save/load/switch/close; {resets} resets, {len(window.viewport.runtime)} frames')
work.cleanup()
