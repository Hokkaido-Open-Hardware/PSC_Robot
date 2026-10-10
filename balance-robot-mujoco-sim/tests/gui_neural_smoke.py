"""Real-display smoke test: train, switch controllers, reset, render and close.

Run from the repository root: myenv/bin/python -B balance-robot-mujoco-sim/tests/gui_neural_smoke.py
All test training artifacts are written to a temporary directory.
"""
import faulthandler
from pathlib import Path
import sys
import tempfile
import time
import numpy as np

faulthandler.enable()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'simulation'))
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QPushButton
from simulate_robot import Window

app = QApplication([])
window = Window()
work = tempfile.TemporaryDirectory(prefix='psc-nn-gui-')
window.training_panel.model_path = Path(work.name) / 'test_nn.npz'
window.training_panel.episodes.setValue(20)
window.training_panel.epochs.setValue(40)
window.show()
errors = []
resets = 0
switches = 0
finished = False
started = time.monotonic()
initial_frames = 0
seen_scenarios = set()
button = next(b for b in window.findChildren(QPushButton) if b.text() == 'Reset')


def fail(kind, error, traceback):
    errors.append(str(error))
    sys.__excepthook__(kind, error, traceback)
    window.close()


sys.excepthook = fail


def begin():
    global initial_frames
    assert window.th.control_status.controller == 'LQR'
    assert window.nn_predictor is not None, 'Run neural_training.py first'
    initial_frames = len(window.viewport.runtime)
    window.training_panel.train_button.click()
    assert not window.training_panel.train_button.isEnabled()


def tick():
    global resets, switches, finished
    if time.monotonic() - started > 90:
        raise AssertionError('Training/GUI timed out')
    if window.training_panel.started is not None:
        window.controller_selector.setCurrentIndex(1 - window.controller_selector.currentIndex())
        switches += 1
        window.speed_slider.setValue(200)
        window.yaw_slider.setValue(500)
        assert window.th.speed == .2 and window.th.yaw == .5
        status = window.training_panel.status
        if status.get('scenario'):
            seen_scenarios.add(status['scenario'])
        if status['state'] == 'Collecting Data' and status.get('current_episode'):
            window.training_panel.refresh()
            assert window.training_panel.labels['episodes'].text() == f"{status['current_episode']} / 20"
            assert window.training_panel.labels['auto_speed'].text() == f"{status['auto_speed']:.3f}"
            assert window.training_panel.labels['auto_yaw'].text() == f"{status['auto_yaw']:.3f}"
            assert window.training_panel.labels['scenario'].text() == status['scenario']
            assert window.training_panel.labels['falls'].text() == str(status['falls'])
            assert window.training_panel.labels['episode_elapsed'].text() == f"{status['episode_elapsed']:.2f} s"
        button.click()
        resets += 1
        assert window.th.isRunning()
    if window.training_panel.status['state'] == 'Failed':
        raise AssertionError(window.training_panel.message.text())
    if window.training_panel.status['state'] == 'Completed' and window.training_panel.process is None:
        poll.stop()
        assert window.training_panel.model_path.exists()
        assert window.training_panel.model_path.with_suffix('.pt').exists()
        assert len(window.training_panel.plot.points) == 40
        assert window.training_panel.progress.value() == 1000
        assert window.training_panel.status['current_episode'] == 20
        assert len(seen_scenarios) == 9, seen_scenarios
        assert len(window.viewport.runtime) > initial_frames + 10
        assert resets > 5 and switches > 5
        window.controller_selector.setCurrentIndex(1)
        button.click()
        QTimer.singleShot(300, verify)


def verify():
    global finished
    assert window.th.control_status.controller == 'Neural Network'
    assert window.th.control_status.pitch is not None
    assert window.th.isRunning()
    window.grab().save('/tmp/psc-nn-gui-smoke.png')
    frame = window.viewport.grabFramebuffer()
    assert not frame.isNull(), 'Empty OpenGL framebuffer'
    pixels = np.frombuffer(frame.constBits(), dtype=np.uint8)
    assert pixels.std() > 10, 'OpenGL framebuffer has no visible scene'
    frame.save('/tmp/psc-nn-viewport-smoke.png')
    print(f'PASS: GUI training completed; {resets} resets, {switches} controller switches, '
          f'{len(window.viewport.runtime)} rendered frames; NN resumed', flush=True)
    finished = True
    # Also exercise closing while a new training process is running.
    window.training_panel.train_button.click()
    QTimer.singleShot(300, window.close)


poll = QTimer()
poll.setInterval(50)
poll.timeout.connect(tick)
QTimer.singleShot(500, begin)
poll.start()
app.exec()
assert not errors, errors
assert finished
assert not window.th.isRunning()
work.cleanup()
