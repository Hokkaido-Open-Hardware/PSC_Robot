"""GUI lifecycle controls; IPC is always outside the MuJoCo/render lock."""
import os
from pathlib import Path
import sys

from PySide6.QtCore import QProcess, QThread, Signal
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QPushButton, QLabel, QLineEdit

ROOT = Path(__file__).resolve().parents[3]


class PSCPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__('PSC-ONE integration', parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        layout = QVBoxLayout(self)
        self.python = QLineEdit(os.getenv('PSC_COCOTB_PYTHON', sys.executable))
        self.python.setPlaceholderText('Python with cocotb 2.0 installed')
        layout.addWidget(QLabel('cocotb Python'))
        layout.addWidget(self.python)
        self.start_button = QPushButton('Start PSC-ONE / PSC-OS')
        self.stop_button = QPushButton('Stop PSC-ONE')
        self.stop_button.setEnabled(False)
        self.status = QLabel('Opens a separate lockstep robot window. Simulation pin adapter.')
        self.status.setWordWrap(True)
        for widget in (self.start_button, self.stop_button, self.status):
            layout.addWidget(widget)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.failed)

    def start(self):
        if self.process.state() != QProcess.NotRunning:
            return
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.python.setEnabled(False)
        self.status.setText('Building/booting PSC-OS; details: build/psc-one/run.log')
        self.process.start(sys.executable, [str(ROOT / 'integration/psc_one/run.py'),
                           '--gui', '--steps', '1000000', '--timeout', '86400',
                           '--sim-python', self.python.text()])

    def stop(self):
        self.status.setText('Stopping and collecting child processes…')
        self.process.terminate()  # supervisor handles the entire process group

    def read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode(errors='replace').strip()
        if text:
            self.status.setText(text[-1500:])

    def failed(self, error):
        self.status.setText(self.process.errorString())
        if self.process.state() == QProcess.NotRunning:
            self.finished(-1, None)

    def finished(self, code, status):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.python.setEnabled(True)
        self.status.setText(self.status.text() + f'\nExited: {code}')

    def shutdown(self):
        if self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            # run.py has a bounded TERM/KILL/reap sequence (<9 seconds).
            self.process.waitForFinished(10000)


class PSCThread(QThread):
    stateChanged = Signal(str)

    def __init__(self, window, link, steps):
        super().__init__(window)
        from psc_worker import RobotPlant
        self.window, self.link, self.steps = window, link, steps
        self.running = True
        self.failed = False
        self.plant = RobotPlant(window.model, window.data)

    def run(self):
        try:
            for _ in range(self.steps):
                if not self.running:
                    break
                with self.window.data_lock:
                    values = self.plant.sample()
                output = self.link.exchange(values)
                with self.window.data_lock:
                    if not self.running:
                        break
                    self.plant.advance(output)
                    self.window.th.robot = self.plant.sensors
                self.stateChanged.emit(f'PSC-OS connected • t={self.link.sequence * .005:.3f}s')
        except Exception as exc:
            if self.running:
                self.failed = True
                print(f'PSC-ONE robot failed: {exc}', file=sys.stderr, flush=True)
                self.stateChanged.emit(f'PSC-ONE disconnected: {exc}')
        finally:
            with self.window.data_lock:
                self.window.data.ctrl[:] = 0
            self.link.close()

    def stop(self):
        self.running = False
        self.link.close()  # interrupts a blocking recv immediately
        self.wait(5000)


def run_robot_gui(link, steps):
    from PySide6.QtWidgets import QApplication, QLabel
    from simulate_robot import Window
    app = QApplication.instance() or QApplication([])
    window = Window(psc_child=True)
    window.setWindowTitle('PSC-ONE / PSC-OS • simulation pin adapter')
    worker = PSCThread(window, link, steps)
    window.psc_worker = worker
    label = QLabel('Waiting for PSC-OS')
    window.statusBar().addPermanentWidget(label)
    stop = QPushButton('Stop PSC-ONE')
    stop.clicked.connect(window.close)
    window.statusBar().addPermanentWidget(stop)
    worker.stateChanged.connect(label.setText)
    worker.finished.connect(window.close)
    # This first protocol supports balancing only (target=0, yaw=0).
    # Freeze local controller/reset/training controls during a remote session.
    for widget in (window.controller_selector, window.command_mode,
                   window.speed_slider, window.yaw_slider, window.training_tabs,
                   window.reset_button):
        widget.setEnabled(False)
    window.show()
    worker.start()
    result = app.exec()
    return 1 if worker.failed else result
