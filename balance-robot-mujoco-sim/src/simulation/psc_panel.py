"""GUI lifecycle controls; IPC is always outside the MuJoCo/render lock."""
import codecs
import os
from pathlib import Path
import sys
import time

from PySide6.QtCore import QProcess, QThread, Signal
from PySide6.QtWidgets import QGroupBox, QVBoxLayout, QPushButton, QLabel, QLineEdit

ROOT = Path(__file__).resolve().parents[3]


class PSCPanel(QGroupBox):
    def __init__(self, parent=None):
        super().__init__('PSC-ONE integration', parent)
        self.robot_window = parent
        self.local_robot = None
        self.stopping = False
        self.closing = False
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
        self.status = QLabel('Connects this robot to PSC-OS. Simulation pin adapter.')
        self.status.setWordWrap(True)
        self.os_state = QLabel('PSC-OS state: not running')
        self.os_state.setWordWrap(True)
        self._output_pending = ''
        self._output_decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        for widget in (self.start_button, self.stop_button, self.os_state, self.status):
            layout.addWidget(widget)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.failed)

    def start(self):
        if self.process.state() != QProcess.NotRunning:
            return
        self.stopping = False
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.python.setEnabled(False)
        self.status.setText('Building/booting PSC-OS; details: build/psc-one/run.log')
        self.os_state.setText('PSC-OS state: building / booting')
        self._output_pending = ''
        self._output_decoder.reset()
        self.process.start(sys.executable, [str(ROOT / 'integration/psc_one/run.py'),
                           '--external-robot', '--timing', 'interactive', '--steps', '1000000', '--timeout', '86400',
                           '--sim-python', self.python.text()])

    def stop(self):
        self.stopping = True
        self.status.setText('Stopping and collecting child processes…')
        self.process.terminate()  # supervisor handles the entire process group
        if self.robot_window is not None and self.robot_window.psc_worker is not None:
            self.robot_window.psc_worker.stop()

    def attach_robot(self, endpoint):
        window = self.robot_window
        if self.stopping or self.closing or window is None or window.psc_worker is not None:
            return
        from psc_protocol import RobotLink
        link = None
        try:
            link = RobotLink(endpoint)
            window.th.stop()
            self.local_robot = window.th.robot
            self.yaw_tooltip = window.yaw_slider.toolTip()
            worker = PSCThread(window, link, 1000000)
            window.psc_worker = worker
            window.reset_simulation()
            worker.stateChanged.connect(self.status.setText)
            worker.start()
        except Exception as exc:
            if link is not None:
                link.close()
            self.stop()
            self.status.setText(f'PSC-ONE connection failed: {exc}')

    def detach_robot(self):
        window = self.robot_window
        if window is None:
            return
        if window.psc_worker is not None:
            window.psc_worker.stop()
            window.psc_worker.deleteLater()
            window.psc_worker = None
        if self.local_robot is not None:
            window.th.robot = self.local_robot
            self.local_robot = None
            window.controller_selector.setEnabled(True)
            window.training_tabs.setEnabled(True)
            window.update_speed_range()
            window.yaw_slider.setToolTip(self.yaw_tooltip)
            window.reset_simulation()
            if not self.closing:
                window.th.running = True
                window.th.start()

    def read_output(self):
        if not self.process.isOpen():
            return
        text = self._output_decoder.decode(bytes(self.process.readAllStandardOutput()))
        if text:
            # QProcess captures stdout; also show it in the launching terminal.
            if sys.stdout is not None:
                sys.stdout.write(text)
                sys.stdout.flush()
            self.consume_output(text)

    def consume_output(self, text):
        self.status.setText(text.strip()[-1500:])
        lines = (self._output_pending + text).split('\n')
        self._output_pending = lines.pop()[-4096:]
        for line in lines:
            socket_marker = 'PSC-ONE robot socket: '
            if line.startswith(socket_marker):
                self.attach_robot(line[len(socket_marker):].strip())
            marker = 'PSC-OS state: '
            if marker in line:
                state = line.split(marker, 1)[1].strip()
                if state.startswith('memset '):
                    state = state.replace('memset ', 'memset 実行中 ', 1)
                self.os_state.setText(marker + state)

    def failed(self, error):
        self.status.setText(self.process.errorString())
        if self.process.state() == QProcess.NotRunning:
            self.finished(-1, None)

    def finished(self, code, status):
        self.stopping = True
        self.read_output()
        self.detach_robot()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.python.setEnabled(True)
        self.status.setText(self.status.text() + f'\nExited: {code}')
        self.os_state.setText('PSC-OS state: stopped')

    def shutdown(self):
        self.closing = True
        self.stopping = True
        if self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            # run.py has a bounded TERM/KILL/reap sequence (<9 seconds).
            self.process.waitForFinished(10000)
        self.detach_robot()


class PSCThread(QThread):
    stateChanged = Signal(str)

    def __init__(self, window, link, steps):
        super().__init__(window)
        from psc_worker import RobotPlant
        self.window, self.link, self.steps = window, link, steps
        self.running = True
        self.failed = False
        with window.data_lock:
            self.plant = RobotPlant(window.model, window.data)
            window.th.robot = self.plant.sensors

        window.speed_slider.setRange(-512, 508)
        window.speed_slider.setToolTip('PSC-OS target speed: −0.512 to +0.508 m/s (I2C INT8).')
        window.yaw_slider.setToolTip('Wheel yaw mixing applied on the robot side, after the PSC-OS balance output.')
        # The remote CPU owns balance control. Commands and reset stay interactive.
        for widget in (window.controller_selector, window.training_tabs):
            widget.setEnabled(False)

    def run(self):
        started = time.monotonic()
        completed = 0
        try:
            for _ in range(self.steps):
                if not self.running:
                    break
                with self.window.data_lock:
                    commands = self.window.th
                    commands._prepare_commands_locked()
                    # Wire target is INT8 at .004 m/s; keep displayed commands
                    # within the representable range, including AUTO randoms.
                    commands.speed = max(-.512, min(.508, commands.speed))
                    epoch = commands.command_epoch
                    values = self.plant.sample(commands.speed, commands.yaw)
                output = self.link.exchange(values)
                with self.window.data_lock:
                    if not self.running:
                        break
                    # AUTO/reset may have changed the plant during IPC. Consume
                    # that reply but never apply an old-state output after reset.
                    if epoch != commands.command_epoch:
                        continue
                    self.plant.advance(output)
                    commands.physics_steps += self.plant.substeps
                    completed += 1
                    elapsed = self.window.data.time
                rate = completed * .005 / max(time.monotonic() - started, .001)
                self.stateChanged.emit(f'PSC-OS connected • t={elapsed:.3f}s • {rate:.3f}× real time')
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
    window.show()
    worker.start()
    result = app.exec()
    return 1 if worker.failed else result
